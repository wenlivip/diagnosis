from copy import deepcopy
import tensorflow as tf
import numpy as np
import os
import random
from tqdm import tqdm


class Model(object):

    def __init__(self, config):
        self.cfg = config
        self.vocab_size = config.vocab_size
        self.batch_size = config.batch_size
        self.max_len = config.max_len
        self.epochs = config.epochs
        self.build_model()

    def build_model(self):
        # placeholders
        self._sent_placeholder = tf.placeholder(tf.int32, shape=[None, None], name='sent_ph')
        self._targets_placeholder = tf.placeholder(tf.int32, shape=[None, None], name='labels_ph')
        self._img_placeholder = tf.placeholder(tf.float32, shape=[None, self.cfg.img_dim], name='img_ph')
        # self._sequence_length = tf.placeholder(tf.float64, shape=[None, None], name='sequence_ph')
        sequence = tf.cast(tf.equal(tf.equal(self._targets_placeholder, 0), False), tf.float32)
        self._class_placeholder = tf.placeholder(tf.int32, shape=[None], name='class_ph')
        self._dropout_placeholder = tf.placeholder(tf.float32, name='dropout_ph')

        # Input layer
        with tf.variable_scope('feature_embeding'):
            W_i = tf.get_variable('W_i', shape=[self.cfg.img_dim, self.cfg.embed_dim])
            b_i = tf.get_variable('b_i', shape=[self.cfg.batch_size, self.cfg.embed_dim])
            img_input = tf.expand_dims(tf.nn.sigmoid(tf.matmul(self._img_placeholder, W_i) + b_i), 1)
            img_input = tf.tile(img_input, [1, self.cfg.max_len, 1])
            print 'Img:', img_input.get_shape()
        with tf.variable_scope('sent_embeding'):
            word_embeddings = tf.get_variable(name='word_embeddings', shape=[self.vocab_size, self.cfg.embed_dim])
            sent_inputs = tf.nn.embedding_lookup(word_embeddings, self._sent_placeholder)
            print 'Sent:', sent_inputs.get_shape()
        with tf.variable_scope('class_embeding'):
            class_embeddings = tf.get_variable('class_embeddings', shape=[self.cfg.class_nums, self.cfg.hidden_dim])
            class_inputs = tf.nn.embedding_lookup(class_embeddings, self._class_placeholder)
            class_inputs = tf.tile(tf.expand_dims(class_inputs, axis=1), [1, self.cfg.max_len, 1])
            print 'Class:', class_inputs.get_shape()
        with tf.variable_scope('all_embedin'):
            all_inputs = tf.concat([img_input, sent_inputs], axis=2)

        # LSTM layer
        lstm = tf.nn.rnn_cell.BasicLSTMCell(self.cfg.hidden_dim)
        lstm_dropout = tf.nn.rnn_cell.DropoutWrapper(lstm, input_keep_prob=self.cfg.keep_prob, output_keep_prob=self.cfg.keep_prob)

        w_hidden = tf.get_variable('w_hidden', [self.cfg.img_dim, self.cfg.hidden_dim], initializer=tf.contrib.layers.xavier_initializer())
        b_hidden = tf.get_variable('b_hidden', [self.cfg.hidden_dim], initializer=tf.constant_initializer(0.0))
        init_h = tf.nn.tanh(tf.matmul(self._img_placeholder, w_hidden) + b_hidden)

        w_cell = tf.get_variable('w_cell', [self.cfg.img_dim, self.cfg.hidden_dim], initializer=tf.contrib.layers.xavier_initializer())
        b_cell = tf.get_variable('b_cell', [self.cfg.hidden_dim], initializer=tf.constant_initializer(0.0))
        init_c = tf.nn.tanh(tf.matmul(self._img_placeholder, w_cell) + b_cell)

        state = (init_c, init_h)

        hiddens = []
        cells_states = []
        for time_step in range(all_inputs.get_shape()[1]):
            inp = all_inputs[:, time_step, :]
            hidden, state = lstm_dropout(inp, state)
            cells_states.append(state[0])
            hiddens.append(state[1])

        hiddens = tf.transpose(hiddens, [1, 0, 2])
        cells_states = tf.transpose(cells_states, [1, 0, 2])

        # sentinel
        sentinel_wx = tf.get_variable(shape=[self.cfg.embed_dim*2, self.cfg.hidden_dim], dtype=tf.float32, name='sentinel_wx')
        sentinel_wh = tf.get_variable(shape=[self.cfg.hidden_dim, self.cfg.hidden_dim], dtype=tf.float32, name='sentinel_wh')
        h_t0 = tf.zeros((self.batch_size, 1, self.cfg.hidden_dim))
        h_t_1 = hiddens[:, :-1, :]
        h = tf.concat([h_t0, h_t_1], axis=1)

        x = tf.reshape(all_inputs, (-1, self.cfg.embed_dim*2))
        h = tf.reshape(h, (-1, self.cfg.hidden_dim))
        gate_t = tf.matmul(x, sentinel_wx) + tf.matmul(h, sentinel_wh)
        sentinel = gate_t * tf.tanh(tf.reshape(cells_states, [-1, self.cfg.hidden_dim]))
        print('Sentinel:{}'.format(sentinel.get_shape()))

        # atten
        atten_ws = tf.get_variable(name='atten_ws', shape=(self.cfg.hidden_dim, self.cfg.hidden_dim), dtype=tf.float32)
        atten_wc = tf.get_variable(name='atten_wc', shape=(self.cfg.hidden_dim, self.cfg.hidden_dim), dtype=tf.float32)
        atten_wh = tf.get_variable(name='atten_wh', shape=(self.cfg.hidden_dim, self.cfg.hidden_dim), dtype=tf.float32)
        atten_wg = tf.get_variable(name='atten_wg', shape=(self.cfg.hidden_dim, 1), dtype=tf.float32)

        zt = tf.matmul(tf.tanh(tf.matmul(sentinel, atten_ws) + tf.matmul(tf.reshape(hiddens, [-1, self.cfg.hidden_dim]), atten_wh)), atten_wg)
        zt_extended = tf.matmul(tf.tanh(tf.matmul(tf.reshape(class_inputs, [-1, self.cfg.hidden_dim]), atten_wc) + tf.matmul(tf.reshape(hiddens, [-1, self.cfg.hidden_dim]), atten_wh)), atten_wg)
        print('zt:{}, zt_ext:{}'.format(zt.get_shape(), zt_extended.get_shape()))   # sentinel, class
        extended = tf.concat([zt, zt_extended], axis=-1)
        alpha = tf.reshape(tf.nn.softmax(extended), (self.batch_size, self.max_len, 2))
        print('alpha:{}'.format(alpha.get_shape()))

        self.beta = tf.tile(tf.expand_dims(alpha[:, :, -1], -1), [1, 1, self.cfg.hidden_dim])
        ct = self.beta * class_inputs + (1-self.beta)* tf.reshape(sentinel, [self.batch_size, self.max_len, -1])
        print('ct:{}'.format(ct.get_shape()))
        output = tf.reshape(ct, [-1, self.cfg.hidden_dim])

        # Softmax layer
        softmax_w = tf.get_variable(shape=[self.cfg.hidden_dim, self.vocab_size], name='softmax_w')
        softmax_b = tf.get_variable(shape=[self.vocab_size], name='softmax_b')
        self.logits = tf.add(tf.matmul(output, softmax_w), softmax_b, name='logits_output')
        self.probs = tf.nn.softmax(tf.reshape(self.logits, (self.batch_size, self.max_len, -1)), axis=2)
        print 'Logits:', self.logits.get_shape()

        # Predictions
        self.predictions = tf.argmax(self.logits, 1, name='predictions')
        print 'Predictions:', self.predictions.get_shape()

        # Minimize Loss
        targets_reshaped = tf.reshape(self._targets_placeholder, [-1])
        print 'Targets (raw):', self._targets_placeholder.get_shape()
        print 'Targets (reshaped):', targets_reshaped.get_shape()

        # Optimizer
        global_steps = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(self.cfg.lr, global_steps, 100, 0.5, staircase=False)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=targets_reshaped)
        self.loss = tf.reduce_sum(loss * tf.reshape(sequence, [-1]))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_op = optimizer.minimize(self.loss, global_step=global_steps, name='train_op')

    # def train_model(self, session=None, training_data=None, features=None):
    #     self.loss_history = []
    #     current_loss = 0
    #     best_loss = np.inf
    #     loss_unchange = 0
    #     iters = int(len(training_data) / self.batch_size) - 1
    #     data_gen = self.gen_data(training_data=training_data, features=features, iters=iters)
    #     saver = tf.train.Saver()
    #     for e in range(self.epochs):
    #         # np.random.shuffle(training_data)
    #         loss_iter = []
    #         for i in tqdm(range(iters)):
    #             img, sent, label, categories = data_gen.next()
    #             feed = {self._sent_placeholder: sent,
    #                     self._img_placeholder: img,
    #                     self._targets_placeholder: label,
    #                     self._class_placeholder: categories,
    #                     self._dropout_placeholder: self.cfg.keep_prob}
    #             loss, _ = session.run([self.loss, self.train_op], feed_dict=feed)
    #             loss_iter.append(loss)
    #             current_loss = np.mean(loss_iter) / self.batch_size
    #         self.loss_history.append(current_loss)
    #         if current_loss < best_loss:
    #             loss_unchange = 0
    #             print 'Epoch:{}, loss:{}--->{}, saving weights.'.format(e + 1, best_loss, current_loss)
    #             best_loss = current_loss
    #             saver.save(sess=session, save_path=self.cfg.save_model)
    #         else:
    #             print 'Epoch:{}, loss:{}.'.format(e + 1, current_loss)
    #             loss_unchange += 1
    #             if loss_unchange == self.cfg.early_stop:
    #                 break

    def train_model(self, session=None, data=None, features=None, val_features=None, val=None):
        self.loss_history = []
        current_train_loss = 0
        current_val_loss = 0
        best_loss = np.inf
        loss_unchange = 0

        # iters = int(np.floor(float(len(data)) / self.batch_size))
        iters = 100
        val_iters = int(np.floor(float(len(val)) / self.batch_size))
        data_gen = self.gen_data(training_data=data, features=features, iters=iters)
        val_gen = self.gen_data(training_data=val, features=val_features, iters=val_iters)

        saver = tf.train.Saver()
        for e in range(self.epochs):
            loss_iter = []
            val_losses = []
            for _ in tqdm(range(iters)):
                img, sent, label, categories = data_gen.next()
                feed = {self._sent_placeholder: sent,
                        self._img_placeholder: img,
                        self._targets_placeholder: label,
                        self._class_placeholder: categories,
                        self._dropout_placeholder: self.cfg.keep_prob}
                loss, _ = session.run([self.loss, self.train_op], feed_dict=feed)
                loss_iter.append(loss)
                current_train_loss = np.mean(loss_iter) / self.batch_size

            for _ in tqdm(range(val_iters)):
                img, sent, label, categories = val_gen.next()
                feed = {self._sent_placeholder: sent,
                        self._img_placeholder: img,
                        self._targets_placeholder: label,
                        self._class_placeholder: categories,
                        self._dropout_placeholder: 1}
                loss = session.run(self.loss, feed_dict=feed)
                val_losses.append(loss)
                current_val_loss = np.mean(val_losses) / self.batch_size

            self.loss_history.append(current_val_loss)
            if current_val_loss < best_loss:
                loss_unchange = 0
                print 'Epoch:{}, train_loss:{}, val_loss:{}--->{}, saving weights.'.format(e + 1, current_train_loss, best_loss, current_val_loss)
                best_loss = current_val_loss
                saver.save(sess=session, save_path=self.cfg.save_model)
                # if e >= 0:
                #     os.mkdir('./results/{}/'.format(str(e)))
                #     save_pth = './results/{}/model'.format(str(e))
                #     saver.save(sess=session, save_path=save_pth)

            else:
                print 'Epoch:{}, train_loss:{}, val_loss:{}.'.format(e + 1, current_train_loss, current_val_loss)
                loss_unchange += 1
                if loss_unchange == self.cfg.early_stop:
                    break

    def test_model(self, session=None, img_captions=None, features=None, i2w=None, w2i=None):
        caption_gen = np.zeros((self.batch_size, self.max_len))
        prediction = None
        sentence = ''
        img = np.zeros((self.batch_size, self.cfg.img_dim))
        for img_name, captions_gt in img_captions.items():
            img_name = '267164457_2e8b4d30aa.jpg'
            img[0, :] = features[img_name]
            print img_name
            for seq in range(self.max_len-1):
                sequence = [0] * self.batch_size
                sequence[0] = seq + 1
                feed = {self._sent_placeholder: caption_gen,
                        self._img_placeholder: img,
                        self._dropout_placeholder: self.cfg.keep_prob}
                prediction = session.run(self.predictions, feed_dict=feed)
                caption_gen[0, seq+1] = prediction[seq]
            pred = [i2w[i] for i in prediction[:self.max_len]]
            for word in pred:
                if word == '<end>':
                    continue
                else:
                    sentence = sentence + ' ' + word
            print sentence
            break

    def gen_sentence(self):
        pass

    def beam_search(self):
        pass

    def gen_data(self, training_data=None, features=None, iters=0):
        while True:
            random.shuffle(training_data)
            for i in range(iters):
                batch = training_data[i * self.batch_size:(i + 1) * self.batch_size]
                imgs = np.zeros((self.batch_size, self.cfg.img_dim))
                sents = np.zeros((self.batch_size, self.max_len))
                labels = np.zeros((self.batch_size, self.max_len))
                # sequence = np.zeros((self.batch_size, self.max_len))
                categories = []
                for index, sample in enumerate(batch):
                    imgs[index, :] = features[sample[0]]
                    sents[index, :len(sample[1]) - 1] = sample[1][:-1]
                    labels[index, :len(sample[1]) - 1] = sample[1][1:]
                    categories.append(sample[2])
                    # sequence[index, :len(sample[1]) - 1] = 1
                yield np.array(imgs), sents, labels, categories