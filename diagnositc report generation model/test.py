#!/usr/bin/python
# -*- coding: UTF-8 -*-

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow as pt
import pickle
from tqdm import tqdm
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from config import Config
from model import Model


def get_test_imgs(pth=None):
    img_names = []
    with open(pth, 'r') as f:
        lines = f.readlines()
    for l in lines:
        img_names.append(l.split('\t')[0].replace('.png', '.jpg'))
    return img_names


def load_pickle(path=None):
    data = pickle.load(open(path, 'rb'))
    return data


def process_data(img_captions=None, w2i=None):
    training_data = []
    for img_name, captions in img_captions.items():
        for caption in captions:
            sample_caption = [w2i[w] for w in caption.split(' ')]
            training_data.append([img_name, sample_caption])
    return training_data


def buid_ENG2CHN_dictionary(pth):
    dic = {}
    with open(pth, 'r') as f:
        lines = f.readlines()
    for l in lines:
        # print l
        chn, eng = l.replace('\n', '').split('\t')
        dic[eng] = chn
    return dic


def encode2chn(string, e2c_dic):
    words = string.split()
    chinese = ''
    for w in words:
        w = w.replace('<end>', '')
        chinese += e2c_dic[w]
    return chinese


def mk_gt_chn(pth=None, dic=None):
    with open(pth, 'r') as f:
        lines = f.readlines()
    res = []
    for l in lines:
        img_name, _, caption = l.split('\t')
        caption.replace('\n', '')
        chn_cap = encode2chn(caption, dic)
        res.append('{}\t{}\n'.format(img_name, chn_cap))
    with open('./dataset/ch_test.csv', 'w') as f:
        f.writelines(res)


class Model(object):

    def __init__(self, config):
        self.cfg = config
        self.vocab_size = config.vocab_size
        self.batch_size = config.batch_size
        self.max_len = config.max_len
        self.epochs = config.epochs

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
            word_embeddings = tf.get_variable('word_embeddings', shape=[self.vocab_size, self.cfg.embed_dim])
            sent_inputs = tf.nn.embedding_lookup(word_embeddings, self._sent_placeholder)
            print 'Sent:', sent_inputs.get_shape()
        with tf.variable_scope('class_embeding'):
            class_embeddings = tf.get_variable('class_embeddings', shape=[self.vocab_size, self.cfg.hidden_dim])
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

        # state = lstm_dropout.zero_state(self.cfg.batch_size, tf.float32)
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

        beta = tf.tile(tf.expand_dims(alpha[:, :, -1], -1), [1, 1, self.cfg.hidden_dim])
        ct = beta * class_inputs + (1-beta)* tf.reshape(sentinel, [self.batch_size, self.max_len, -1])
        print('ct:{}'.format(ct.get_shape()))
        output = tf.reshape(ct, [-1, self.cfg.hidden_dim])

        # Softmax layer
        softmax_w = tf.get_variable(shape=[self.cfg.hidden_dim, self.vocab_size], name='softmax_w')
        softmax_b = tf.get_variable(shape=[self.vocab_size], name='softmax_b')
        self.logits = tf.add(tf.matmul(output, softmax_w), softmax_b, name='logits_output')
        print 'Logits:', self.logits.get_shape()

        # Predictions
        self.predictions = tf.argmax(self.logits, 1, name='predictions')
        print 'Predictions:', self.predictions.get_shape()

        # Minimize Loss
        targets_reshaped = tf.reshape(self._targets_placeholder, [-1])
        print 'Targets (raw):', self._targets_placeholder.get_shape()
        print 'Targets (reshaped):', targets_reshaped.get_shape()

        # Optimizer
        # self.loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=targets_reshaped), name='ce_loss')
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=targets_reshaped)
        self.loss = tf.reduce_sum(loss * tf.reshape(sequence, [-1]))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.cfg.lr)
        self.train_op = optimizer.minimize(self.loss, name='train_op')


def main():
    cfg = Config()
    dictionary = load_pickle('./dataset/dictionary.pkl')
    test_features = load_pickle('./dataset/features_test.pkl')    # train
    # img_captions = load_pickle('./dataset/test_captions.pkl')
    img_names = get_test_imgs(pth='./dataset/test_imporession.csv')
    # ENG2CHN = buid_ENG2CHN_dictionary('./dataset/dictionary.csv')
    w2i = dictionary['w2i']
    i2w = dictionary['i2w']
    max_len = dictionary['max_len']
    vocab_size = len(w2i)
    cfg.max_len = max_len
    cfg.vocab_size = vocab_size
    print 'start:{} end:{} null:{}'.format(w2i['<start>'], w2i['<end>'], w2i['<null>'])
    print 'Data has been loaded!'

    model = Model(cfg)
    model.build_model()
    # chn_sentences = []
    sentences = []
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess=sess, save_path=cfg.save_model)
        print 'Model has been loaded'
        for img_name in tqdm(img_names):
        # for img_name in img_names[:]:
            feature = test_features[img_name]
            # print '{}: {}'.format(img_name, encode2chn(img_captions[img_name], ENG2CHN))
            features = np.zeros((model.batch_size, cfg.img_dim))
            features[:, :] = feature
            sent_pred = np.zeros((model.batch_size, max_len))
            sent_pred[:, 1] = w2i['<start>']
            sequence = [1] * model.batch_size
            sent = []
            for _ in range(max_len):
                feed = {model._sent_placeholder: sent_pred,
                        model._img_placeholder: features,
                        model._class_placeholder: categories,
                        model._dropout_placeholder: 1}
                prediction = sess.run(model.predictions, feed_dict=feed)
                prediction = np.reshape(prediction, (model.batch_size, -1))
                pred_word = prediction[0, sequence[0]-1]
                if pred_word == w2i['<end>']:
                    break
                sent.append(i2w[pred_word])
                sent_pred[:, sequence[0]] = pred_word
                sequence = [i+1 for i in sequence]
            # chn_sentences.append(encode2chn(' '.join(sent), ENG2CHN))
            sent = ' '.join(sent)
            sent = '{}'.format(sent)
            sent = '{}\t{}\n'.format(img_name, sent)
            sentences.append(sent)
    with open('./results.csv', 'w') as f:
        f.writelines(sentences)


if __name__ == '__main__':
    main()
    # ENG2CHN = buid_ENG2CHN_dictionary('./dataset/dictionary.csv')
    # mk_gt_chn(pth='./dataset/test.csv', dic=ENG2CHN)
