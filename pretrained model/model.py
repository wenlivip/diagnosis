import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import tensorflow as tf
from tensorflow.contrib.slim import nets
slim = tf.contrib.slim



class Model():
    def __init__(self, batch_size=16, classes=11, H=448, W=448, learning_rate=0.0001, samples=1000):
        self.batch_size = batch_size
        self.H = H
        self.W = W
        self.lr = learning_rate
        self.classes = classes
        self.samples = int(samples / batch_size)

    def build_model(self):
        self.inputs = tf.placeholder(dtype=tf.float32, shape=[None, self.H, self.W, 3], name='inputs_ph')
        self.labels = tf.placeholder(dtype=tf.int32, shape=[None], name='labels_ph')

        with slim.arg_scope(nets.resnet_v1.resnet_arg_scope()):
            net, endpoints = nets.resnet_v1.resnet_v1_50(self.inputs, num_classes=None, is_training=True)

        with tf.variable_scope('Logits'):
            net = tf.squeeze(net, axis=[1, 2])
            net = slim.dropout(net, keep_prob=0.5, scope='scope')
            logits = slim.fully_connected(net, num_outputs=self.classes, activation_fn=None, scope='fc')

        checkpoint_exclude_scopes = 'Logits'
        exclusions = None
        if checkpoint_exclude_scopes:
            exclusions = [scope.strip() for scope in checkpoint_exclude_scopes.split(',')]
        self.variables_to_restore = []
        for var in slim.get_model_variables():
            excluded = False
            for exclusion in exclusions:
                if var.op.name.startswith(exclusion):
                    excluded = True
            if not excluded:
                self.variables_to_restore.append(var)

        global_steps = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(self.lr, global_steps, self.samples, 0.9, staircase=False)

        self.predictions = tf.argmax(logits, axis=1)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(self.predictions, dtype=tf.int32), self.labels), dtype=tf.float32))
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=logits)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_op = optimizer.minimize(self.loss, global_step=global_steps)