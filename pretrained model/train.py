import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cv2
import numpy as np
from tqdm import tqdm
from model import Model
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle as data_shuffle


def read_data(pth=None, h=224, w=224):
    pre_pth = './ul/train/'
    with open(pth, 'r') as f:
        lines = f.readlines()

    # lines = lines[:200]
    imgs = np.zeros((len(lines), h, w, 3))
    labels = [0] * len(lines)
    for index, l in tqdm(enumerate(lines)):
        img_pth, label, _ = l.split('\t')
        img_pth = os.path.join(pre_pth, img_pth)
        # print(img_pth)
        img = cv2.imread(img_pth)
        img = cv2.resize(img, (h, w))
        imgs[index, :, :, :] = img
        labels[index] = int(label.split(' ')[0])
    return imgs, np.array(labels)


def train():

    epochs = 50
    batch_size = 16

    print('Loading data...')
    imgs, labels = read_data(pth='./ul/train.csv')
    x_train, x_valid, y_train, y_valid = train_test_split(imgs, labels, test_size=0.1, random_state=0)
    print('Train:{} {}'.format(x_train.shape, y_train.shape))
    print('Valid:{} {}'.format(x_valid.shape, y_valid.shape))

    model = Model(batch_size=batch_size, classes=11, H=224, W=224, learning_rate=0.001, samples=x_train.shape[0])
    model.build_model()

    print('Start to train...')
    saver = tf.train.Saver(model.variables_to_restore)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.8

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, './model/resnet_v1_50.ckpt')

        best_loss = np.inf
        train_samples = x_train.shape[0]
        val_samples = x_valid.shape[0]
        for e in range(epochs):
            x_train, y_train = data_shuffle(x_train, y_train)
            losses = []
            val_losses = []
            acces = []
            for i in tqdm(range(train_samples / batch_size)):
                train_batch = x_train[i * batch_size:(i + 1) * batch_size, :, :, :]
                labels_batch = y_train[i * batch_size:(i + 1) * batch_size]
                loss, _ = sess.run([model.loss, model.train_op],
                                   feed_dict={model.inputs: train_batch, model.labels: labels_batch})
                losses.append(loss)

            for i in tqdm(range(val_samples / batch_size)):
                val_batch = x_valid[i * batch_size:(i + 1) * batch_size, :, :, :]
                labels_val = y_valid[i * batch_size:(i + 1) * batch_size]
                val_loss, acc = sess.run([model.loss, model.accuracy],
                                         feed_dict={model.inputs: val_batch, model.labels: labels_val})
                val_losses.append(np.mean(val_loss))
                acces.append(acc)

            # print(val_losses)
            e_loss = np.mean(val_losses)
            e_acc = np.mean(acces)
            # print('val_losses:{}'.format(val_losses))
            # print('acces:{}'.format(acces))
            print('Epochs:{}  loss:{} acc:{}'.format(e, e_loss, e_acc))
            if e_loss < best_loss:
                saver.save(sess, './results/resnet_v1_50.ckpt')
                best_loss = e_loss


if __name__ == '__main__':
    train()