import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cv2
import numpy as np
from tqdm import tqdm
from model import Model
import time
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

    model = Model(batch_size=batch_size, classes=11, H=512, W=512, learning_rate=0.001, samples=5000)
    model.build_model()
    times = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(150):
            train_batch = np.random.random((batch_size, 512, 512, 3))
            start = time.time()
            _ = sess.run(model.predictions, feed_dict={model.inputs: train_batch})
            end = time.time()
            if i > 50:
                times.append(end-start)
    print('Time:{}'.format(np.mean(times)))


if __name__ == '__main__':
    train()