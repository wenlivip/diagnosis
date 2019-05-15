import tensorflow as tf
from tensorflow.python import pywrap_tensorflow as pt
import pickle
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from config import Config
from model import Model


def load_pickle(path=None):
    data = pickle.load(open(path, 'rb'))
    return data


def process_data(img_captions=None, w2i=None,img_category=None):
    training_data = []
    for img_name, captions in img_captions.items():
        for caption in captions:
            sample_caption = [w2i[w] for w in caption.split(' ')]
            training_data.append([img_name, sample_caption, img_category[img_name]])
    return training_data


def main():
    use_pretrained = False
    cfg = Config()
    dictionary = load_pickle('./dataset/dictionary.pkl')
    features = load_pickle('./dataset/trainval_feature.pkl')
    img_captions = dictionary['img_captions']
    w2i = dictionary['w2i']
    i2w = dictionary['i2w']
    img_category = dictionary['img_category']
    max_len = dictionary['max_len']
    vocab_size = len(w2i)
    print 'vocab_size:', vocab_size
    print 'max_len:', max_len
    cfg.max_len = max_len
    cfg.vocab_size = vocab_size

    print('Loading data...')
    training_data = process_data(img_captions=img_captions, w2i=w2i, img_category=img_category)

    print('Building model...')
    model = Model(cfg)
    model.build_model()

    print('Start to train...')
    with tf.Session() as sess:
        if use_pretrained:
            saver = tf.train.Saver()
            saver.restore(sess=sess, save_path='./pre_model/model')
        else:
            init = tf.global_variables_initializer()
            sess.run(init)
        model.train_model(session=sess, training_data=training_data, features=features)


if __name__=='__main__':
    main()