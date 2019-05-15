import os
import numpy as np
import pickle


def process_caption(path=None, img_names=None):
    with open(path, 'r') as f:
        lines = f.readlines()

    img_captions = {}
    img_category = {}
    for line in lines:
        img_name, pos, caption = line.split('\t')
        img_name = img_name.split('/')[-1]
        if img_name not in img_names:
            print 'Error!'
        caption = caption.replace('\n', '').replace('\t', '').replace('jlvq', '')
        caption = '<start> ' + caption + '<end>'
        caption = " ".join(caption.split())  # replace multiple spaces
        caption = caption.lower()

        category = int(pos.split(' ')[0])

        if img_name not in img_captions:
            img_captions[img_name] = []
        img_captions[img_name].append(caption)
        img_category[img_name] = category
    return img_captions, img_category


def content2dict(path):
    pass


def max_length(img_captions):
    max_len = 0
    for _, captions in img_captions.items():
        for caption in captions:
            words = caption.split(' ')
            if len(words) > max_len:
                max_len = len(words)
    return max_len


def build_dictionary(img_captions):
    word2idx = {}
    idx2word = {}
    idx = 1
    for _, captions in img_captions.items():
        for caption in captions:
            words = caption.split(' ')
            for w in words:
                if w not in word2idx:
                    word2idx[w] = idx
                    idx2word[idx] = w
                    idx += 1
    word2idx['<null>'] = 0
    idx2word[idx] = '<null>'
    return word2idx, idx2word


def load_pickle(path=None):
    data = pickle.load(open(path, 'rb'))
    return data


def read_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return lines


def get_imgnames(features=None):
    img_names = []
    for img_name, _ in features.items():
        img_names.append(img_name)
    return img_names


def test_captions(pth=None):
    test_captions = {}
    with open(pth, 'r') as f:
        lines = f.readlines()
    for l in lines:
        img_name, _, caption = l.split('\t')
        caption = caption.replace('\n', '')
        test_captions[img_name] = caption
    return test_captions


def main():
    features = load_pickle('./dataset/trainval_feature.pkl')
    img_names = get_imgnames(features=features)
    captions_path = './dataset/train.csv'
    lines = read_file(captions_path)
    img_captions, img_category = process_caption(path=captions_path, img_names=img_names)
    print 'total training images: {}'.format(len(img_captions))
    max_len = max_length(img_captions)
    print 'max length of sentence: {}'.format(max_len)
    word2idx, idx2word = build_dictionary(img_captions)
    print 'vocab_size:', len(word2idx)
    # dictionary = {'img_captions': img_captions, 'w2i': word2idx, 'i2w': idx2word, 'max_len': max_len, 'img_names': img_names}
    dictionary = {'img_category': img_category, 'img_captions':img_captions, 'w2i': word2idx, 'i2w': idx2word, 'max_len': max_len, 'img_names': img_names}
    print 'the index of <null> is:{}'.format(word2idx['<null>'])
    print 'the index of <start> is:{}'.format(word2idx['<start>'])
    print 'the index of <end> is:{}'.format(word2idx['<end>'])
    with open('./dataset/dictionary.pkl', 'w') as f:
        pickle.dump(dictionary, f)

    # t_captions = test_captions('./dataset/test.csv')
    # with open('./dataset/test_captions.pkl', 'w') as f:
    #     pickle.dump(t_captions, f)


if __name__=='__main__':
    main()