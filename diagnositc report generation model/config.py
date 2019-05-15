import pickle

class Config(object):
    def __init__(self):
        # train
        self.img_dim = 2048
        self.hidden_dim = 512
        self.embed_dim = 1024
        self.lr = 0.001
        self.batch_size = 16
        self.keep_prob = 0.75
        self.layers = 1

        self.use_pretrained = False
        self.pre_model = './pre_model/model'
        self.save_model = './results/model'
        self.dictionary = './dictionary.pkl'

        self.max_len = 24
        self.class_nums = 11
        self.vocab_size = 202
        self.w2i = None
        self.i2w = None

        self.epochs = 15
        self.early_stop = 5

        # test
        self.beam_search = 5