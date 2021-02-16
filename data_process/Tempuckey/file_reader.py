# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 19:24:46 2021

@author: shuha
"""
import numpy as np
import pandas as pd
import pickle
#file_name ="5_TRIPPING_2018-04-14-col-nsh-national_02_29_37.802000_to_02_29_44.442000.bin"

vocab_path = './vocab/word_vocab_5.pkl'

#f = open(file_name,"rb")
#bin_data = f.read()
#data = pickle.loads(bin_data, encoding='bytes')
#res = data.get('feature')
#print(type(res[0][0]))
#print(res[0][0].shape)

class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self, text_style):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.text_style = text_style

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx and 'bow' not in self.text_style:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)



data = pd.read_pickle(vocab_path)
#data = pickle.load(vocab_path)
    
print(data.__call__())