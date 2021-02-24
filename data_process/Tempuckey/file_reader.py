# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 19:24:46 2021

@author: shuha
"""
import numpy as np
import pandas as pd
import pickle
file_name ="./feature/3_TRIPPING_2018-04-14-col-nsh-national_00_56_33.857000_to_00_56_38.963000_2.bin"

#vocab_path = './vocab/word_vocab_5_bow.pkl'

f = open(file_name,"rb")
bin_data = f.read()
data = pickle.loads(bin_data, encoding='bytes')
res = data.get('feature')
print(type(res[0][0]))
print(res[0][0].shape)





#data = pd.read_pickle(vocab_path)
#data = pickle.load(vocab_path)
    
#print(data.__call__("sda"))