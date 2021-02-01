# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 19:24:46 2021

@author: shuha
"""
import numpy as np
import pickle
file_name ="5_TRIPPING_2018-04-14-col-nsh-national_02_29_37.802000_to_02_29_44.442000.bin"

f = open(file_name,"rb")
bin_data = f.read()
data = pickle.loads(bin_data, encoding='bytes')
res = data.get('feature')
print(type(res[0][0]))
print(res[0][0].shape)