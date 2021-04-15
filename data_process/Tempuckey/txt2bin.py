# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 10:34:32 2021

@author: shuha
"""

import datetime
import logging
import pickle
import os
import string

import cv2
import pandas as pd
import numpy as np
import math
import torch
from tensorflow.keras.applications import ResNet152
from tensorflow.keras.applications.resnet50 import preprocess_input
import skimage.transform as st

train_id_feature_dir = './feature_tempuckey/msrvtt_train/FeatureData/'
test_id_feature_dir = './feature_tempuckey/msrvtt_test/FeatureData/'
eval_id_feature_dir = './feature_tempuckey/msrvtt_eval/FeatureData/'

def process(feat_dim, inputTextFiles, resultdir, overwrite):
    res_binary_file = os.path.join(resultdir, 'feature.bin')
    res_id_file = os.path.join(resultdir, 'id.txt')

    if os.path.isdir(resultdir) is False:
        os.makedirs(resultdir)

    fw = open(res_binary_file, 'wb')
    processed = set()
    imset = []
    count_line = 0
    failed = 0

    for filename in inputTextFiles:
        print ('>>> Processing %s' % filename)
        for line in open(filename):
            count_line += 1
            elems = line.strip().split()
            if not elems:
                continue
            name = elems[0]
            if name in processed:
                continue
            processed.add(name)

            del elems[0]
            vec = np.fromiter(map(float, elems), dtype=np.float32)
            okay = True
            for x in vec:
                if math.isnan(x):
                    okay = False
                    break
            if not okay:
                failed += 1
                continue
            
            if feat_dim == 0:
               feat_dim = len(vec)
            else: 
                assert(len(vec) == feat_dim), "dimensionality mismatch: required %d, input %d, id=%s, inputfile=%s" % (feat_dim, len(vec), name, filename)
            vec.tofile(fw)
            #print name, vec
            imset.append(name)
    fw.close()

    fw = open(res_id_file, 'w')
    fw.write(' '.join(imset))
    fw.close()
    fw = open(os.path.join(resultdir,'shape.txt'), 'w')
    fw.write('%d %d' % (len(imset), feat_dim))
    fw.close() 
    print ('%d lines parsed, %d ids,  %d failed ->  %d unique ids' % (count_line, len(processed), failed, len(imset)))


train_txt_file = os.path.join(train_id_feature_dir, 'id.feature.txt')
res_train_test_txt_file =[]
res_train_test_txt_file.append(train_txt_file)

test_txt_file = os.path.join(test_id_feature_dir, 'id.feature.txt')
res_test_txt_file =[]
res_test_txt_file.append(test_txt_file)

eval_txt_file = os.path.join(eval_id_feature_dir, 'id.feature.txt')
res_eval_txt_file =[]
res_eval_txt_file.append(eval_txt_file)



process(0,res_train_test_txt_file,train_id_feature_dir,False)
process(0,res_test_txt_file,test_id_feature_dir,False)
process(0,res_eval_txt_file,eval_id_feature_dir,False)

