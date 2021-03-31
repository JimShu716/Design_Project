# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 00:21:43 2021

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

txtfile =''
id_feature_dir = '.\\feature\\'
def extract_feature( frame, model):
    # ==========resize the frame to fit the model ========
    x = st.resize(frame, (224, 224, 3))
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)
    feature = model.predict(x)
    return feature


def mean_pooling(feat_matrix):
    return feat_matrix[0][0]



def process(feat_dim, inputTextFiles, resultdir, overwrite):
    res_binary_file = os.path.join(resultdir, 'feature.bin')
    res_id_file = os.path.join(resultdir, 'id.txt')

    if checkToSkip(res_binary_file, overwrite):
        return 0

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
            vec = np.array(map(float, elems), dtype=np.float32)
            okay = True
            for x in vec:
                if math.isnan(x):
                    okay = False
                    break
            if not okay:
                failed += 1
                continue
          
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




def save_frame_to_binary(frame_dict: dict, save_path: str,video_img_dict: dict,):
    
    print("saving frame into txt .....")
    model = ResNet152(weights='imagenet',pooling="avg")
    res_txt_file = os.path.join(save_path, 'id.feature.txt')
    video2frame_txt_file = os.path.join(save_path, 'video2frame.txt')
    
    f = open(res_txt_file, 'w')
    
    for key in frame_dict:
        
        frame = frame_dict[key]
        f.write(str(key)+" ")

        feature = extract_feature(frame,model)
        
        for feature_elem in feature:
            for feature_item in feature_elem:
                f.write(str(feature_item)+" ")
            
        f.write("\n")
    
    f.close()
    print("txt file saved！")
   
    #-----------Save video2frame-------------
    f = open(video2frame_txt_file, 'w')
    f.write(str(video_img_dict))
    f.close()
    print("video2frame file saved！")
        
    pass

