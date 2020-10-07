# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 16:34:16 2020

@author: zhouh
"""

import json
from json import JSONEncoder
from pytube import YouTube
import pandas as pd
import numpy as np
import os
import cv2
from os import path

class dataprocessor():
    
    def __init__(self,jpath,subset_name = 'msrvtt201710k',feature_name = 'feature', every_x_frame = 15):
        
        # data
        with open(self.filepath) as json_file:
            self.data = json.load(json_file)
            
        self.info = self.data['info']
        self.sentences = self.data['sentences']
        self.videos = self.data['videos']
        
        self.num_video = len(self.videos)
        self.num_cap = len(self.sentences)
        
        self.filepath = jpath
        self.every_x_frame = every_x_frame
        self.subset_name = subset_name
        self.feature_name = feature_name
        
        # output
        
        
    def read_all(self):
        self.init_env()
        video_ids = {}
        for video in self.videos:
            datatype = video['split']
            if (datatype not in video_ids):
                video_ids[datatype] = []
            video_ids.append(video['video_id'])
            
        
    def read_one(self, index = -1):
        pass
    
    def extract_frame(self, video_id, url, start_time, end_time):
        frames = {}
        if ( not path.exists('tmp.mp4')):
            video = YouTube(url)
            video.streams.filter(file_extension = "mp4").first().download(filename = 'tmp')
        vid_cap = cv2.VideoCapture('.\\tmp.mp4')
        fps = int(vid_cap.get(cv2.CAP_PROP_FPS))
        start_fps = int(start_time*fps)
        end_fps = int(end_time*fps)
        frame_cnt = 0
        img_cnt = 0
        vid_cap.set(cv2.CAP_PROP_POS_FRAMES, start_fps)
        while vid_cap.isOpened():
            suc, img = vid_cap.read() 
            
            if not suc:
                break
            if frame_cnt > end_fps-start_fps:
                break
            if frame_cnt % self.every_x_frame == 0:
                frames[video_id+'_'+str(frame_cnt+start_fps)] = img
                img_cnt += 1
            frame_cnt += 1
            
        vid_cap.release()
        cv2.destroyAllWindows()
        print('extract '+str(img_cnt)+' images with separation of '+str(self.every_x_frame)+' frames')
        with open(video_id+".json", "w") as write_file:
            json.dump(frames, write_file, cls=NumpyArrayEncoder)
        os.remove("tmp.mp4")
        return frames
        
    def getJson(self):
        return self.data
    
    def init_env(self):
        if not path.exists('.\\'+self.subset_name):
            try:
                os.mkdir('.\\'+self.subset_name)
                os.mkdir('.\\'+self.subset_name+'\\FeatureData')
                os.mkdir('.\\'+self.subset_name+'\\FeatureData\\'+self.feature_name)
                os.mkdir('.\\'+self.subset_name+'\\TextData')
            except:
                raise 'create dir failed'
        
       
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
    
if __name__ == '__main__':
    dp = dataprocessor('.\\videodatainfo_2017.json')
    frames = dp.extract_frame('video1','https://www.youtube.com/watch?v=9lZi22qLlEo',137.72,149.44)
    data = dp.getJson()