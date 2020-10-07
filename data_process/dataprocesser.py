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
        self.filepath = jpath
        with open(self.filepath) as json_file:
            self.data = json.load(json_file)
            
        self.info = self.data['info']
        self.sentences = self.data['sentences']
        self.videos = self.data['videos']
        
        self.num_video = len(self.videos)
        self.num_cap = len(self.sentences)
        

        self.every_x_frame = every_x_frame
        self.subset_name = subset_name
        self.feature_name = feature_name
        
        # output
        
        
    def read_all(self):
        range_all = 10
        self.init_env()
        video_ids = []
        v2fs = {}
        caps = []
        for i in range(range_all):
            video_id, frames, captions = self.read_one(i)
            video_ids.append(video_id)
            v2fs[video_id] = list(frames.keys())
            j = 0 
            for cap in captions:
                caps.append(video_id+'#'+str(j)+' '+cap)
                j+=1
        return video_ids, v2fs, caps
            
        
    def read_one(self, index, save = False):
        video = self.videos[index]
        video_id = video['video_id']
        print('getting frames...' + video['url'])
        frames = self.extract_frame(video_id,video['url'],video['start time'],video['end time'],save)
        cap_cnt = 0
        captions = []
        print('getting captions...')
        while (cap_cnt < 200000):
            if not self.sentences[cap_cnt]['video_id'] == video_id:
                cap_cnt += 20
                continue
            captions.append(self.sentences[cap_cnt]['caption'])
            cap_cnt+=1
            if (len(captions)==20): break
        return video_id, frames, captions
        
        
    def extract_frame(self, video_id, url, start_time, end_time, save = False):
        frames = {}
        video = YouTube(url)
        video.streams.filter(file_extension = "mp4").first().download(filename = video_id)
        vid_cap = cv2.VideoCapture(video_id+'.mp4')
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
                if (save):
                    cv2.imwrite(".\\imgs\\%s_%d.jpg" % (video_id, frame_cnt+start_fps), img)
                img_cnt += 1
            frame_cnt += 1
            
        vid_cap.release()
        cv2.destroyAllWindows()
        print('extract '+str(img_cnt)+' images with separation of '+str(self.every_x_frame)+' frames')
        # with open(video_id+".json", "w") as write_file:
        #     json.dump(frames, write_file, cls=NumpyArrayEncoder)
        os.remove(video_id+".mp4")
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

    def getsentence(self, index):
        return self.sentences[index]['caption']
    
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
    
if __name__ == '__main__':
    dp = dataprocessor('.\\videodatainfo_2017.json')
    #frames = dp.extract_frame('video1','https://www.youtube.com/watch?v=9lZi22qLlEo',137.72,149.44)
    data = dp.getJson()
    # video_ids, v2fs, caps = dp.read_one(65,save = True)
    video_ids, v2fs, caps = dp.read_one(2980,save = True)
    sent = dp.getsentence(57849)
    # video_ids, v2fs, caps = dp.read_one(147,save = True)
    # video_ids, v2fs, caps = dp.read_one(148,save = True)
    #video_ids, v2fs, caps = dp.read_one(2)
    #dp.init_env()