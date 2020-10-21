# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 16:34:16 2020

@author: zhouh
"""

import json
from json import JSONEncoder
from pytube import YouTube
import numpy as np
import os
import cv2
from os import path

class dataprocessor():
    '''
        README: you need to install pytube before using this pipeline
                when use, please put videodatainfo_2017.json under the same directory 
                if pytube has some cipher error, you need to manually go into pytube 
                file and change the code. Details refer to https://github.com/nficano/pytube/pull/643/commits/10c57109f87fe864d8f38bbc8d76941e695de93a
                
    '''
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
        
    def do_all(self):
        records,video_ids,v2fs,caps = self.read_all()
        # record fps, frame_count, weight, height per video
        video_meta_file = './imgs/id.videometa.txt'
        fw = open(video_meta_file, 'a')
        for video_id in video_ids:
            if video_id in records:
                fps, length, width, height = records[video_id]
                fw.write('%s %d %d %d %d\n' % (video_id, fps, length, width, height))
        fw.close()
        # record caption
        caption_file = './imgs/caption.txt'
        fw = open(caption_file, 'a')
        for cap in caps:
            fw.write('%s\n' % cap)
        fw.close()
        
    def read_all(self):
        range_all = 10
        self.init_env()
        video_ids = []
        v2fs = {}
        caps = []
        records = {}
        for i in range(range_all):
            record, video_id, frames, captions = self.read_one(i)
            video_ids.append(video_id)
            v2fs[video_id] = frames
            if not record == ():
                records[video_id]=record
            j = 0 
            for cap in captions:
                caps.append(video_id+'#'+str(j)+' '+cap)
                j+=1
        return records,video_ids, v2fs, caps
            
        
    def read_one(self, index, save = False):
        video = self.videos[index]
        video_id = video['video_id']
        print(':: %s [%s]...' % (video['video_id'], video['url']))
        
        
        record,frames = self.extract_frame(video_id,video['url'],video['start time'],video['end time'],save)
        
        
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
        
        
        return record, video_id, frames, captions
        
        
    def extract_frame(self, video_id, url, start_time, end_time, save = False):
        frames = []
        record = ()
        if (path.exists('.\\imgs\\%s'  % video_id )):
            print('directory %s already exists, extraction cancelled.' % video_id)
            return record,frames
        try:
            video = YouTube(url)
            video.streams.filter(file_extension = "mp4").first().download(filename = video_id)
        except:
            print ("pytube download failed for video "+video_id)
            return record,frames

        
        vid_cap = cv2.VideoCapture(video_id+'.mp4')
        
        
        fps = int(vid_cap.get(cv2.CAP_PROP_FPS))
        start_fps = int(start_time*fps)
        end_fps = int(end_time*fps)
        frame_cnt = 0
        img_cnt = 0
        vid_cap.set(cv2.CAP_PROP_POS_FRAMES, start_fps)
        
        if cv2.__version__.startswith('3'):
            length  = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width   = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height  = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps     = int(vid_cap.get(cv2.CAP_PROP_FPS))
        else:
            length  = int(vid_cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
            width   = int(vid_cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
            height  = int(vid_cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
            fps     = int(vid_cap.get(cv2.cv.CV_CAP_PROP_FPS))
            
        record = (fps, length, width, height)
        
        os.mkdir('.\\imgs\\%s' % video_id)
        while vid_cap.isOpened():
            suc, img = vid_cap.read() 
            
            if not suc:
                break
            if frame_cnt > end_fps-start_fps:
                break
            if frame_cnt % self.every_x_frame == 0:
                #frames[video_id+'_'+str(frame_cnt+start_fps)] = img
                # Optional: save feature frames
                #if (save):
                cv2.imwrite(".\\imgs\\%s\\%s_%d.jpg" % (video_id, video_id, frame_cnt+start_fps), img)
                img_cnt += 1
                frames.append(video_id+'_'+str(frame_cnt+start_fps))
            frame_cnt += 1
            
        vid_cap.release()
        cv2.destroyAllWindows()
        print('extract '+str(img_cnt)+' images with separation of '+str(self.every_x_frame)+' frames')
        os.remove(video_id+".mp4")
        return record,frames
        
    def getJson(self):
        return self.data
    
    def init_env(self):
        if not path.exists('.\\'+self.subset_name):
            try:
                os.mkdir('.\\imgs')
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
    data = dp.getJson()
    
    # sent = dp.getsentence(57849)
    # record,video_ids, v2fs, caps = dp.read_one(1,save = True)
    dp.do_all()
