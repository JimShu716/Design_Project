# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 19:10:21 2020

@author: zhouh
"""

import pickle
import argparse
import os
import cv2

SAVE_PATH = '.\\feature\\'
TEST_PATH = '.\\test\\'
VIDEO_SOURCE_PATH = '.\\videos\\'
CAPTION_SOURCE_PATH = '.\\captions\\'

VID_1 = '1_TRIPPING_2017-11-28-fla-nyr-home_00_44_55.826000_to_00_45_06.437000.mp4'

class ExtractionPipeline():
    def __init__(self):
        self.caption_dict = []
        self.video_list = []
        self.extracted_fps = 10
        self.frame_per_package = 15
        
        dirpath, dirnames, files = next(os.walk(CAPTION_SOURCE_PATH))
        self.caption_dict = [file for file in files]

        dirpath, dirnames, files = next(os.walk(VIDEO_SOURCE_PATH))
        self.video_list = [file for file in files]
        
        
    
    def read(self,num_videos = 10, save = True):
        #if num_videos == -1:
        for i in range(num_videos):
            self.read_once(self.video_list[i],save)
            
    
    
    def read_once(self,video_name, save = True):
        video_info = self.process_video_name(video_name)
        captions = self.retrieve_captions(video_info)
        frames = self.retrieve_video_frames(video_info)
        feature = self.frame_to_feature(frames)
        
        file = {
            'video_info':video_info,
            'captions':captions,
            'feature':feature,
        }
        
        file_pickle = pickle.dumps(file)
        file_name = video_info['video_name'][:-4]+'.bin'
        filepath = os.path.join(SAVE_PATH,file_name)
        f = open(filepath, 'wb')
        f.write(file_pickle)
        f.close()
        return file
    
    def read_from_saved_binary_file(self, file_name):
        file_name = file_name[:-4]+'.bin'
        filepath = os.path.join(SAVE_PATH,file_name)
        f = open(filepath, 'rb')
        file = f.read()
        f.close()
        file = pickle.loads(file)
        
        return file
    
    def frame_to_feature(self, frames):
        # TODO: put code here to do feature embedding extraction
        feature = frames
        
        
        return feature
    
    
    def process_video_name(self,video_name):
        vlist = video_name.split("_",3)
        # remove '.mp4'
        vlist[3] = vlist[3][:-4]
        critical_time = vlist[3].split("_to_")
        start_time, start_frame     = critical_time[0].split(".")
        end_time,   end_frame       = critical_time[1].split(".")
        info = {
            'video_name':video_name,
            'video_id'  :vlist[0],
            'video_type':vlist[1],
            'video_info':vlist[2],
            'start_time':start_time,
            'start_frame':start_frame,
            'end_time':end_time,
            'end_frame':end_frame,
            }
        return info
    
    
    def retrieve_captions(self,video_info, save = False):
        lines = []
        for file in self.caption_dict:
            if file.split(".")[0] == video_info['video_info']:
                filepath = os.path.join(CAPTION_SOURCE_PATH,file)
                fp = open( filepath, "rb")
                lines = fp.readlines()
                lines = [line.decode('utf8') for line in lines]
                fp.close()
        else:
            print('D')
        # TODO: fix line format
        
        return lines
    
    
    def retrieve_video_frames(self,video_info, save = False):
        frames = {}
        video_path = os.path.join(VIDEO_SOURCE_PATH, video_info['video_name'])
        vid_cap = cv2.VideoCapture(video_path)
        fps = int(vid_cap.get(cv2.CAP_PROP_FPS))
        factor = round(fps/self.extracted_fps)
    
        cur_frame = 0
        img_cnt = 0
        vid_cap.set(cv2.CAP_PROP_POS_FRAMES, cur_frame)
        
        # read frames
        while vid_cap.isOpened():
            suc, img = vid_cap.read() 
            if not suc:
                break
            if cur_frame % factor == 0:
                frames[str(cur_frame)] = img
                # Optional: save feature frames
                if (save):
                    cv2.imwrite("%s\\%s\\%d.jpg" % (TEST_PATH,video_info['video_id'],cur_frame), img)
                img_cnt += 1
            cur_frame += 1
            
        # clean up
        vid_cap.release()
        cv2.destroyAllWindows()
        
        # pack frames
        frame_list = list(frames.values())
        packed_frames = [frame_list[i:i + self.frame_per_package] for i in range(0, len(frame_list), self.frame_per_package)]
        print(f"{video_info['video_name']}: packed {len(packed_frames)} packs from extracted {img_cnt} frames from total {cur_frame} frames.")
        return packed_frames

if __name__ == '__main__':
    
    pipe = ExtractionPipeline()
    pipe.read()
    #file = pipe.read_once(VID_1)
    #file_2 = pipe.read_from_saved_binary_file(VID_1)
        
