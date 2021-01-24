# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 19:10:21 2020

@author: zhouh
"""

import json
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
        self.extracted_fps = 10
        self.frame_per_package = 15
        self.init_caption_dictionary()

    def init_caption_dictionary(self):
        dirpath, dirnames, files = next(os.walk(CAPTION_SOURCE_PATH))
        self.caption_dict = [file for file in files]
        
    
    def read(self,num_videos = 10):
        #if num_videos == -1:
        dirpath, dirnames, files = next(os.walk(VIDEO_SOURCE_PATH))
    
    
    def read_once(self,video_name):
        video_info = self.process_video_name(video_name)
        
    
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
                print('Found!')
                filepath = os.path.join(CAPTION_SOURCE_PATH,file)
                fp = open( filepath, "rb")
                lines = fp.readlines()
                fp.close()
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
    
    a = ExtractionPipeline()
    lines = a.retrieve_captions(a.process_video_name(VID_1))
    #frames = retrieve_video_frames(process_video_name(VID_1))
        
