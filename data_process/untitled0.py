# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 19:10:21 2020

@author: zhouh
"""

import json
import argparse
import os
import shutil
from tqdm import tqdm
import cv2
import logging

EXTRACTED_FRAME_PER_SECOND = 10
SECOND_PER_PACKAGE = 5
SAVE_PATH = '.\\feature\\'
SOURCE_PATH = '.\\videos\\'

def extract_frame_from_video(video_name:str, save = True):
    frames = {}
    video_path = os.path.join(SOURCE_PATH, video_name)
    vid_cap = cv2.VideoCapture(video_path)
    return vid_cap
    
    # fps = int(vid_cap.get(cv2.CAP_PROP_FPS))
    # start_fps = int(start_time*fps)
    # end_fps = int(end_time*fps)
    # frame_cnt = 0
    # img_cnt = 0
    # vid_cap.set(cv2.CAP_PROP_POS_FRAMES, start_fps)
    # while vid_cap.isOpened():
    #     suc, img = vid_cap.read() 
        
    #     if not suc:
    #         break
    #     if frame_cnt > end_fps-start_fps:
    #         break
    #     if frame_cnt % every_x_frame == 0:
    #         frames[video_id+'_'+str(frame_cnt+start_fps)] = img
    #         # Optional: save feature frames
    #         if (save):
    #             cv2.imwrite("%s\\%s_%d.jpg" % (path,video_id, frame_cnt+start_fps), img)
    #         img_cnt += 1
    #     frame_cnt += 1
        
    # vid_cap.release()
    # cv2.destroyAllWindows()
    # print('extract %s images from %s' % str(img_cnt), video_id)
    # os.remove(video_id+".mp4")
    # return frames

if __name__ == '__main__':
    vid = extract_frame_from_video('')
        
