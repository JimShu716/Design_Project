# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 19:10:21 2020

@author: zhouh
"""

import pickle
import os
import cv2
import srt

SAVE_PATH = '.\\feature\\'
TEST_PATH = '.\\test\\'
VIDEO_SOURCE_PATH = '.\\videos\\'
CAPTION_SOURCE_PATH = '.\\captions\\'

VID_1 = '1_TRIPPING_2017-11-28-fla-nyr-home_00_44_55.826000_to_00_45_06.437000.mp4'

class ExtractionPipeline():
    def __init__(self, num_video = 10, extracted_fps = 10, frame_per_package = 15):
        self.caption_list = []
        self.video_list = []
    
        # 10/15 => 1.5 sec per package
        self.extracted_fps = extracted_fps
        self.frame_per_package = frame_per_package
        self.num_video = num_video
        
        # init environment
        for p in SAVE_PATH, TEST_PATH, VIDEO_SOURCE_PATH, CAPTION_SOURCE_PATH:
            if not os.path.exists(p):
                os.mkdir(p)    
        
        # init caption lookup list
        dirpath, dirnames, files = next(os.walk(CAPTION_SOURCE_PATH))
        self.caption_list = [file for file in files]

        # init video lookup list
        dirpath, dirnames, files = next(os.walk(VIDEO_SOURCE_PATH))
        self.video_list = [file for file in files]
        
        # if number of video to process is too large or less than 0, read all videos
        if self.num_video > len(self.video_list) or self.num_video < 0:
            self.num_video = len(self.video_list)
   
    
    def read(self, save = True):
        print(f"Start reading process, read total number of {self.num_videos} files from {VIDEO_SOURCE_PATH}:")
        for i in range(self.num_videos):
            print(f"=== Task {i}/{self.num_videos}:")
            self.read_once(self.video_list[i],save)
            print(f"===")
    
    def read_once(self,video_name, save = True, over_write = False):
        print(f'{video_name}:')
        if not over_write:
            save_filepath = os.path.join(SAVE_PATH,video_name[:-4]+'.bin')
            if os.path.exists(save_filepath):
                print('Found saved feature file,skip this file.')
                return None
        
        video_info = self.process_video_name(video_name)
        
        
        captions = self.retrieve_captions(video_info)
        if (len(captions)==0):
            print(f"Error: Did not find the subtitle for video {video_name}")
            return None
        
        frames = self.retrieve_frames(video_info)
        if (len(frames)==0):
            print(f"Error: Loading frame error for video {video_name}")
            return None
        
        feature = self.frame_to_feature(frames)
        
        file = {
            'video_info':video_info,
            'captions':captions,
            'feature':feature,
        }
        
        if save:
            file_pickle = pickle.dumps(file)
            file_name = video_info['video_name'][:-4]+'.bin'
            filepath = os.path.join(SAVE_PATH,file_name)
            f = open(filepath, 'wb')
            f.write(file_pickle)
            f.close()
            print('File generated.')
        else:
            print("File not saved.")
        
        return file
    
    def read_from_saved_binary_file(self, file_name):
        file_name = file_name[:-4]+'.bin'
        filepath = os.path.join(SAVE_PATH,file_name)
        f = open(filepath, 'rb')
        file = f.read()
        f.close()
        file = pickle.loads(file)
        
        print(f'Read from file: {file_name}:')
        print(f'Feature packs: {len(file["feature"])}')
        print(f'Captions: {len(file["captions"])}')
        
        return file
    
    def frame_to_feature(self, frames):
        # TODO: put code here to do feature embedding extraction
        feature = frames
        # ===========================
        
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
            'fps':-1,
            'tripping_feature_index':[],
            'tripping_caption_index':[],
            }
        return info
       
    def retrieve_captions(self,video_info, save = False):
        lines = []
        for file in self.caption_list:
            if file.split(".")[0] == video_info['video_info']:
                filepath = os.path.join(CAPTION_SOURCE_PATH,file)
                with open(rf'{filepath}',encoding='utf-8') as fd:
                    lines = list(srt.parse(fd))
        
        print(f'Get {len(lines)} lines of subtitles.')
        return lines
     
    def retrieve_frames(self,video_info, save = False):
        frames = {}
        video_path = os.path.join(VIDEO_SOURCE_PATH, video_info['video_name'])
        vid_cap = cv2.VideoCapture(video_path)
        video_info['fps'] = vid_cap.get(cv2.CAP_PROP_FPS)
        factor = round(video_info['fps']/self.extracted_fps)
    
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
        print(f"Packed {len(packed_frames)} packs from extracted {img_cnt} frames from total {cur_frame} frames.")
        return packed_frames

if __name__ == '__main__':
    
    pipe = ExtractionPipeline()
    #pipe.read()
    file = pipe.read_once(VID_1, over_write=True)
    #file_2 = pipe.read_from_saved_binary_file(VID_1)
