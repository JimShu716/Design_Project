# -*- coding: utf-8 -*-
"""
Extraction pipeline for tempuckey dataset.
Author:
    Han Zhou
    Hao Shu
"""
import datetime
import logging
import pickle
import os
import cv2
import pandas as pd
import math
import torch
from tensorflow.keras.applications.resnet50 import ResNet50
import tensorflow.keras.preprocessing

from tensorflow.keras.applications.resnet50 import preprocess_input

from PIL import Image
import skimage.transform as st
import numpy as np
import string
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from numpy import argmax

import gensim.models.keyedvectors
from gensim.models import Word2Vec

# from nltk.corpus import stopwords

SAVE_PATH = '.\\feature\\'
VIDEO_SOURCE_PATH = '.\\videos\\'
CAPTION_SOURCE_PATH = '.\\captions\\'
LABEL_PATH = '.\\tempuckey_groundtruth_splits_videoinfo_20201026.csv'

SAVE_PATH_SERVER = '.\\feature\\'
VIDEO_SOURCE_PATH_SERVER = '/usr/local/data02/zahra/datasets/Tempuckey/all_videos_UNLABELED/TRIPPING'
CAPTION_SOURCE_PATH_SERVER = '/usr/local/data01/zahra/datasets/NHL_ClosedCaption/corpus_with_timestamp'
LABEL_PATH_SERVER = '/usr/local/data02/zahra/datasets/Tempuckey/labels/tempuckey_groundtruth_splits_videoinfo_20201026.csv'


#VOCABULARY_DATA_PATH = '.\\30flickr.txt'

VOCABULARY_PATH = './vocab/word_vocab_5_bow.pkl'
#WORD2VEC_PATH = '..\\word2vec\\feature.bin'

VID_1 = '1_TRIPPING_2017-11-28-fla-nyr-home_00_44_55.826000_to_00_45_06.437000.mp4'
VID_10 = '10_TRIPPING_2017-11-07-vgk-mtl-home_00_42_14.766000_to_00_42_24.142000.mp4'


class ExtractionPipeline():
    def __init__(self, num_video=10, extracted_fps=10, frame_per_package=15, on_server=False, suppress_log=True):
        self.caption_list = []
        self.video_list = []
        self.logging = ""

        # 15/10 => 1.5 sec per package
        self.extracted_fps = extracted_fps
        self.frame_per_package = frame_per_package
        self.num_video = num_video
        self.suppress_log = suppress_log

        if on_server:
            self.SAVE_PATH = SAVE_PATH_SERVER
            self.VIDEO_SOURCE_PATH = VIDEO_SOURCE_PATH_SERVER
            self.CAPTION_SOURCE_PATH = CAPTION_SOURCE_PATH_SERVER
            self.LABEL_PATH = LABEL_PATH_SERVER
        else:
            self.SAVE_PATH = SAVE_PATH
            self.VIDEO_SOURCE_PATH = VIDEO_SOURCE_PATH
            self.CAPTION_SOURCE_PATH = CAPTION_SOURCE_PATH
            self.LABEL_PATH = LABEL_PATH


        # init environment
        for p in self.SAVE_PATH, self.VIDEO_SOURCE_PATH, self.CAPTION_SOURCE_PATH:
            if not os.path.exists(p):
                os.mkdir(p)

        # init caption lookup list
        dirpath, dirnames, files = next(os.walk(self.CAPTION_SOURCE_PATH))
        self.caption_list = files

        # init video lookup list
        dirpath, dirnames, files = next(os.walk(self.VIDEO_SOURCE_PATH))
        self.video_list = files

        # init label lookup list
        df = pd.read_csv(self.LABEL_PATH)
        df = df[df['action'] == 'tripping']
        self.label = df

        # if number of video to process is too large or less than 0, read all videos
        if self.num_video > len(self.video_list) or self.num_video < 0:
            self.num_video = len(self.video_list)

        self.log("Initializing the environment...")
        self.log(f"Video file source from \t\t: {self.VIDEO_SOURCE_PATH}")
        self.log(f"Caption file source from \t: {self.CAPTION_SOURCE_PATH}")
        self.log(f"Label file source from \t\t: {self.LABEL_PATH}")
        self.log(f"Save file to \t\t\t\t: {self.SAVE_PATH}\n")
        self.log(f"Found {len(self.video_list)} videos and {len(self.caption_list)} subtitle files from environment.\n")

        # =========== Construct the word dictionary (Using the 30 flickr)
        # vocabulary = self.txt_to_vocabulary(VOCABULARY_DATA_PATH)
        # word_dict = self.vocab_to_dict(vocabulary)

        self.dictionary = pd.read_pickle(VOCABULARY_PATH)

    def read(self, save=True, over_write=False):
        self.log(f"Start reading process...(Total Task number:{self.num_video})\n")
        task_cnt = 0
        try:
            for i in range(self.num_video):
                self.log(f"=== Task {i + 1}/{self.num_video}:")
                file = self.read_once(self.video_list[i], save, over_write)
                if not file == None:
                    task_cnt += 1
                self.log("===")
        except:
            logging.exception("message")
            self.log(f"Job disrrupted, stop at task {task_cnt}")
        self.log(f"Finish job with {task_cnt} file generated.")
        with open("log.txt", "w") as fp:
            fp.write(self.logging)
            fp.close()

    def read_once(self, video_name, save=True, over_write=False):
        self.log(f'{video_name}:')
        if not over_write:
            save_filepath = os.path.join(self.SAVE_PATH, video_name[:-4] + '_0.bin')
            if os.path.exists(save_filepath):
                self.log('Found saved feature file, skip this file.')
                return None

        video_info = self.process_video_name(video_name)

        captions = self.retrieve_captions(video_info)
        # print(captions)
        if (len(captions) == 0):
            self.log(f"Error: Did not find the subtitle for video {video_name}")
            return None

        frames = self.retrieve_frames(video_info)
        if (len(frames) == 0):
            self.log(f"Error: Loading frame error for video {video_name}")
            return None

        captions = self.caption_to_feature(frames, captions, video_info)

        feature = self.frame_to_feature(frames)
        #feature = frames
        #self.get_crtical_time(feature, captions, video_info)


        file = {
            'video_info': video_info,
            'captions': captions,
            'feature': feature,
        }

        assert len(feature) == len(captions)
        if save:
            for i in range(len(captions)):
                video_info['patch_id']=i
                file_patch = {
                    'video_info': video_info,
                    'captions': captions[i],
                    'feature': feature[i],
                }
                file_pickle = pickle.dumps(file_patch, protocol=2)
                file_name = video_info['video_name'][:-4] + '_' + str(i) + '.bin'
                filepath = os.path.join(self.SAVE_PATH, file_name)
                with open(filepath, 'wb') as f:
                    f.write(file_pickle)
            self.log(f'{len(captions)} File generated.')
        else:
            self.log("File not saved.")

        return file

    def read_from_saved_binary_file(self, file_name):
        file_name = file_name[:-4] + '.bin'
        filepath = os.path.join(self.SAVE_PATH, file_name)
        f = open(filepath, 'rb')
        file = f.read()
        f.close()
        file = pickle.loads(file)

        self.log(f'Read from file: {file_name}:')
        self.log(f'Feature packs: {len(file["feature"])}')
        self.log(f'Captions: {len(file["captions"])}')

        return file

    def process_video_name(self, video_name):
        vlist = video_name.split("_", 3)
        # remove '.mp4'
        vlist[3] = vlist[3][:-4]
        game_time = vlist[3].split("_to_")
        start_time = [int(t) for t in game_time[0].split(".")[0].split('_')]
        end_time = [int(t) for t in game_time[1].split(".")[0].split('_')]
        start_time = start_time[0] * 3600 + start_time[1] * 60 + start_time[2]
        end_time = end_time[0] * 3600 + end_time[1] * 60 + end_time[2]

        info = {
            'video_name': video_name,
            'video_id': vlist[0],
            'video_type': vlist[1],
            'video_info': vlist[2],
            'fps': 30,
            'factor': 1,
            'sec_per_package': self.frame_per_package / self.extracted_fps,
            'patch_length': self.frame_per_package/self.extracted_fps,
            'start_time': start_time,
            'end_time': end_time,
            'video_length': end_time - start_time,
            'tripping_feature_index': [],
            'tripping_caption_index': [],
            'patch_id': -1,
        }
        return info

    def retrieve_captions(self, video_info, save=False):
        content = {}
        video_info_name = video_info['video_info']
        for file in self.caption_list:
            if file.split(".")[0] == video_info_name:
                filepath = os.path.join(self.CAPTION_SOURCE_PATH, file)
                with open(filepath, 'rb') as f:     
                    content = pickle.load(f)

        self.log(f'Get {len(content)} lines of subtitles.')
        return content

    def retrieve_frames(self, video_info, save=False):
        frames = {}
        video_path = os.path.join(self.VIDEO_SOURCE_PATH, video_info['video_name'])
        vid_cap = cv2.VideoCapture(video_path)
        video_info['fps'] = round(vid_cap.get(cv2.CAP_PROP_FPS))
        factor = round(video_info['fps'] / self.extracted_fps)
        factor = 1 if factor == 0 else factor
        video_info['factor'] = factor

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
                img_cnt += 1
            cur_frame += 1

        acctual_video_length = cur_frame / vid_cap.get(cv2.CAP_PROP_FPS)
        video_info['actual_video_length'] = acctual_video_length

        # clean up
        vid_cap.release()
        cv2.destroyAllWindows()

        # pack frames
        frame_list = list(frames.values())
        packed_frames = [frame_list[i:i + self.frame_per_package] for i in
                         range(0, len(frame_list), self.frame_per_package)]

        self.log(f"Packed {len(packed_frames)} packs from extracted {img_cnt} frames from total {cur_frame} frames.")
        return packed_frames

    def log(self, string):
        self.logging += string + '\n'
        if not self.suppress_log:
            print(string)

    def get_critical_time(self, feature, captions, video_info):
        events = self.label[self.label["video"] == video_info["video_name"]]
        c_feature = []
        c_caption = []
        for event in events.iterrows():
            start_pack = math.floor(event[1].get('beg_frame') / video_info['factor'] / self.frame_per_package)
            end_pack = math.ceil(event[1].get('end_frame') / video_info['factor'] / self.frame_per_package)
            for i in range(start_pack, end_pack + 1):
                c_feature.append(i)
            # for cap in captions:
            #     start_time = cap.start.total_seconds()
            #     end_time = cap.end.total_seconds()
            #     if not (end_time < event[1].get('beg_ts') or start_time > event[1].get('end_ts')):
            #         c_caption.append(cap.index)

        video_info['tripping_feature_index'] = list(set(c_feature))
        # video_info['tripping_caption_index'] = list(set(c_caption))

    def frame_to_feature(self, frames):
        # =========convert into tensor object==================
        #print("======== Start converting to feature =====")
        model = ResNet50(weights='imagenet')
        feature = frames
        for i in range(len(frames)):
            for j in range(len(frames[i])):
                # ================convert into embeddings (tensor object)===========
                feature = self.extract_feature(frames[i][j], model)
                feature_torch = torch.tensor(feature)
                frames[i][j] = feature_torch
        # ================fill the list with zero tensors to make same length===========
        second_last_list_length = len(frames[len(frames) - 2])
        last_list_length = len(frames[len(frames) - 1])
        while len(frames[len(frames) - 1]) < second_last_list_length:
            additional_tensor = torch.tensor(np.zeros((1, 1000)))
            frames[len(frames) - 1].append(additional_tensor)
        return frames

    def extract_feature(self, frame, model):
        # ==========resize the frame to fit the model ========
        x = st.resize(frame, (224, 224, 3))
        x = np.expand_dims(x, axis=0)

        x = preprocess_input(x)
        feature = model.predict(x)
        return feature

    def caption_to_feature(self, frames, captions, video_info):
        all_sentence = ""  # String to store all the sentence
        feature = [[] for i in range(len(frames))]
        frame_start_time = video_info['start_time']
        video_length = video_info['actual_video_length']

        for cap, w in captions.items():
            all_sentence += " "
            all_sentence += w

            # 5.1s/ 1.5s = 3.4 -> 3
            # 6.15s /1.5s = 4.1 -> 4
            start_time = datetime.timedelta(hours=cap[0].hour, minutes=cap[0].minute,
                                            seconds=cap[0].second).total_seconds() - frame_start_time
            end_time = datetime.timedelta(hours=cap[1].hour, minutes=cap[1].minute,
                                          seconds=cap[1].second).total_seconds() - frame_start_time
            if start_time > round(video_length) or end_time < 0:
                continue
            start_index = int(start_time / video_info['sec_per_package'])
            end_index = int(end_time / video_info['sec_per_package'])
            for i in range(start_index, end_index + 1):
                if i < len(feature):
                    feature[i].append((cap, w))

        # ================ Convert all sentence to list ===========
        # all_sentence= all_sentence.lower() # all to lower case
        # all_sentence = all_sentence.translate(str.maketrans('', '', string.punctuation)) # remove all punctuations
        # word_list = all_sentence.split()

        # ================ Create stop word list ===========
        #       all_stopwords = stopwords.words('english')

        feature_result = self.caption_to_one_hot(feature, self.dictionary)  # generate one-hot encoding
        # feature_result = self.word2vec_embeddings(feature) # generate embeddings by word2vec

        return feature_result

    """
    Function to preprocess the raw caption text into one-hot encodings
    
    feature - raw caption feature
    word_dict - dictionary of words to construct one- hot
    
    Return: the processed caption feature
    """

    def caption_to_one_hot(self, feature, word_vocab):

        dict_size = word_vocab.__len__()

        for i in range(len(feature)):
            sentence = ""
            for j in range(len(feature[i])):
                timestamps = feature[i][j][0]
                sentence += feature[i][j][1]
            # print("sentence is =",sentence)
            sentence = sentence.lower()  # to lower case
            sentence = sentence.translate(str.maketrans('', '', string.punctuation))  # remove all punctuations
            sentence_word = sentence.split()

            integer_encoded_sentence = []

            for word in sentence_word:
                word_integer = word_vocab.__call__(word)
                if word_integer == -1:
                    continue
                integer_encoded_sentence.append(word_integer)

            # print(integer_encoded_sentence)

            # ================ Initialize matrix for one hot encoding===========
            # one_hot_sentence = []
            one_hot_sentence = np.zeros(dict_size).tolist()
            for idx in range(len(integer_encoded_sentence)):
                one_hot_sentence[integer_encoded_sentence[idx]] = 1.0
                # one_hot_sentence.append(initial_arr)

            one_hot_sentence = np.array(one_hot_sentence)
            feature[i] = one_hot_sentence

        return feature

    """
    Function to preprocess the data in txt file into word vocabulary
    
    Return: the extracted word vocabulary
    """

    def txt_to_vocabulary(self, file_path):
        word_vocab = ""

        #print("====== Start processing vocabulary ====")
        with open(file_path, 'rb') as reader:
            for line in reader:
                line = line.decode("utf-8")
                cap_id, caption = line.split(' ', 1)
                caption = caption.lower()  # all to lower case
                caption = caption.translate(str.maketrans('', '', string.punctuation))  # remove all punctuations
                word_vocab += ""
                word_vocab += caption

        vocab_result = word_vocab.split()

        # ========= Remove duplicates in the vocabulary ========
        vocab_set = set()
        final_vocab = []

        for word in vocab_result:
            if word not in vocab_set:
                vocab_set.add(word)
                final_vocab.append(word)

        return final_vocab

    """
    Function to preprocess the word vocabulary into word dictionary for one-hot
    
    Return: the word dictionary
    """

    def vocab_to_dict(self, vocabulary):

        #print("====== Start constructing dictionary ====")
        # integer encode
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(vocabulary)  # encode labels
        integer_encoded_list = integer_encoded.tolist()

        # ================ Construct a word dictionary===========
        word_dict = {}

        for key in vocabulary:
            for value in integer_encoded_list:
                word_dict[key] = value
                integer_encoded_list.remove(value)
                break
        #print("==== Dictionary Construction Completed =====")
        return (word_dict)

    """
    Function to generate word embedding by word2vec
    
    feature - feature with raw caption texts
    
    Return: the feature with sentence embeddings
    """

    def word2vec_embeddings(self, feature):
        # Load pretrained model 
        model = gensim.models.KeyedVectors.load_word2vec_format(WORD2VEC_PATH, binary=True, unicode_errors='ignore')

        for i in range(len(feature)):
            for j in range(len(feature[i])):
                timestamps = feature[i][j][0]
                sentence = feature[i][j][1]

                sentence = sentence.lower()  # to lower case
                sentence = sentence.translate(str.maketrans('', '', string.punctuation))  # remove all punctuations
                sentence_word = sentence.split()

                sentence_embeddings = []

                # ======== Generate word embeddings and sentence embeddings by pretrained word2vec
                for word in sentence_word:
                    word_embeddings = model[word]
                    sentence_embeddings.append(word_embeddings)

                feature[i][j] = (timestamps, sentence_embeddings)

                return feature


class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self, text_style):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.text_style = text_style

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return -1
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


if __name__ == '__main__':
    #pipe = ExtractionPipeline(num_video=-1, on_server=True, suppress_log=False)
    pipe = ExtractionPipeline(num_video=-1, on_server=True, suppress_log=False)
    pipe.read()
    #file = pipe.read_once(VID_10)

    # file_2 = pipe.read_from_saved_binary_file(VID_1)
    print('end')
