import torch
import torch.utils.data as data
import numpy as np
import json as jsonmod

from basic.util import getVideoId
from vocab import clean_str

import pickle
import os
import torch._utils
import torch
import io
#Tempuckey starts============================
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2


SAVE_PATH = '.\\feature\\'
SSP = '/usr/local/extstore01/zhouhan/Tempuckey/feature_somewhere'


VIDEO_MAX_LEN = 100

#Tempuckey ends============================

#VIDEO_MAX_LEN=64


def read_video_ids(cap_file):
    video_ids_list = []
    with open(cap_file, 'r') as cap_reader:
        for line in cap_reader.readlines():
            #if len(line.strip().split(' ')) < 2:
             #   continue
            cap_id, caption = line.strip().split(' ', 1)
            video_id = getVideoId(cap_id)
            if video_id not in video_ids_list:
                video_ids_list.append(video_id)
    return video_ids_list



def collate_frame_gru_fn(data):
    """
    Build mini-batch tensors from a list of (video, caption) tuples.
    """
    # Sort a data list by caption length
    if data[0][1] is not None:
        data.sort(key=lambda x: len(x[1]), reverse=True)
    videos, captions, cap_bows, idxs, cap_ids, video_ids = zip(*data)

    # Merge videos (convert tuple of 1D tensor to 4D tensor)
    video_lengths = [min(VIDEO_MAX_LEN,len(frame)) for frame in videos]
    frame_vec_len = len(videos[0][0])
    vidoes = torch.zeros(len(videos), max(video_lengths), frame_vec_len)
    videos_origin = torch.zeros(len(videos), frame_vec_len)
    vidoes_mask = torch.zeros(len(videos), max(video_lengths))
    for i, frames in enumerate(videos):
            end = video_lengths[i]
            vidoes[i, :end, :] = frames[:end,:]
            videos_origin[i,:] = torch.mean(frames,0)
            vidoes_mask[i,:end] = 1.0
    
    if captions[0] is not None:
        # Merge captions (convert tuple of 1D tensor to 2D tensor)
        lengths = [len(cap) for cap in captions]
        target = torch.zeros(len(captions), max(lengths)).long()
        words_mask = torch.zeros(len(captions), max(lengths))
        for i, cap in enumerate(captions):
            end = lengths[i]
            target[i, :end] = cap[:end]
            words_mask[i, :end] = 1.0
    else:
        target = None
        lengths = None
        words_mask = None


    cap_bows = torch.stack(cap_bows, 0) if cap_bows[0] is not None else None

    video_data = (vidoes, videos_origin, video_lengths, vidoes_mask)
    text_data = (target, cap_bows, lengths, words_mask)

    return video_data, text_data, idxs, cap_ids, video_ids


def collate_frame(data):

    videos, idxs, video_ids = zip(*data)

    # Merge videos (convert tuple of 1D tensor to 4D tensor)
    video_lengths = [min(VIDEO_MAX_LEN,len(frame)) for frame in videos]
    frame_vec_len = len(videos[0][0])
    vidoes = torch.zeros(len(videos), max(video_lengths), frame_vec_len)
    videos_origin = torch.zeros(len(videos), frame_vec_len)
    vidoes_mask = torch.zeros(len(videos), max(video_lengths))
    for i, frames in enumerate(videos):
            end = video_lengths[i]
            vidoes[i, :end, :] = frames[:end,:]
            videos_origin[i,:] = torch.mean(frames,0)
            vidoes_mask[i,:end] = 1.0

    video_data = (vidoes, videos_origin, video_lengths, vidoes_mask)

    return video_data, idxs, video_ids


def collate_text(data):
    if data[0][0] is not None:
        data.sort(key=lambda x: len(x[0]), reverse=True)
    captions, cap_bows, idxs, cap_ids = zip(*data)

    if captions[0] is not None:
        # Merge captions (convert tuple of 1D tensor to 2D tensor)
        lengths = [len(cap) for cap in captions]
        target = torch.zeros(len(captions), max(lengths)).long()
        words_mask = torch.zeros(len(captions), max(lengths))
        for i, cap in enumerate(captions):
            end = lengths[i]
            target[i, :end] = cap[:end]
            words_mask[i, :end] = 1.0
    else:
        target = None
        lengths = None
        words_mask = None


    cap_bows = torch.stack(cap_bows, 0) if cap_bows[0] is not None else None

    text_data = (target, cap_bows, lengths, words_mask)

    return text_data, idxs, cap_ids


class Dataset4DualEncoding(data.Dataset):
    """
    Load captions and video frame features by pre-trained CNN model.
    """

    def __init__(self, cap_file, visual_feat, bow2vec, vocab, n_caption=None, video2frames=None):
        # Captions
        self.captions = {}
        self.cap_ids = []
        self.video_ids = set()
        self.video2frames = video2frames
        with open(cap_file, 'r') as cap_reader:
            for line in cap_reader.readlines():
                #if len(line.strip().split(' ')) < 2:
                 #   continue
                cap_id, caption = line.strip().split(' ', 1)
                video_id = getVideoId(cap_id)
                self.captions[cap_id] = caption
                self.cap_ids.append(cap_id)
                self.video_ids.add(video_id)
        self.visual_feat = visual_feat
        self.bow2vec = bow2vec
        self.vocab = vocab
        self.length = len(self.cap_ids)
        # if n_caption is not None:
        #     assert len(self.video_ids) * n_caption == self.length, "%d != %d" % (len(self.video_ids) * n_caption, self.length)

    def __getitem__(self, index):
        cap_id = self.cap_ids[index]
        video_id = getVideoId(cap_id)

        # video
        frame_list = self.video2frames[video_id]
        frame_vecs = []
        for frame_id in frame_list:
            frame_vecs.append(self.visual_feat.read_one(frame_id))
        frames_tensor = torch.Tensor(frame_vecs)

        # text
        caption = self.captions[cap_id]
        if self.bow2vec is not None:
            cap_bow = self.bow2vec.mapping(caption)
            if cap_bow is None:
                cap_bow = torch.zeros(self.bow2vec.ndims)
            else:
                cap_bow = torch.Tensor(cap_bow)
        else:
            cap_bow = None

        if self.vocab is not None:
            tokens = clean_str(caption)
            caption = []
            caption.append(self.vocab('<start>'))
            caption.extend([self.vocab(token) for token in tokens])
            caption.append(self.vocab('<end>'))
            cap_tensor = torch.Tensor(caption)
        else:
            cap_tensor = None

        return frames_tensor, cap_tensor, cap_bow, index, cap_id, video_id

    def __len__(self):
        return self.length
     

class VisDataSet4DualEncoding(data.Dataset):
    """
    Load video frame features by pre-trained CNN model.
    """
    def __init__(self, visual_feat, video2frames=None, video_ids=None):
        self.visual_feat = visual_feat
        self.video2frames = video2frames
        
        if video_ids is None:
            self.video_ids = video2frames.keys()
        else:
            self.video_ids = video_ids
        self.length = len(self.video_ids)

    def __getitem__(self, index):
        video_id = self.video_ids[index]

        frame_list = self.video2frames[video_id]
        frame_vecs = []
        for frame_id in frame_list:
            frame_vecs.append(self.visual_feat.read_one(frame_id))
        frames_tensor = torch.Tensor(frame_vecs)

        return frames_tensor, index, video_id

    def __len__(self):
        return self.length


class TxtDataSet4DualEncoding(data.Dataset):
    """
    Load captions
    """
    def __init__(self, cap_file, bow2vec, vocab):
        # Captions
        self.captions = {}
        self.cap_ids = []
        with open(cap_file, 'r') as cap_reader:
            for line in cap_reader.readlines():
                #if len(line.strip().split(' ')) < 2:
                 #   continue
                cap_id, caption = line.strip().split(' ', 1)
                self.captions[cap_id] = caption
                self.cap_ids.append(cap_id)
        self.bow2vec = bow2vec
        self.vocab = vocab
        self.length = len(self.cap_ids)

    def __getitem__(self, index):
        cap_id = self.cap_ids[index]

        caption = self.captions[cap_id]
        if self.bow2vec is not None:
            cap_bow = self.bow2vec.mapping(caption)
            if cap_bow is None:
                cap_bow = torch.zeros(self.bow2vec.ndims)
            else:
                cap_bow = torch.Tensor(cap_bow)
        else:
            cap_bow = None

        if self.vocab is not None:
            tokens = clean_str(caption)
            caption = []
            caption.append(self.vocab('<start>'))
            caption.extend([self.vocab(token) for token in tokens])
            caption.append(self.vocab('<end>'))
            cap_tensor = torch.Tensor(caption)
        else:
            cap_tensor = None

        return cap_tensor, cap_bow, index, cap_id

    def __len__(self):
        return self.length

def get_data_loaders(cap_files, visual_feats, vocab, bow2vec, batch_size=100, num_workers=1, n_caption=2, video2frames=None, padding_size=0):
    """
    Returns torch.utils.data.DataLoader for train and validation datasets
    Args:
        cap_files: caption files (dict) keys: [train, val]
        visual_feats: image feats (dict) keys: [train, val]
    """
    dset = {'train': Dataset4DualEncoding(cap_files['train'], visual_feats['train'], bow2vec, vocab, video2frames=video2frames['train']),
            'val': Dataset4DualEncoding(cap_files['val'], visual_feats['val'], bow2vec, vocab, n_caption, video2frames=video2frames['val']) }

    data_loaders = {x: torch.utils.data.DataLoader(dataset=dset[x],
                                    batch_size=batch_size,
                                    shuffle=(x=='train'),
                                    pin_memory=True,
                                    num_workers=num_workers,
                                    collate_fn=collate_frame_gru_fn)
                        for x in cap_files }
    return data_loaders


def get_train_data_loaders(cap_files, visual_feats, vocab, bow2vec, batch_size=100, num_workers=1, n_caption=2, video2frames=None):
    """
    Returns torch.utils.data.DataLoader for train and validation datasets
    Args:
        cap_files: caption files (dict) keys: [train, val]
        visual_feats: image feats (dict) keys: [train, val]
    """
    dset = {'train': Dataset4DualEncoding(cap_files['train'], visual_feats['train'], bow2vec, vocab, video2frames=video2frames['train'])}

    data_loaders = {x: torch.utils.data.DataLoader(dataset=dset[x],
                                    batch_size=batch_size,
                                    shuffle=False,#(x=='train'),
                                    pin_memory=True,
                                    num_workers=num_workers,
                                    collate_fn=collate_frame_gru_fn)
                        for x in cap_files if x=='train' }
    return data_loaders


def get_test_data_loaders(cap_files, visual_feats, vocab, bow2vec, batch_size=100, num_workers=1, n_caption=2, video2frames = None):
    """
    Returns torch.utils.data.DataLoader for test dataset
    Args:
        cap_files: caption files (dict) keys: [test]
        visual_feats: image feats (dict) keys: [test]
    """
    dset = {'test': Dataset4DualEncoding(cap_files['test'], visual_feats['test'], bow2vec, vocab, n_caption, video2frames = video2frames['test'])}


    data_loaders = {x: torch.utils.data.DataLoader(dataset=dset[x],
                                    batch_size=batch_size,
                                    shuffle=False,
                                    pin_memory=True,
                                    num_workers=num_workers,
                                    collate_fn=collate_frame_gru_fn)
                        for x in cap_files }
    return data_loaders


def get_vis_data_loader(vis_feat, batch_size=100, num_workers=1, video2frames=None, video_ids=None):
    dset = VisDataSet4DualEncoding(vis_feat, video2frames, video_ids=video_ids)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=num_workers,
                                              collate_fn=collate_frame)
    return data_loader


def get_txt_data_loader(cap_file, vocab, bow2vec, batch_size=100, num_workers=1):
    dset = TxtDataSet4DualEncoding(cap_file, bow2vec, vocab)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=num_workers,
                                              collate_fn=collate_text)
    return data_loader
#Tempuckey starts============================

def collate(data):
    # Sort a data list by caption length
    # if data[0][1] is not None:
    #     data.sort(key=lambda x: len(x[1]), reverse=True)
    # videos, captions, cap_bows, idxs, cap_ids, video_ids = zip(*data)

    videos, video_infos, captions, caption_lengths = zip(*data)

    # Merge videos (convert tuple of 1D tensor to 4D tensor)
    frame_vec_len = len(videos[0][0][0])
    video_lengths = [min(VIDEO_MAX_LEN, len(frame)) for frame in videos]
    video_datas = torch.zeros(len(videos), max(video_lengths), frame_vec_len)
    video_means = torch.zeros(len(videos), frame_vec_len)
    video_masks = torch.zeros(len(videos), max(video_lengths))
    video_names = [info['video_name'] for info in video_infos]

    for i, video in enumerate(videos):
        end = video_lengths[i]
        video = [v[0] for v in video]
        video = torch.stack(video)
        video_datas[i, :end, :] = video[:end, :]
        video_means[i, :] = torch.mean(video, 0)
        video_masks[i, :end] = 1.0


    # Merge captions (convert tuple of 1D tensor to 2D tensor)
    cap_lengths = [len(cap) for cap in captions]
    cap_datas = torch.zeros(len(captions), max(cap_lengths)).long()
    cap_masks = torch.zeros(len(captions), max(cap_lengths))

    for i, cap in enumerate(captions):
        end = cap_lengths[i]
        cap_datas[i, :end] = cap[:end]
        cap_masks[i, :end] = 1.0

    #cap_bows = torch.stack(cap_bows, 0) if cap_bows[0] is not None else None
    #TODO: bow2vec
    cap_bows = None

    video_data_pack = (video_datas,
                       video_means,
                       video_lengths,
                       video_masks,
                       video_names)

    text_data_pack = (cap_datas,
                      cap_bows,
                      cap_lengths,
                      cap_masks)

    return video_data_pack, text_data_pack


"""
A class to solve unpickling issues
"""
class CPU_Unpickler(pickle.Unpickler,object):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super(CPU_Unpickler,self).find_class(module, name)

print (type(CPU_Unpickler))

class TempuckeyDataSet(data.Dataset):
    def __init__(self, read_path=SSP):
        self.read_path = read_path
        _, _, self.file_pool = next(os.walk(read_path))
        self.length = len(self.file_pool)
        print 'Initializing TempuckeyDataSet...'
        print 'Read path: %s' % self.read_path
        print 'Find %d files in the path.' % self.length

    def __getitem__(self, index):
        file_path = os.path.join(self.read_path, self.file_pool[index])
        
        with open(file_path, 'rb') as f:
        
            file = CPU_Unpickler(f).load()
          
        video = file['feature']
        video_info = file['video_info']
        caption = (file['captions'])
        caption_length = np.count_nonzero(caption == 1.0)

        return video, video_info, caption, caption_length

    def __len__(self):
        return self.length


def get_tempuckey_data_loader(batch_size=10, num_workers=1):
    """
    Returns torch.utils.data.DataLoader for train and validation datasets
    Args:
        cap_files: caption files (dict) keys: [train, val]
        visual_feats: image feats (dict) keys: [train, val]
        :param num_workers: 
        :param batch_size: 
    """
    
    data_loader = torch.utils.data.DataLoader(dataset=TempuckeyDataSet(),
                                    batch_size=batch_size,
                                    shuffle=True,
                                    pin_memory=True,
                                    num_workers=num_workers,
                                    collate_fn=collate)
    return data_loader
#Tempuckey ends============================


if __name__ == '__main__':
    pass
