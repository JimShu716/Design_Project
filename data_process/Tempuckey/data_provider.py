
import numpy as np
import torch.utils.data as data
import pickle
import os
import torch._utils
import torch
import io

try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2


# SAVE_PATH = '.\\feature\\'
SSP = '/usr/local/extstore01/zhouhan/Tempuckey/feature_somewhere'

VIDEO_MAX_LEN = 100

def collate(data):
    # Sort a data list by caption length
    # if data[0][1] is not None:
    #     data.sort(key=lambda x: len(x[1]), reverse=True)
    # videos, captions, cap_bows, idxs, cap_ids, video_ids = zip(*data)

    videos, video_infos, captions, bow = zip(*data)

    # Merge videos (convert tuple of 1D tensor to 4D tensor)
    frame_vec_len = len(videos[0][0][0])
    video_lengths = [min(VIDEO_MAX_LEN, len(frame)) for frame in videos]
    video_datas = torch.zeros(len(videos), max(video_lengths), frame_vec_len)
    video_means = torch.zeros(len(videos), frame_vec_len)
    video_masks = torch.zeros(len(videos), max(video_lengths))
    video_names = [info['video_name'] for info in video_infos]

    for i, video in enumerate(videos):
        end = video_lengths[i]
        video = [v[0].float() for v in video]
        video = torch.stack(video)
        video_datas[i, :end, :] = video[:end, :]
        video_means[i, :] = torch.mean(video, 0)
        video_masks[i, :end] = 1.0

    # Merge captions (convert tuple of 1D tensor to 2D tensor)
    cap_lengths = [len(cap) for cap in captions]
    cap_datas = torch.zeros(len(captions), max(cap_lengths)).long()
    cap_masks = torch.zeros(len(captions), max(cap_lengths))

    for i, cap in enumerate(bow):
        cap = torch.from_numpy(cap)
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
        bow = file['bow']

        return video, video_info, caption, bow

    def __len__(self):
        return self.length


def get_data_loader(batch_size=10, num_workers=2):
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


if __name__ == '__main__':
    data_loader = get_data_loader()
    for i, (video, caps) in enumerate(data_loader):
        print 'enum ========== '+str(i)
    print 'test break pt.'
