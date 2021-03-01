
import numpy as np
import torch.utils.data as data
import pickle
import os
import torch._utils
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


SAVE_PATH = '.\\feature\\'

def collate(data):
    # Sort a data list by caption length
    if data[0][1] is not None:
        data.sort(key=lambda x: len(x[1]), reverse=True)
    videos, captions, cap_bows, idxs, cap_ids, video_ids = zip(*data)

    # Merge videos (convert tuple of 1D tensor to 4D tensor)
    video_lengths = [min(VIDEO_MAX_LEN, len(frame)) for frame in videos]
    frame_vec_len = len(videos[0][0])
    vidoes = torch.zeros(len(videos), max(video_lengths), frame_vec_len)
    videos_origin = torch.zeros(len(videos), frame_vec_len)
    vidoes_mask = torch.zeros(len(videos), max(video_lengths))
    for i, frames in enumerate(videos):
        end = video_lengths[i]
        vidoes[i, :end, :] = frames[:end, :]
        videos_origin[i, :] = torch.mean(frames, 0)
        vidoes_mask[i, :end] = 1.0

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


"""
A class to solve unpickling issues

"""
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)


class TempuckeyDataSet(data.Dataset):
    def __init__(self, read_path=SAVE_PATH):
        self.read_path = read_path
        _, _, self.file_pool = next(os.walk(read_path))
        self.length = len(self.file_pool)
#        print 'Initializing TempuckeyDataSet...'
  #      print 'Read path: %s' % self.read_path
 #       print 'Find %d files in the path.' % self.read_path

    def __getitem__(self, index):
        file_path = os.path.join(self.read_path, self.file_pool[index])
        f= open(file_path, 'rb')
        file = CPU_Unpickler(f).load()
        f.close()
        video = (file['feature'], file['video_info']['patch_length'])
        caption = (file['captions'])
        caption_length = np.count_nonzero(caption == 1.0)
        
        print(caption_length)
        return video, caption,caption_length

    def __len__(self):
        return self.length


if __name__ == '__main__':
    tt = TempuckeyDataSet()
    file = tt.__getitem__(0)
   #print 'test break pt.'