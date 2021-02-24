
import numpy as np
import torch.utils.data as data
import pickle
import os
import torch._utils

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



class TempuckeyDataSet(data.Dataset):
    def __init__(self, read_path=SAVE_PATH):
        self.read_path = read_path
        _,_,self.file_pool = next(os.walk(read_path))
        self.length = len(self.file_pool)
        print('Initailize TempuckeyDataSet...')
        print('Read path:',self.read_path)
        print('Number of files in the path:',self.length)

    def __getitem__(self, index):
        file_path = os.path.join(self.read_path, self.file_pool[index])
        f= open(file_path, 'rb')
        file = pickle.load(f)
        f.close()
        video = (file['feature'], file['video_info']['patch_length'])
        caption = (file['captions'])
        caption_length = np.count_nonzero(caption == 1.0)
        
        print(caption_length)
        return video, caption,caption_length

    def __len__(self):
        return self.length



tt = TempuckeyDataSet()
file = tt.__getitem__(4)
print('test break pt.')


