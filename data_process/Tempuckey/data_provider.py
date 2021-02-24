
import numpy as np
import torch.utils.data as data
import pickle
import os

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
        with open(file_path, 'rb') as f:
            file = pickle.loads(f.read())
        video = (file['feature'], file['video_info']['patch_length'])
        caption = (file['captions'])
        caption_length = np.count_nonzero(caption == 1.0)
        
        print(caption_length)
        return video, caption,caption_length

    def __len__(self):
        return self.length


if __name__ == '__main__':
    tt = TempuckeyDataSet()
    file = tt.__getitem__(3)
    print('test break pt.')


