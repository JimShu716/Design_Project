from data_process.Tempuckey.extraction_pipeline import SAVE_PATH
import torch.utils.data as data
import pickle
import os


class TempuckeyDataSet(data.Dataset):
    def __init__(self, read_path=SAVE_PATH):
        self.read_path = read_path
        _,_,self.file_pool = next(os.walk(read_path))
        self.length = len(self.file_pool)
        print(f'Initailize TempuckeyDataSet...')
        print(f'Read path: {self.read_path}')
        print(f'Find {self.length} files in the path.')

    def __getitem__(self, index):
        file_path = os.path.join(self.read_path, self.file_pool[index])
        with open(file_path, 'rb') as f:
            file = pickle.loads(f.read())
        video = (file['feature'], file['video_info']['patch_length'])
        caption = (file['captions'])
        return video, caption

    def __len__(self):
        return self.length


if __name__ == '__main__':
    tt = TempuckeyDataSet()
    file = tt.__getitem__(0)
    print('test break pt.')