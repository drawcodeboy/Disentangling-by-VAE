import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from einops import rearrange
import os, sys

class Shape3DDataset(Dataset):
    def __init__(self, 
                 root='data/3dshapes',
                 mode='train'):
        self.root = root
        self.data_li = []
        self._check()
        
        data_num = self.__len__()
        train_ratio, val_ratio, test_ratio = 0.7, 0.1, 0.2
        train_size = int(train_ratio * data_num)
        val_size = int(val_ratio * data_num)
        test_size = data_num - train_size - val_size

        if mode == 'train':
            self.data_li = self.data_li[:train_size]
        elif mode == 'val':
            self.data_li = self.data_li[train_size:train_size + val_size]
        elif mode == 'test':
            self.data_li = self.data_li[train_size + val_size:]
        else:
            raise Exception("Check mode argument")

    def __len__(self):
        return len(self.data_li)
    
    def _scale(self, value, min_v, max_v):
        return (value - min_v) / (max_v - min_v)

    def __getitem__(self, idx):
        # label = generative factor
        image_path = self.data_li[idx]['image']
        label_path = self.data_li[idx]['label']
        image, label = np.load(image_path), np.load(label_path)
        
        # Image preprocessing
        image = image.astype(np.float32) / 255.
        image = torch.tensor(image)
        image = rearrange(image, 'h w c -> c h w')
        
        # generative factor에 대해 disentanglement 같은 metric을 쓴다면, regressor가 필요하다.
        # regressor가 y scale에 큰 영향을 받지는 않겠다만, generative factor의 스케일을 통일 해두어서 나쁠 건 없을 거 같다.
        
        # Factor 1, 2, 3, 4, 5, 6
        label[0] = self._scale(label[0], 0.0, 0.9)
        label[1] = self._scale(label[1], 0.0, 0.9)
        label[2] = self._scale(label[2], 0.0, 0.9)
        label[3] = self._scale(label[3], 0.75, 1.25)
        label[4] = self._scale(label[4], 0.0, 3.0)
        label[5] = self._scale(label[5], -30.0, 30.0)
        return image, label

    def _check(self):
        images_root = f"{self.root}/images"
        labels_root = f"{self.root}/labels"
        
        for image_filename, label_filename in zip(os.listdir(images_root), os.listdir(labels_root)):
            self.data_li.append({"image": f"{images_root}/{image_filename}", 
                                 "label": f"{labels_root}/{label_filename}"})
    
    @classmethod
    def from_config(cls, cfg):
        return cls(root=cfg['root'],
                   mode=cfg['mode'])

# Check random seed application
# ds = Shape3DDataset()
# print(ds.data_li[0])

# Check label scale
# ds = Shape3DDataset()
# print(ds[0][1])

# Check dataset size
# ds = Shape3DDataset(mode='train')
# train_len = len(ds)
# ds = Shape3DDataset(mode='val')
# val_len = len(ds)
# ds = Shape3DDataset(mode='test')
# test_len = len(ds)
# print(train_len + val_len + test_len)

# Check dataset image size
# ds = Shape3DDataset()
# image, label = ds[0]
# print(image.shape)

# from torch.utils.data import DataLoader
# import time
# ds = Shape3DDataset()
# dl = DataLoader(ds, shuffle=True, batch_size=32, drop_last=True, num_workers=16)

# start_time = time.time()
# next(iter(dl))
# elapsed_time = time.time() - start_time

# print(elapsed_time)