from torch.utils.data import Dataset
import h5py
import numpy as np

class Shape3DDataset(Dataset):
    def __init__(self, 
                 root='data/3dshapes.h5',
                 mode='train'):
        self.root = root

        # 데이터셋의 특성 상 h5 파일 구조는 이렇게 불러온 것만으로는 참조(handle) 상태에 해당하기 때문에
        # 메모리 걱정 없이 이렇게 올려두었음.
        self.dataset = h5py.File(self.root, 'r')

        data_num = len(self.dataset['images'])
        
        np.random.seed(42)
        self.data_li = np.arange(data_num)
        np.random.shuffle(self.data_li)
        
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
        image = self.dataset['images'][self.data_li[idx]]
        label = self.dataset['labels'][self.data_li[idx]]
        
        # Image preprocessing
        image = image.astype(np.float32) / 255.
        image = torch.tensor(image)
        
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
        dataset = h5py.File(self.root, 'r')
        
        images = dataset['images']  # array shape [480000,64,64,3], uint8 in range 256
        labels = dataset['labels']  # array shape [480000,6], float64
        
        return len(images)
    
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

# Check dataset
# ds = Shape3DDataset(mode='train')
# train_len = len(ds)
# ds = Shape3DDataset(mode='val')
# val_len = len(ds)
# ds = Shape3DDataset(mode='test')
# test_len = len(ds)
# print(train_len + val_len + test_len)