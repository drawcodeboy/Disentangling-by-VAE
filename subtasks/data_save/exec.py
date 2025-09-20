import os, sys
sys.path.append(sys.path.append(os.getcwd()))
import h5py
import cv2
import numpy as np
import time

dataset = h5py.File('data/3dshapes.h5', 'r')
print(dataset.keys())
images = dataset['images']  # array shape [480000,64,64,3], uint8 in range(256)
labels = dataset['labels']  # array shape [480000,6], float64

os.makedirs('data/3dshapes/images', exist_ok=True)
os.makedirs('data/3dshapes/labels', exist_ok=True)

start_time = time.time()
arr = np.arange(0, 480000)
np.random.seed(42)
np.random.shuffle(arr)

data_num = 10000 # 필요한 데이터의 수
arr = arr[:data_num]

for i, random_index in enumerate(arr):
    np.save(f'data/3dshapes/images/{i+1:06d}.npy', images[random_index])
    np.save(f'data/3dshapes/labels/{i+1:06d}.npy', labels[random_index])
    now_time = int(time.time() - start_time)
    print(f"\rProcessing [{i+1:06d}/{data_num:06d}] ({(i+1)/data_num*100:.6f}%) | [{now_time//60}m {now_time%60}s]" ,end="")