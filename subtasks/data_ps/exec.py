import os, sys
sys.path.append(sys.path.append(os.getcwd()))
import h5py
import cv2
import numpy as np

# Code from "https://github.com/google-deepmind/3d-shapes/blob/master/3dshapes_loading_example.ipynb"
# load dataset
dataset = h5py.File('data/3dshapes.h5', 'r')
print(dataset.keys())
images = dataset['images']  # array shape [480000,64,64,3], uint8 in range(256)
labels = dataset['labels']  # array shape [480000,6], float64
image_shape = images.shape[1:]  # [64,64,3]
label_shape = labels.shape[1:]  # [6]
n_samples = labels.shape[0]  # 10*10*10*8*4*15=480000

_FACTORS_IN_ORDER = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape',
                     'orientation']
_NUM_VALUES_PER_FACTOR = {'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10, 
                          'scale': 8, 'shape': 4, 'orientation': 15}

sample_index = 10000

image_sample = cv2.resize(images[sample_index], (512, 512))

cv2.imwrite('subtasks/data_ps/sample_3dshape.png', image_sample)

print(f"label sample: {labels[sample_index]}")

for factor_index in range(0, label_shape[0]):
    print(f"factor {factor_index+1}: min={min(labels[:, factor_index])}, max={max(labels[:, factor_index])}")
    #unique_values, counts = np.unique(labels[:, factor_index], return_counts=True)
    # print(f"unique values: {unique_values}, counts: {counts}")

'''
from pympler import asizeof
print(asizeof.asizeof(dataset)) # handle(참조 상태) -> 용량 적음
print(asizeof.asizeof(image_sample)) # image(실제 데이터 상태) -> 용량 큼
print(asizeof.asizeof(images)) # handle(참조 상태) -> 용량 적음
'''