import torch
from torch import nn

import argparse
import time, sys, os, yaml
sys.path.append(os.getcwd())

from utils import validate, evaluate
from models import load_model
from datasets import load_dataset

import pdb
from einops import rearrange
import cv2
import numpy as np

def add_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', type=str)
    
    return parser

def main(cfg, config):
    print(f"=====================[{cfg['title']}]=====================")
    
    # Device Setting
    device = None
    if cfg['device'] != 'cpu' and torch.cuda.is_available():
        device = cfg['device']
    else: 
        device = 'cpu'
    print(f"device: {device}")
    
    # Hyperparameter Settings
    hp_cfg = cfg['hyperparameters']
    
    # Other Important Configurations
    task_cfg = cfg['task']
    save_cfg = cfg['save']
    
    # Load Dataset
    test_data_cfg = cfg['data']['test']
    test_ds = load_dataset(test_data_cfg)
    print(f"Load Dataset {test_data_cfg['name']}")
    
    # Load Model
    model_cfg = cfg['model']
    model = load_model(model_cfg).to(device)
    ckpt = torch.load(os.path.join(save_cfg['weights_path'], save_cfg['weights_filename']),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    
    sample_index = np.random.randint(0, len(test_ds))
    print(f"sample index: {sample_index}")
    x, label = test_ds[sample_index]
    x = rearrange(x, 'c h w -> 1 c h w').to(device)
    
    x_prime, z, mu, log_var = model(x)
    x_inter = model.intervention(num_samples=1, 
                                 z=z,
                                 device=device)
    
    x = x.cpu().detach().numpy()
    x_prime = x_prime.cpu().detach().numpy()
    x_inter = x_inter.cpu().detach().numpy()

    # print(x.min(), x.max(), x.mean())
    # print(x_prime.min(), x_prime.max(), x_prime.mean())
    x_prime = np.clip(x_prime, 0., 1.) # 픽셀 값이 튀어서 결과 이상하게 나옴, 이 제약을 필요로 함.
    # print(x_inter.min(), x_inter.max(), x_inter.mean())
    x_inter = np.clip(x_inter, 0., 1.) # 픽셀 값이 튀어서 결과 이상하게 나옴, 이 제약을 필요로 함.
    
    x = (np.transpose(x[0], (1, 2, 0)) * 255.).astype(np.uint8)
    x_prime = (np.transpose(x_prime[0], (1, 2, 0)) * 255.).astype(np.uint8)
    x_inter = (np.transpose(x_inter[0], (1, 2, 0)) * 255.).astype(np.uint8)
    
    model_name = config.split('.')[0]
    os.makedirs('assets/intervention', exist_ok=True)
    cv2.imwrite(f'assets/intervention/{model_name}_GT.jpg', x)
    cv2.imwrite(f'assets/intervention/{model_name}_reconstruction.jpg', x_prime)
    cv2.imwrite(f'assets/intervention/{model_name}_intervention.jpg', x_inter)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Test', parents=[add_args_parser()])
    args = parser.parse_args()

    with open(f'configs/test/{args.config}.yaml') as f:
        cfg = yaml.full_load(f)
    
    main(cfg, args.config)