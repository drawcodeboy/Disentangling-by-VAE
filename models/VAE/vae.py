from .encoder import Encoder
from .decoder import Decoder

import torch
from torch import nn

from einops import rearrange
from typing import List, Tuple
import numpy as np

class VAE(nn.Module):
    def __init__(self,
                 latent:int = 6, # Number of Generative factor
                 img_size:Tuple = (3, 64, 64)):
        super().__init__()
        
        self.latent = latent
        self.img_size = img_size
        
        self.encoder = Encoder(latent = self.latent,
                               img_size = (img_size[1], img_size[2]))
        self.decoder = Decoder(latent = self.latent)

    def forward(self, x):
        z, mu, log_var = self.encoder(x)
        
        x = self.decoder(z)
        return x, z, mu, log_var

    def intervention(self, num_samples, z, device):
        # 의도적인 intervention
        
        i = np.random.randint(0, 6)
        random = torch.randn((1,)).to(device) * 3.
        z[0, i] = z[0, i] + random
        
        samples = self.decoder(z)
        
        return samples
    
    def sample2(self, mu):
        return self.decoder(mu)

    @classmethod
    def from_config(cls, cfg):
        return cls(latent=cfg['latent'],
                   img_size=cfg['img_size'])