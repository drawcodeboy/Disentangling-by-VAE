from .encoder import Encoder
from .decoder import Decoder

import torch
from torch import nn

from einops import rearrange
from typing import List, Tuple

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

    def sample(self, num_samples, device):
        z = torch.randn((num_samples, self.latent)).to(device)
        
        samples = self.decoder(z)
        
        return samples

    @classmethod
    def from_config(cls, cfg):
        return cls(latent=cfg['latent'],
                   img_size=cfg['img_size'])