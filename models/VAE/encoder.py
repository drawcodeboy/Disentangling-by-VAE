import torch
from torch import nn
from torch.distributions.normal import Normal

from typing import List, Tuple
from einops import rearrange
import pdb

class Encoder(nn.Module):
    def __init__(self,
                 latent:int = 6,
                 img_size:Tuple = (64, 64)):
        super().__init__()
        
        # (H, W) -> (H/32, W/32)
        self.block_li = nn.ModuleList([
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=0),
            nn.ReLU()
        ])
            
        self.latent = latent
        
        vector_dims = 64 * (img_size[0] // 32) * (img_size[1] // 32) # (out_channels * feature_map_H * feature_map_W)
        self.mu = nn.Linear(vector_dims, self.latent)
        self.log_var = nn.Linear(vector_dims, self.latent)
        
    def forward(self, x):
        for block in self.block_li:
            x = block(x)
        
        # Flatten to extract mu, log_var
        x = x.flatten(start_dim=1)
        
        # x = self.acti(self.li_bn(self.li(x)))
        mu = self.mu(x)
        log_var = self.log_var(x)
        
        z = self.reparameterization_trick(mu, log_var)

        return z, mu, log_var
    
    def reparameterization_trick(self, mu, log_var):
        gaussian = Normal(loc=torch.zeros(mu.size()),
                          scale=torch.ones(log_var.size()))
        
        eps = gaussian.sample().to(log_var.device)
        
        std = torch.exp(0.5 * log_var) # (1) 0.5 * log_var -> log_std, (2) torch.exp(log_std) -> std
        
        z = mu + eps * std # Reparameterization
        return z