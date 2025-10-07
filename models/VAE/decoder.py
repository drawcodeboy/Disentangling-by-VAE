import torch
from torch import nn

from typing import List, Tuple

from einops import rearrange
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self,
                 latent:int = 6):
        super().__init__()
        
        self.latent = latent
        
        vector_dims = 256
        self.from_z_li = nn.Linear(self.latent, 256)
        self.from_z_acti = nn.ReLU()
        
        self.fc2 = nn.Linear(256, 4*4*64)
        self.acti = nn.ReLU()

        self.block_li = nn.ModuleList([
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        ])
        
    def forward(self, x):
        x = self.from_z_acti(self.from_z_li(x))
        x = self.acti(self.fc2(x))
        x = rearrange(x, 'b (c w h) -> b c w h', c=64, w=4, h=4)
        
        for block in self.block_li:
            x = block(x)
        
        # Sigmoid 쓰지 말 것, 가운데 몰림 현상 때문인지 결과가 뿌옇게 나옴
        # x = F.sigmoid(x)
        
        return x