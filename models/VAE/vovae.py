from .encoder import Encoder
from .decoder import Decoder

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange
from typing import List, Tuple

import pdb

class VOVAE(nn.Module):
    def __init__(self,
                 latent:int = 6, # Number of Generative factor
                 img_size:Tuple = (3, 64, 64),
                 latent_vec_dim:int = 128,
                 orth_coef:float = 0.1):
        super().__init__()
        
        self.latent = latent
        self.img_size = img_size
        
        self.encoder = Encoder(latent = self.latent,
                               img_size = (img_size[1], img_size[2]))
        self.decoder = Decoder(latent = self.latent)
        
        self.latent_vec_li = nn.ModuleList([
            nn.Linear(1, latent_vec_dim) for _ in range(0, self.latent)
        ])
        
        self.orth_coef = orth_coef

    def forward(self, x):
        z, mu, log_var = self.encoder(x)
        
        latent_unit_vec_li = []
        for idx, latent_li in enumerate(self.latent_vec_li):
            latent_unit = mu[:, idx]
            latent_unit = rearrange(latent_unit, 'b -> b 1')
            latent_unit_vec = latent_li(latent_unit)
            latent_unit_vec_li.append(latent_unit_vec)
        orth_loss = self.get_orth_loss(latent_unit_vec_li)
        
        x = self.decoder(z)
        return x, z, mu, log_var, orth_loss

    def intervention(self, num_samples, z, device):
        # 의도적인 intervention
        
        i = 3
        random = torch.randn((1,)).to(device) * 3.
        z[0, i] = z[0, i] + random
        
        samples = self.decoder(z)
        
        return samples

    def get_orth_loss(self, latent_unit_vec_li):
        """
        latent_unit_vec_li: list of tensors, each shape (batch_size, d)
        """
        num_tensors = len(latent_unit_vec_li)
        batch_size, d = latent_unit_vec_li[0].shape

        # (num_tensors, batch_size, d)
        vectors = torch.stack(latent_unit_vec_li, dim=0)

        # 단위 벡터라면 굳이 normalize 필요 없음, 혹시 모르니 normalize
        vectors = F.normalize(vectors, p=2, dim=-1)

        # (batch_size, num_tensors, d) 로 reshape
        vectors = vectors.permute(1, 0, 2)

        # (batch_size, num_tensors, num_tensors) → pairwise cosine similarity matrix
        sim_matrix = torch.bmm(vectors, vectors.transpose(1, 2))

        # 대각선은 자기 자신과의 similarity라 제외해야 함
        mask = torch.eye(num_tensors, device=vectors.device).unsqueeze(0)  # (1, num_tensors, num_tensors)
        sim_matrix = sim_matrix * (1 - mask)

        # 모든 pairwise cosine similarity 제곱 평균
        loss = (sim_matrix**2).sum(dim=(1, 2)) / (num_tensors * (num_tensors - 1))

        # batch 평균 후 coef 곱
        loss = self.orth_coef * loss.mean()
        return loss
    '''
    def get_orth_loss(self, latent_unit_vec_li):
        """
        latent_unit_vec_li: list of tensors, each shape (batch_size, d)
        """
        num_tensors = len(latent_unit_vec_li)
        batch_size, d = latent_unit_vec_li[0].shape

        loss = 0.0
        count = 0

        for b in range(batch_size):
            # 같은 샘플 index에 해당하는 벡터 모음 (num_tensors, d)
            vectors = [t[b] for t in latent_unit_vec_li]
            vectors = torch.stack(vectors, dim=0)  # (num_tensors, d)

            # 모든 pairwise cosine similarity
            for i in range(num_tensors):
                for j in range(i + 1, num_tensors):
                    sim = F.cosine_similarity(
                        vectors[i].unsqueeze(0), vectors[j].unsqueeze(0), dim=1
                    )
                    loss += sim**2 # [-1, 1] 값을 갖기 떄문에, 제곱으로 했음 abs는 gradient 불안정성 문제도 있으니까
                    count += 1

        loss = loss / count # Batch size에 대한 평균
        loss = self.orth_coef * loss # loss 영향력(?) 제어
        loss = loss.squeeze() # Shape: [1] -> []
        return loss
    '''
    
    @classmethod
    def from_config(cls, cfg):
        return cls(latent=cfg['latent'],
                   img_size=cfg['img_size'],
                   latent_vec_dim=cfg['latent_vec_dim'],
                   orth_coef=cfg['orth_coef'])