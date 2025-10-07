import torch
from torch import nn
import torch.distributions as dist
import torch.nn.functional as F

class ELBO(nn.Module):
    def __init__(self,
                 beta=1.0):
        super().__init__()
        self.beta = beta # if beta == 1.0 -> VanilaVAE, beta > 1.0 -> BetaVAE 
    
    def forward(self, x_prime, x, mu, log_var):
        # Reconstruction Loss 관련 Line 429-433 참고
        # https://github.com/YannDubs/disentangling-vae/blob/master/disvae/models/losses.py#L394

        # dim별로는 summation, batch별로는 mean을 구하는 구조

        # 1) Regularization
        first_term = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()
        
        # 2) Reconstruction
        n = x.size(0)
        # second_term = F.mse_loss(x_prime * 255., x * 255., reduction='sum') / (255. * n)
        second_term = F.mse_loss(x_prime, x, reduction='sum') / n

        minus_elbo = self.beta * first_term + second_term
        
        return minus_elbo, first_term, second_term