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
        # 1) Regularization
        first_term = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()
        
        # 2) Reconstruction
        second_term = F.mse_loss(x_prime, x)
        
        minus_elbo = self.beta * first_term + second_term
        
        return minus_elbo, first_term, second_term