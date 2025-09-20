import torch
from models import load_model

cfg = {
    'name': 'VAE',
    'latent': 6,
    'img_size': (3, 64, 64)
}

model = load_model(cfg)
x = torch.randn([128, 3, 64, 64])
x_prime, z, mu, log_var = model(x)

print(x_prime.shape, z.shape, mu.shape, log_var.shape)