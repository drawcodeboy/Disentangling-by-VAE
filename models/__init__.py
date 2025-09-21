from .VAE.vae import VAE
from .VAE.vovae import VOVAE

def load_model(cfg):
    if cfg['name'] in ['VAE', 'BetaVAE']:
        return VAE.from_config(cfg)
    elif cfg['name'] in ['VOVAE']:
        return VOVAE.from_config(cfg)