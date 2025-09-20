from .VAE.vae import VAE

def load_model(cfg):
    if cfg['name'] in ['VAE', 'BetaVAE']:
        return VAE.from_config(cfg)