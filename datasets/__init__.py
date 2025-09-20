from .shape3d_dataset import Shape3DDataset

def load_dataset(cfg):
    if cfg['name'] =='shape3d':
        return Shape3DDataset.from_config(cfg)