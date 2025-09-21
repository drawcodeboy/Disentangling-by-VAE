from .dci import dci
import numpy as np

def get_metrics(total_x, total_x_prime, total_codes, total_factors):
    disentanglement, completeness, informativeness = dci(total_factors, total_codes)
    mae = np.mean(np.abs(total_x - total_x_prime))
    
    result = {
        'DCI': disentanglement,
        # 'Completeness': completeness,
        # 'Informativeness': informativeness,
        'MAE': mae
    }
    
    return result