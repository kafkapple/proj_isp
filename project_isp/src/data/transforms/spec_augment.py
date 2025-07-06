import torch
import random

class SpecAugment:
    def __init__(self, freq_mask_param=20, time_mask_param=20):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        
    def __call__(self, x):
        """
        x: (batch_size, time_steps, freq_bins)
        """
        # Frequency masking
        f = random.randint(0, self.freq_mask_param)
        f0 = random.randint(0, x.shape[2] - f)
        x[:, :, f0:f0 + f] = 0
        
        # Time masking
        t = random.randint(0, self.time_mask_param)
        t0 = random.randint(0, x.shape[1] - t)
        x[:, t0:t0 + t, :] = 0
        
        return x 