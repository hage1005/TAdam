import torch
import numpy as np
import random

def set_seed(seed: int):
    """
    Set the seed for generating random numbers for reproducibility.
    
    Args:
        seed (int): The seed value to set for reproducibility.
    """
    random.seed(seed)  # Python's built-in random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU
    torch.cuda.manual_seed_all(seed)  # If using multiple GPUs
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable if true, as it can affect reproducibility

    print(f"Seed set to {seed}")
