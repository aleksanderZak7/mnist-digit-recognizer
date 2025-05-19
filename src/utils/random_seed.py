import torch
import random
import numpy as np

def set_random_seed(seed: int = 42) -> None:
    """
    Sets the random seed for Python's random, NumPy and PyTorch libraries to ensure reproducibility.

    :param seed: The integer value to use as the seed. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)