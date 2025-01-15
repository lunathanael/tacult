import numpy as np
import torch
import warnings


def to_obs(canonicalBoard):
    obs = np.array(canonicalBoard.get_obs())
    obs = obs.reshape(2, 9, 9)
    return obs


def get_device(cuda: bool = None) -> torch.device:
    """Get the appropriate device (CUDA or CPU) based on availability and preference.
    
    Args:
        cuda: If True, forces CUDA if available. If False, forces CPU.
               If None, uses CUDA if available, otherwise CPU.
    
    Returns:
        torch.device: The selected device
    """
    if cuda is None:
        cuda = torch.cuda.is_available()
    elif cuda and not torch.cuda.is_available():
        warnings.warn("CUDA requested but not available! Using CPU instead.")
        cuda = False
    
    return torch.device("cuda" if cuda else "cpu")