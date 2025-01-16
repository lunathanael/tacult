import numpy as np
import torch
import logging

log = logging.getLogger(__name__)

def to_obs(canonicalBoard):
    obs = np.array(canonicalBoard.get_obs())
    obs = obs.reshape(4, 9, 9)
    return torch.from_numpy(obs).float()


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
        log.warning("CUDA requested but not available! Using CPU instead.")
        cuda = False
    
    return torch.device("cuda" if cuda else "cpu")