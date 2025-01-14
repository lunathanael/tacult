import numpy as np


def to_obs(canonicalBoard):
    obs = np.array(canonicalBoard.get_obs())
    obs = obs.reshape(2, 9, 9)
    return obs