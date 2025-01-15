from . import network, utac_game, utac_nn, utils, base, trainer, export_model
from .base import coach, mcts, nn_wrapper, utils, arena
from .trainer import train
from .export_model import export_model
from .utac_game import UtacGame


try:
    import sys
    assert "utac_gym" in globals().keys()
    del sys
except AssertionError:
    import warnings
    warnings.warn("utac_gym module is required but not found", ImportWarning)
    del warnings


__all__ = [
    "network",
    "utac_game",
    "utac_nn",
    "utils",
    "base",
    "trainer",
    "export_model",
    "coach",
    "mcts",
    "nn_wrapper",
    "arena",
    "train",
    "UtacGame",
]
