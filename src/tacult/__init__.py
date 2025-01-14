from . import network, utac_game, utac_nn, utils, base, trainer, export_model
from .base import coach, mcts, nn_wrapper, utils, arena
from .trainer import train
from .export_model import export_model


try:
    import sys
    assert "utac_gym" in globals().keys()
    del sys
except AssertionError:
    raise ImportError("utac_gym module is required but not found")


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
]
