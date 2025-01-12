from gymnasium.envs.registration import register
from .envs import UtacEnv

register(
    id="UtacEnv-v0",
    entry_point="utacenv.envs.utacenv:UtacEnv",
)

__all__ = ["UtacEnv"]
