from gymnasium.envs.registration import register
from .envs import UtacEnv

register(
    id="utacenv/UtacEnv-v0",
    entry_point="utacenv.envs:UtacEnv",
)
