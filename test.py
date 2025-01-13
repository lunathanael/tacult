from src import evaluate, Agent
import numpy as np
import gymnasium as gym
import utac_gym

pth = "models/utac-v0__opp_self__1__1736703452.pth"

env = gym.make("utac-v0", render_mode="text")
# agent = Agent.load(pth, np.prod(env.observation_space.shape), env.action_space.n)

# pth = "models/utac-v0__opp_self__1__1736703452.pth"
# agent2 = Agent.load(pth, np.prod(env.observation_space.shape), env.action_space.n)


evaluation = evaluate.evaluate(utac_gym.wrappers.MCTSOpponent(), utac_gym.wrappers.MCTSOpponent(), num_episodes=1)
print(evaluation)