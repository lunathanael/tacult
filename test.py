from src import evaluate, Agent
import numpy as np
import gymnasium as gym
import utac

pth = "models/utac-v0__trainer__1__1736677780.pth"

env = gym.make("utac-v0")
agent = Agent.load(pth, np.prod(env.observation_space.shape), env.action_space.n)


evaluation = evaluate.evaluate_random(agent, num_episodes=10000)
print(evaluation)