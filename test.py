from src import evaluate, Agent
import numpy as np
import gymnasium as gym
import utac

pth = "models/utac-v0__opp_model_1__1__1736704754.pth"

env = gym.make("utac-v0")
agent = Agent.load(pth, np.prod(env.observation_space.shape), env.action_space.n)

pth = "models/utac-v0__opp_self__1__1736703452.pth"
agent2 = Agent.load(pth, np.prod(env.observation_space.shape), env.action_space.n)


evaluation = evaluate.evaluate(agent, agent2, num_episodes=100)
print(evaluation)