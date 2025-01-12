from .agent import Agent
import torch
import gymnasium as gym
import utac
from utac.wrappers import PlayOpponentWrapper, RandomOpponent


def evaluate_random(
    agent: Agent, num_episodes: int, allow_illegal: bool = False
) -> tuple[float, float]:
    total_score = 0
    total_length = 0
    env = gym.make("utac-v0")
    env = PlayOpponentWrapper(env, RandomOpponent())
    for _ in range(num_episodes):
        obs, info = env.reset()
        agent_player = info["current_player"]
        random_player = "O" if agent_player == "X" else "X"
        done = False
        while not done:
            with torch.no_grad():
                obs = torch.unsqueeze(torch.tensor(obs), 0)
                obs = obs.to(torch.float32)
                action = agent.get_action(obs)
            obs, reward, done, truncated, info = env.step(action)
            total_length += 1
        if info["winner"] == agent_player:
            total_score += 1
        elif info["winner"] == random_player:
            total_score -= 1
    return total_score / num_episodes, total_length / num_episodes
