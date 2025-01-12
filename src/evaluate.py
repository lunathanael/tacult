from .agent import Agent
import torch
import gymnasium as gym
import utac
from utac.wrappers import PlayOpponentWrapper, RandomOpponent

def evaluate(agent_1, agent_2, num_episodes=100) -> tuple[float, float]:
    env = gym.make("utac-v0")
    env = PlayOpponentWrapper(env, agent_2)

    total_score = 0
    total_length = 0
    for _ in range(num_episodes):
        obs, info = env.reset()
        agent_1_player = info["current_player"]
        done = False
        while not done:
            with torch.no_grad():
                obs = torch.unsqueeze(torch.tensor(obs), 0)
                obs = obs.to(torch.float32)
                action = agent_1.get_action(obs, info=info)
            obs, reward, done, truncated, info = env.step(action)
            total_length += 1
        if info["winner"] == agent_1_player:
            total_score += 1
        elif info["winner"] == "Draw":
            total_score += 0
        else:
            total_score -= 1
    return total_score / num_episodes, total_length / num_episodes


def evaluate_random(
    agent: Agent, num_episodes: int, allow_illegal: bool = False
) -> tuple[float, float]:
    return evaluate(agent, RandomOpponent(), num_episodes)