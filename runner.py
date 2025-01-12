from src import UtacEnv, MCTS, Network, MCTSNN
import torch
import numpy as np

from train import Trainer

def main():
    env = UtacEnv()
    env.reset()
    while True:
        nenv = env.clone()
        action = nenv.get_legal_actions()[0]
        nenv.step(action)
        if nenv.is_terminal():
            nenv.state.print()
            print(f"Action: {action}")
            break
        env = nenv

    # Load trainer from saved checkpoint
    trainer = Trainer.load_trainer("model_2fc3705_238.50000.pkl")
    network = trainer.network
    network.eval()
    
    mcts = MCTSNN(network, selection_method="argmax")
    print(mcts.choose_action(env, 1000))
    print(mcts.get_root_evaluation())
    eval = mcts.get_root_evaluation()

    prob_matrix = np.zeros((9, 9))

    total_visits = sum(action_eval["visits"] for action_eval in eval[1])
    
    for i in range(9):
        for j in range(9):
            action_idx = i*9 + j
            found = False
            for action_eval in eval[1]:
                if action_eval["action"] == action_idx:
                    prob_matrix[i, j] = action_eval["visits"] / total_visits
                    found = True
                    break
            if not found:
                prob_matrix[i, j] = 0.0

    print("\nAction probabilities from MCTS visits:")
    for i in range(9):
        for j in range(9):
            print(f"{prob_matrix[i, j]:.3f}", end=" ")
        print()

if __name__ == "__main__":
    main()
