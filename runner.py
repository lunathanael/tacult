from src import UtacEnv, MCTS, Network, MCTSNN
import torch
import numpy as np

def main():
    env = UtacEnv()
    env.reset()
    #env.step((4, 1, 1))

    network = Network()
    network.load_state_dict(torch.load('model_checkpoint_3900.pt'))
    network.eval()
    
    mcts = MCTSNN(network, selection_method="argmax")
    print(mcts.choose_action(env, 1000, temperature=0))
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

