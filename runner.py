from src import UtacEnv, MCTS, Network, MCTSNN

def main():
    env = UtacEnv()
    env.reset()
    mcts = MCTS()
    print(mcts.choose_action(env, 1000))

    network = Network()
    mcts = MCTSNN(network)
    print(mcts.choose_action(env, 1000))

if __name__ == "__main__":
    main()

