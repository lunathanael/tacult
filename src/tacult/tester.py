import logging
import os
import re


logging.basicConfig(filename='log.txt',
                    filemode='a',
                    format='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

import coloredlogs

from tacult.utac_game import UtacGame as Game
from tacult.utils import dotdict
from tacult.base.coach import Coach
from tacult.network import UtacNNet
from tacult.utac_nn import UtacNN
from tacult.base import NNetWrapper, load_network
from tacult.base.mcts import MCTS, RawMCTS, VectorizedMCTS
from tacult.base.arena import Arena
import numpy as np
import torch
from tacult.pit import Pit

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

_args = dotdict({
    'numMCTSSims': 300,          # Number of games moves for MCTS to simulate.
    'cpuct': 1,
    'cuda': False,
    'model_file': ('./temp/run3','best.pt'),
    'ignore_optimizer': True,
    'numRollouts': 50,
    'onnx_export': True,

    'lr': 0.001, # redundancy
    'steps_per_epoch': 10,
    'epochs': 10,
    'batch_size': 512,
    'numIters': 10,
    'num_games': 10,
    'verbose': False,

    'lr': 0.001,
    'dropout': 0.3,
    'cuda': False,
    'maxlenOfQueue': 200000,
    'numItersForTrainExamplesHistory': 20,
    'numEnvs': 1,
    'minNumEps': 1,
    'tempThreshold': 100,
})


def main(args=_args):
    log.info('Loading %s...', Game.__name__)


    game = Game()

    pit = Pit([], [], game, games_per_match=8, num_rounds=1000, verbose=3)

    # pit.load_checkpoints_to_pit(checkpoint_dir="./temp/resnet", num_sims_list=[2])
    pit.load_checkpoints_to_pit(checkpoint_dir="./temp/resnet3", num_sims_list=[2])
    # pit.load_checkpoints_to_pit(checkpoint_dir="./temp/run3", num_sims_list=[2])

    # Create and add random player
    class RandomPlayer(Pit.PitAgent):
        def __init__(self, game, args):
            self.game = game
            self.args = args

        def __call__(self, x):
            valids = game.getValidMoves(x, 1)
            valids = np.array(valids)
            valid_moves = np.where(valids)[0]
            action = np.random.choice(valid_moves)
            return action

    pit.add_agent(RandomPlayer(game, _args), "Random")

    
    class RawMCTSAgent(Pit.PitAgent):
        def __init__(self, game, args, temp):
            self.game = game
            self.args = args
            self.mcts_instance = None
            self.temp = temp

        def __call__(self, x):
            return np.argmax(self.mcts_instance.getActionProb(x, temp=self.temp))
        
        def reset(self):
            self.mcts_instance = RawMCTS(self.game, self.args)

    rollouts_list = [10]
    num_sims_list = [9, 81]
    for num_sims in num_sims_list:
        for num_rollouts in rollouts_list:
            marg = {
                'numMCTSSims': num_sims,
                'numRollouts': num_rollouts,
                'cpuct': args.cpuct,
            }
            marg = dotdict(marg)

            pit.add_agent(RawMCTSAgent(game, marg, 0), f"RawMCTS_sims{num_sims}_rollouts{num_rollouts}")

    
    pit.play_tournament()
    

def test(args=_args):
    main(args)


if __name__ == "__main__":
    main()