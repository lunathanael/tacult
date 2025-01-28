import logging

import coloredlogs

from tacult.utac_game import UtacGame as Game
from tacult.utils import dotdict
from tacult.base.coach import Coach
from tacult.network import UtacNNet
from tacult.utac_nn import UtacNN
from tacult.base import NNetWrapper
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

    log.info('Loading %s...', UtacNNet.__name__)
    nnet = UtacNN(args)

    log.info('Loading checkpoint "%s/%s"...', args.model_file[0], args.model_file[1])
    nnet.load_checkpoint(args.model_file[0], args.model_file[1])

    game = Game()

    # First create the MCTS tournament as before
    num_sims_list = [2, 32, 64]
    pit = Pit.create_mcts_tournament(
        game=game,
        nnet=nnet,
        num_sims_list=num_sims_list,
        cpuct=1.0,
        games_per_match=32,
        num_rounds=100,
        temperature=0,
        verbose=3
    )

    # Create and add random player
    def random_player(board):
        valids = game.getValidMoves(board, 1)
        valids = np.array(valids)
        valid_moves = np.where(valids)[0]
        action = np.random.choice(valid_moves)
        return action

    pit.add_agent(random_player, "Random")

    rollouts_list = [1, 8, 64]
    for num_sims in num_sims_list:
        for num_rollouts in rollouts_list:
            _args = {
                'numMCTSSims': num_sims,
                'numRollouts': num_rollouts,
                'cpuct': args.cpuct,
            }

            _args = dotdict(_args)
            def create_raw_mcts_player(args):
                def raw_mcts_player(board):
                    raw_mcts = RawMCTS(game, args)
                    return np.argmax(raw_mcts.getActionProb(board, temp=0))
                return raw_mcts_player

            pit.add_agent(create_raw_mcts_player(_args), f"RawMCTS_sims{num_sims}_rollouts{num_rollouts}")

    pit.play_tournament()
    
def test(args=_args):
    main(args)


if __name__ == "__main__":
    main()