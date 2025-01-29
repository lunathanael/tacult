import logging



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

    log.info('Loading %s...', UtacNNet.__name__)
    nnet = UtacNN(args)

    log.info('Loading checkpoint "%s/%s"...', args.model_file[0], args.model_file[1])
    nnet.load_checkpoint(args.model_file[0], args.model_file[1])

    game = Game()

    # First create the MCTS tournament as before
    num_sims_list = [2, 64]
    pit = Pit.create_mcts_tournament(
        game=game,
        nnet=nnet,
        prefix="r3",
        num_sims_list=num_sims_list,
        cpuct=1.0,
        games_per_match=8,
        num_rounds=1000,
        temperature=0,
        verbose=3
    )

    onet = load_network("./temp/resnet", "best.pt")

    opit = Pit.create_mcts_tournament(
        game=game,
        nnet=onet,
        prefix="resnet",
        num_sims_list=num_sims_list,
        cpuct=1.0,
        games_per_match=8,
        num_rounds=1000,
        temperature=0,
        verbose=3
    )

    pit += opit

    # Create and add random player
    def random_player(board):
        valids = game.getValidMoves(board, 1)
        valids = np.array(valids)
        valid_moves = np.where(valids)[0]
        action = np.random.choice(valid_moves)
        return action

    pit.add_agent(random_player, "Random")

    
    pit.play_tournament()
    
def test(args=_args):
    main(args)


if __name__ == "__main__":
    main()