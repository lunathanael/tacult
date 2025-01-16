import logging

import coloredlogs

from tacult.utac_game import UtacGame as Game
from tacult.utils import dotdict

from tacult.network import UtacNNet
from tacult.base import NNetWrapper
from tacult.base.mcts import MCTS, RawMCTS
from tacult.base.arena import Arena
import numpy as np

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

_args = dotdict({
    'numMCTSSims': 40,          # Number of games moves for MCTS to simulate.
    'cpuct': 1,
    'cuda': False,
    'model_file': ('./temp/rerun_1','best.pt'),
    'ignore_optimizer': True,
    'numRollouts': 5,

    'lr': 0.001, # redundancy
    'steps_per_epoch': 10,
    'epochs': 10,
    'batch_size': 512,
    'numIters': 10,
    'num_games': 10,
    'verbose': False,
})


def main(args=_args):
    log.info('Loading %s...', Game.__name__)
    g = Game()

    # mcts = RawMCTS(g, args)
    # act = mcts.getActionProb(g.getInitBoard(), temp=1)
    # print(act)
    # act = np.array(act).reshape(9, 9)
    # print(act.round(2))
    # print(mcts.getCurrentEvaluation(g.getInitBoard()))
    log.info('Loading %s...', UtacNNet.__name__)
    nnet = NNetWrapper(UtacNNet(args.cuda, onnx_export=True), args)

    log.info('Loading checkpoint "%s/%s"...', args.model_file[0], args.model_file[1])
    nnet.load_checkpoint(args.model_file[0], args.model_file[1])

    board = g.getInitBoard()
    mcts = MCTS(g, nnet, args)
    raw_mcts = RawMCTS(g, args)

    arena = Arena(lambda x: np.argmax(mcts.getActionProb(x, temp=0)), lambda x: np.argmax(raw_mcts.getActionProb(x, temp=0)), g)
    a = arena.playGames(args.num_games, verbose=False)
    print(a)
def test(args=_args):
    main(args)


if __name__ == "__main__":
    main()