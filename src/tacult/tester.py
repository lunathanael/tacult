import logging

import coloredlogs

from tacult.utac_game import UtacGame as Game
from tacult.utils import dotdict

from tacult.network import UtacNNet
from tacult.base import NNetWrapper
from tacult.base import MCTS
from tacult.base.raw_mcts import RawMCTS
import numpy as np

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

_args = dotdict({
    'numMCTSSims': 400,          # Number of games moves for MCTS to simulate.
    'cpuct': 1,
    'cuda': False,
    'model_file': ('./temp/','best.pth.tar'),
    'ignore_optimizer': True,
    'numRollouts': 5,

    'lr': 0.001, # redundancy
})


def main(args=_args):
    log.info('Loading %s...', Game.__name__)
    g = Game()

    mcts = RawMCTS(g, args)
    act = mcts.getActionProb(g.getInitBoard(), temp=1)
    print(act)
    act = np.array(act).reshape(9, 9)
    print(act.round(2))
    print(mcts.getCurrentEvaluation(g.getInitBoard()))
    # log.info('Loading %s...', UtacNNet.__name__)
    # nnet = NNetWrapper(UtacNNet(args.cuda, onnx_export=True), args)

    # log.info('Loading checkpoint "%s/%s"...', args.model_file[0], args.model_file[1])
    # nnet.load_checkpoint(args.model_file[0], args.model_file[1])

    # board = g.getInitBoard()
    # mcts = MCTS(g, nnet, args)
    # ret = mcts.getActionProb(board, temp=0)
    # obs = g._get_obs(g.getCanonicalForm(board, 1))
    # obs = np.array([[

    # ],
    # ])
    # obs = np.expand_dims(obs, axis=0)
    # pred = nnet.predict(obs)
    # a, b = pred
    # print(a, b)
    # print(np.round(a.reshape(9, 9), decimals=2))
    # print(np.array(ret).reshape(9, 9))


def test(args=_args):
    main(args)


if __name__ == "__main__":
    main()