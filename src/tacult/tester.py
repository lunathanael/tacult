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
log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

_args = dotdict({
    'numMCTSSims': 100,          # Number of games moves for MCTS to simulate.
    'cpuct': 1,
    'cuda': False,
    'model_file': ('./temp/run1','best.pt'),
    'ignore_optimizer': True,
    'numRollouts': 50,

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
    g = Game()

    # mcts = RawMCTS(g, args)
    # act = mcts.getActionProb(g.getInitBoard(), temp=1)
    # print(act)
    # act = np.array(act).reshape(9, 9)
    # print(act.round(2))
    # print(mcts.getCurrentEvaluation(g.getInitBoard()))
    log.info('Loading %s...', UtacNNet.__name__)
    # nnet = NNetWrapper(UtacNNet(args.cuda, onnx_export=True), args)
    nnet = UtacNN(args)

    log.info('Loading checkpoint "%s/%s"...', args.model_file[0], args.model_file[1])
    nnet.load_checkpoint(args.model_file[0], args.model_file[1])

    board = g.getInitBoard()

    current_player = 1
    moves = [40]
    for action in moves:
        # moves = g.getValidMoves(board, current_player)
        # action = np.random.choice(np.where(moves)[0])
        board, current_player = g.getNextState(board, current_player, action)
        if g.getGameEnded(board, current_player) != 0:
            assert False
            board = g.getInitBoard()
            current_player = 1

    board.print()
    g.getCanonicalForm(board, current_player).print()
    print(current_player)
    # obs = g._get_obs(g.getCanonicalForm(board, current_player))
    # print(obs)
    # obs = torch.tensor(obs).reshape(1, 4, 9, 9).float()
    # pi, v = nnet.predict(obs)
    # print(pi.reshape(9, 9).round(2))
    # print(v)

    # print("Thinking...")
    # mcts = VectorizedMCTS(g, nnet, args)
    # pi = mcts.getActionProbs([g.getCanonicalForm(board, current_player)], temps=[1])[0]
    # print(pi.reshape(9, 9).round(2))
    return

    # pi = np.zeros(9*9)
    # pi[action] = 1

    # sym = g.getSymmetries(board, pi)
    # for b, p in sym:
    #     b.print()
    #     print(p.reshape(9, 9))
    
    # print(g._get_obs(board))
    # return
    # nnet = UtacNN(args)

    # coach = Coach(g, nnet, args)
    # q = coach.generateTrainExamples()
    # print([x[2] for x in q])
    
    # canonical_board = g.getCanonicalForm(board, 1)
    # obs = g._get_obs(canonical_board).unsqueeze(0)
    
    obs = [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0, -1,
        1,  0,  0,  1,  0,  0,  0, -1, -1,  0,  0,  0,  0,  0,  0,  1,  0,
        0,  0,  0,  0,  0,  0, -1,  0,  0,  1,  0,  0,  0,  0,  1,  1,  1,
       -1, -1,  0,  0,  0,  1,  0,  0,  1, -1, -1, -1, -1, -1, -1,  0, -1,
        1,  0, -1,  0,  0,  0,  1,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  0,  0,  0,  0,  0,
        0,  1,  1,  1,  0,  0,  0,  0,  0,  0,  1,  1,  1,  0,  0,  0,  1,
        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
        1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  1,  1,  1,  0,  0,  0,  0,  0,  0,  1,  1,  1,
        0,  0,  0,  0,  0,  0,  1,  1,  1,  0,  0,  0, -1, -1, -1,  1,  1,
        1, -1, -1, -1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1, -1,  1,
        1,  1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,----
        1,  0,  0,  0,  0,  0,  0,  1,  1,  1,  0,  0,  0,  0,  0,  0,  1,
        1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0]
    obs = torch.tensor(obs).reshape(1, 4, 9, 9).float()

    print(obs)
    # obs[:,0,:,:] *= -1
    # obs[:,2,:,:] *= -1
    # print(obs)

    pred = nnet.predict(obs)
    print(pred)
    pi, v = pred
    print(pi.reshape(1, 9, 9).round(2))

    mask = obs[:,-1,:,:].flatten()
    print(mask.reshape(1, 9, 9))

    pi = pi * mask.numpy()
    print(pi.reshape(1, 9, 9).round(2))

    sum_pi = np.sum(pi, keepdims=True)
    pi = pi / sum_pi
    print(pi.reshape(1, 9, 9).round(2))
    print(v)
    # mcts = MCTS(g, nnet, args)
    # raw_mcts = RawMCTS(g, args)

    # arena = Arena(lambda x: np.argmax(mcts.getActionProb(x, temp=0)), lambda x: np.argmax(raw_mcts.getActionProb(x, temp=0)), g)
    # a = arena.playGames(args.num_games, verbose=False)
    # print(a)

def test(args=_args):
    main(args)


if __name__ == "__main__":
    main()