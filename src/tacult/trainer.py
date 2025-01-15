import logging

import coloredlogs

from tacult.base.coach import Coach
from tacult.utac_game import UtacGame as Game
from tacult.utils import dotdict

from tacult.utac_nn import UtacNN

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

_args = dotdict({
    'numIters': 1000,
    'minNumEps': 4,              # Minimum number of complete self-play games to simulate during a new iteration, an upper bound over this minimum is the number of environments.
    'numEnvs': 4,
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 331776,    # Number of game examples to train the neural networks. Do (minNumEps + numEnvs) * 81 * 8
    'numMCTSSims': 4,          # Number of games moves for MCTS to simulate.
    'cpuct': 1,

    'arenaCompare': 20,         # Number of games to play during arena play to determine if new net will be accepted.
    'verbose': False,            # Whether to print verbose output for Arena.

    'saveAllModels': True,
    'saveTrainExamples': False,

    'shuffle_data': True,
    'steps_per_epoch': 10,   
    'epochs': 10,
    'batch_size': 4,

    'lr': 0.001,
    'dropout': 0.3,
    'cuda': True,

    'load_checkpoint': False,
    'load_model': False,
    'checkpoint': './temp/run_1',
    'load_folder_file': ('./temp','copy'),
    'numItersForTrainExamplesHistory': 20,
})


def main(args=_args):
    log.info('Loading %s...', Game.__name__)
    g = Game()

    
    log.info('Loading %s...', UtacNN.__name__)
    nnet = UtacNN(args)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_checkpoint:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()

def train(args=_args):
    # main(args)
    main(args)


if __name__ == "__main__":
    main()