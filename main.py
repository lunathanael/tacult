import logging

import coloredlogs

from src.base.coach import Coach
from src.utac_game import UtacGame as Game
from src.utils import dotdict

from src.utac_nn import UtacNN

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 1000,
    'minNumEps': 128,              # Minimum number of complete self-play games to simulate during a new iteration, an upper bound over this minimum is the number of environments.
    'numEnvs': 128,
    'tempThreshold': 9,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 165888,    # Number of game examples to train the neural networks. Do (minNumEps + numEnvs) * 81 * 8
    'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
    'cpuct': 1,

    'arenaCompare': 256,         # Number of games to play during arena play to determine if new net will be accepted.
    'verbose': False,            # Whether to print verbose output for Arena.

    'shuffle_data': False,
    'validation_split': 0.1,
    'steps_per_epoch': 5,   
    'epochs': 10,
    'batch_size': 1024,

    'lr': 0.001,
    'dropout': 0.3,
    'cuda': False,

    'checkpoint': './temp/',
    'load_model': True,
    'load_folder_file': ('./temp/','checkpoint_2.pth.tar'),
    'numItersForTrainExamplesHistory': 10,

})


def main():
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

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()