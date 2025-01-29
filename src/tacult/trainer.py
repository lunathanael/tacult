import logging
import coloredlogs
import os

from tacult.utils import dotdict

_args = dotdict(
    {
        "numIters": 300,
        "minNumEps": 128,  # Minimum number of complete self-play games to simulate during a new iteration, an upper bound over this minimum is the number of environments.
        "numEnvs": 128,
        "tempThreshold": 10,  #
        "updateThreshold": 0.6,  # During arena playoff, new neural net will be accepted if threshold or more of games are won.
        "maxlenOfQueue": 1_000_000,  # Number of game examples to train the neural networks. Do (minNumEps + numEnvs) * 81 * 8
        "numMCTSSims": 800,  # Number of games moves for MCTS to simulate.
        "cpuct": 2.4,

        "arenaNumMCTSSims": 2,
        "arenaCompare": 256,  # Number of games to play during arena play to determine if new net will be accepted.
        "verbose": False,  # Whether to print verbose output for Arena.

        "shuffle_data": True,
        "steps_per_epoch": 10,
        "epochs": 10,
        "batch_size": 4096,
        "lr": 0.2,
        "num_warm_restarts": 3,
        "model_args": {
            "channels": 16,
            "num_residual_blocks": 3,
            "embedding_size": 256,
            "dropout": 0.3,
            "onnx_export": False,
            "cuda": False,
        },
        "saveAllModels": True,
        "saveTrainExamples": False,
        "load_checkpoint": True,
        "load_model": False,
        "cuda": False,

        "load_folder": "./temp/",
        "checkpoint_folder": "./temp/resnet2/",
    }
)

if not os.path.exists(_args.checkpoint_folder):
    os.makedirs(_args.checkpoint_folder)

logging.basicConfig(
    filename=f"{_args.checkpoint_folder}/trainer.log",
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)
coloredlogs.install(level="INFO")  # Change this to DEBUG to see more info.


from tacult.base.coach import Coach
from tacult.utac_game import UtacGame as Game

from tacult.utac_nn import UtacNN
from tacult.base.nn_wrapper import load_network


def main(args=_args):
    log.info("Loading %s...", Game.__name__)
    g = Game()

    log.info("Loading %s...", UtacNN.__name__)
    nnet = UtacNN(args)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder, "best.pt")
        nnet = load_network(args.load_folder, "best.pt")
    else:
        log.warning("Not loading a checkpoint!")
        log.info("Network architecture:")
        log.info(nnet.nnet)

    log.info("Loading the Coach...")
    c = Coach(g, nnet, args)

    if args.load_checkpoint:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info("Starting the learning process 🎉")
    c.learn()


def train(args=_args):
    # main(args)
    main(args)


if __name__ == "__main__":
    main()
