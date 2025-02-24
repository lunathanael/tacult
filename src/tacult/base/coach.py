import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle
import time
import torch
import numpy as np
from tqdm import tqdm

from .arena import VectorizedArena as Arena
from .arena import Arena as SingleArena
from .mcts import VectorizedMCTS as MCTS
from .mcts import MCTS as SingleMCTS

from utac_gym.core import GameState
from utac_gym.core.types import GAMESTATE

log = logging.getLogger(__name__)


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(args)  # the competitor network
        self.args = args
        self.trainExamplesHistory = deque(maxlen=args.maxlenOfQueue)  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.trainExamplesSizes = deque(maxlen=args.numItersForTrainExamplesHistory + 1)
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

    
    def prepExecuteEpisode(self):
        self._trainExamples = [[] for _ in range(self.args.numEnvs)]
        self._board = [self.game.getInitBoard() for _ in range(self.args.numEnvs)]
        self._curPlayer = np.ones(self.args.numEnvs, dtype=int)
        self._episodeStep = np.zeros(self.args.numEnvs, dtype=int)

        self._autoresetEnvs = np.zeros(self.args.numEnvs, dtype=bool)

      
    def executeEpisode(self, mcts: MCTS):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        results = []

        for i in range(self.args.numEnvs):
            if self._autoresetEnvs[i]:
                # self._trainExamples[i] = []
                # self._board[i] = self.game.getInitBoard()
                # self._curPlayer[i] = 1
                # self._episodeStep[i] = 0
                # self._autoresetEnvs[i] = False
                # mcts.reset(i)
                pass

        self._episodeStep += 1
        temps = np.less(self._episodeStep, self.args.tempThreshold)
        canonicalBoards = [
            self.game.getCanonicalForm(self._board[i], self._curPlayer[i])
            for i in range(self.args.numEnvs)
        ]

        pis = mcts.getActionProbs(canonicalBoards, temps=temps)
        for i in range(self.args.numEnvs):
            if self._autoresetEnvs[i]:
                continue
            pi = pis[i]
            sym = self.game.getSymmetries(canonicalBoards[i], pi)
            for b, p in sym:
                self._trainExamples[i].append([self.game._get_obs(b), self._curPlayer[i], torch.from_numpy(p).float()])

            action = np.random.choice(len(pi), p=pi)
            self._board[i], self._curPlayer[i] = self.game.getNextState(self._board[i], int(self._curPlayer[i]), action)

            r = self.game.getGameEnded(self._board[i], self._curPlayer[i])

            if r != 0:
                results.append(
                    [
                        (
                            x[0], x[2], r * ((-1) ** (x[1] != self._curPlayer[i]))
                        )
                        for x in self._trainExamples[i]
                    ]
                )
                self._autoresetEnvs[i] = True
        
        return results

    
    def generateTrainExamples(self):
        iterationTrainExamples = []
        mcts = MCTS(self.game, self.nnet, self.args)  # reset search tree

        games_played = 0
        time_started = time.time()
        print(tqdm.format_meter(
            n=games_played,
            total=self.args.minNumEps,
            elapsed=time.time() - time_started,
            unit="episode",
            prefix="Self Play: "
        ), end="\r")

        self.prepExecuteEpisode()
        while games_played < self.args.minNumEps:
            new_examples = self.executeEpisode(mcts)
            iterationTrainExamples.extend([
                example
                for episode_examples in new_examples
                for example in episode_examples
            ])
            games_played += len(new_examples)
            elapsed = time.time() - time_started
            print(tqdm.format_meter(
                n=games_played,
                total=self.args.minNumEps,
                elapsed=elapsed,
                unit="episode",
                prefix="Self Play: "
            ), end="\r")
        print()

        return iterationTrainExamples

    
    def learn(self):
        """
        Performs numIters iterations with at least minNumEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.numIters + 1):
            # bookkeeping
            log.info(f'Starting Iter #{i} ...')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = self.generateTrainExamples()

                # save the iteration examples to the history 
                self.trainExamplesHistory.extend(iterationTrainExamples)
                self.trainExamplesSizes.append(len(iterationTrainExamples))

            if len(self.trainExamplesSizes) >= self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the old entries in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                for _ in range(self.trainExamplesSizes.popleft()):
                    self.trainExamplesHistory.popleft()

            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)
            if self.args.saveTrainExamples:
                self.saveTrainExamples(i - 1)
                self.deleteTrainExamples(i - 2)
            # shuffle examples before training

            log.info(f"Training on {len(self.trainExamplesHistory)} examples")

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pt')

            self.nnet.train(self.trainExamplesHistory)

            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pt')

            pmcts = SingleMCTS(self.game, self.pnet, self.args)
            nmcts = SingleMCTS(self.game, self.nnet, self.args)

            log.info('PITTING AGAINST PREVIOUS VERSION')
            arena = SingleArena(
                lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                lambda x: np.argmax(nmcts.getActionProb(x, temp=0)),
                self.game,
                lambda x: x.print(),
            )
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare, verbose=self.args.verbose)

            log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:
                log.info('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pt')
            else:
                log.info('ACCEPTING NEW MODEL')
                if self.args.saveAllModels:
                    self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pt')

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def deleteTrainExamples(self, iteration):
        folder = self.args.checkpoint
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        if os.path.exists(filename):
            os.remove(filename)

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True