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
from ..utils import Rating, Glicko2
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

        self.nnet.save_checkpoint(folder=args.checkpoint_folder, filename='best.pt')
        self.pnet = self.nnet.__class__(args)
        self.pnet.load_checkpoint(folder=args.checkpoint_folder, filename='best.pt')

        self.args = args
        self.trainExamplesHistory = deque(maxlen=args.maxlenOfQueue)  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

        # Print args in a more readable format
        args_str = '\n'.join(f'  {k}: {v}' for k, v in args.items())
        log.info(f'Coach initialized with arguments:\n{args_str}')

        self.glicko2 = Glicko2()

        self.nnet_elo = self.rate_with_random_player()
        self.pnet_elo = self.nnet_elo

    def rate_with_random_player(self, random_player_elo: Rating = None):
        numEnvs = self.args.numEnvs
        numMCTSSims = self.args.numMCTSSims
        self.args.numEnvs = min(self.args.numEnvs, self.args.arenaCompare // 2)
        self.args.numMCTSSims = self.args.arenaNumMCTSSims
        nmcts = MCTS(self.game, self.nnet, self.args)

        def random_player(states):
            valids = np.array([self.game.getValidMoves(x, 1) for x in states])
            valid_moves = [np.where(valid)[0] for valid in valids]
            actions = np.array([np.random.choice(moves) if len(moves) > 0 else -1 for moves in valid_moves])
            return actions

        log.info(f'Random player arena with {self.args.numEnvs} environments and {self.args.numMCTSSims} simulations')
        arena = Arena(
            random_player,
            lambda x: np.argmax(nmcts.getActionProbs(x, temps=np.zeros(self.args.numEnvs)), axis=1),
            self.game,
            lambda x: x.print(),
            self.args.numEnvs
        )
        rwins, nwins, draws = arena.playGames(self.args.arenaCompare, verbose=self.args.verbose)
        self.args.numEnvs = numEnvs
        self.args.numMCTSSims = numMCTSSims
        log.info(f'Random player arena wins: {nwins}, losses: {rwins}, draws: {draws}')

        if random_player_elo is None:
            random_player_elo = Rating(mu=750, phi=500, sigma=1)

        nnet_rating = self.glicko2.create_rating()
        series = [(1.0, random_player_elo)] * nwins + [(0.0, random_player_elo)] * rwins + [(0.5, random_player_elo)] * draws

        nnet_rating = self.glicko2.rate(nnet_rating, series)
        log.info(f'NNet rating: {nnet_rating.mu:.1f} +- {nnet_rating.phi:.1f}')
        return nnet_rating

    
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
        mcts = MCTS(self.game, self.pnet, self.args)  # reset search tree

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

            log.info(f"Training on {len(self.trainExamplesHistory)} examples")

            self.nnet.train(self.trainExamplesHistory)

            self.nnet.save_checkpoint(folder=self.args.checkpoint_folder, filename='temp.pt')

            numEnvs = self.args.numEnvs
            numMCTSSims = self.args.numMCTSSims
            self.args.numEnvs = min(self.args.numEnvs, self.args.arenaCompare // 2)
            self.args.numMCTSSims = self.args.arenaNumMCTSSims
            pmcts = MCTS(self.game, self.pnet, self.args)
            nmcts = MCTS(self.game, self.nnet, self.args)

            log.info(f'Previous player arena with {self.args.numEnvs} environments and {self.args.numMCTSSims} simulations')
            arena = Arena(
                lambda x: np.argmax(pmcts.getActionProbs(x, temps=np.zeros(self.args.numEnvs)), axis=1),
                lambda x: np.argmax(nmcts.getActionProbs(x, temps=np.zeros(self.args.numEnvs)), axis=1),
                self.game,
                lambda x: x.print(),
                self.args.numEnvs
            )
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare, verbose=self.args.verbose)
            self.args.numEnvs = numEnvs
            self.args.numMCTSSims = numMCTSSims

            series = [(1.0, self.pnet_elo)] * nwins + [(0.0, self.pnet_elo)] * pwins + [(0.5, self.pnet_elo)] * draws
            self.nnet_elo = self.glicko2.rate(self.nnet_elo, series)
            log.info(f'NEW/PREV WINS : {nwins} / {pwins} ; DRAWS : {draws}')
            log.info(f'Estimated NNet rating: {self.nnet_elo.mu:.1f} +- {self.nnet_elo.phi:.1f}')
            log.info(f"Change in rating: {self.nnet_elo.mu - self.pnet_elo.mu:.1f}")
            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:
                log.info('REJECTING NEW MODEL')
                # self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pt')
            else:
                log.info('ACCEPTING NEW MODEL')
                if self.args.saveAllModels:
                    log.info(f'Saving new best model {i} to {self.args.checkpoint_folder} as {self.getCheckpointFile(i)}')
                    self.nnet.save_checkpoint(folder=self.args.checkpoint_folder, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint_folder, filename='best.pt')
                self.saveTrainExamples(directory=self.args.checkpoint_folder, filename='best.pt.examples')
                self.pnet.load_checkpoint(folder=self.args.checkpoint_folder, filename='best.pt')

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pt'

    def deleteTrainExamples(self, directory: str, filename: str):
        folder = directory
        filename = os.path.join(folder, filename)
        if os.path.exists(filename):
            os.remove(filename)

    def saveTrainExamples(self, directory: str, filename: str):
        folder = directory
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, filename)
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder, "best.pt")
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = deque(Unpickler(f).load(), maxlen=self.args.maxlenOfQueue)
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True
