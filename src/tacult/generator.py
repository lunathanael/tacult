import logging
import os
from tqdm import tqdm
import numpy as np
import torch
from tacult.utac_game import UtacGame as Game
from tacult.base.mcts import RawMCTS
from collections import deque
from tacult.utils import dotdict
from pickle import Pickler
log = logging.getLogger(__name__)

args = dotdict({
    'numMCTSSims': 81,
    'numRollouts': 5,
    'cpuct': 1.0,
    'maxlenOfQueue': 331776,
})

class Generator():
    def __init__(self, game: Game, args: dotdict) -> None:
        """
        Initialize the generator with game rules and parameters.
        
        Args:
            game: Game implementation
            args: Arguments including numMCTSSims, numRollouts, cpuct, etc.
        """
        self.game = game
        self.args = args
        self.mcts = RawMCTS(game, args)

    @torch.compile
    def execute_episode(self):
        """
        Execute one episode of self-play using MCTS.
        
        Returns:
            list: [(board, current_player, pi, result)], game history
        """
        train_examples = []
        board = self.game.getInitBoard()
        current_player = 1
        episode_step = 0

        while True:
            episode_step += 1
            canonical_board = self.game.getCanonicalForm(board, current_player)
            
            pi = self.mcts.getActionProb(canonical_board, temp=1)
            
            sym = self.game.getSymmetries(canonical_board, pi)
            for b, p in sym:
                train_examples.append([
                    self.game._get_obs(b),
                    current_player,
                    p,
                ])

            action = np.random.choice(len(pi), p=pi)
            board, current_player = self.game.getNextState(board, current_player, action)
            
            r = self.game.getGameEnded(board, current_player)
            if r != 0:
                # Update all examples with the game result
                return [(x[0], x[1], x[2], r * ((-1) ** (x[1] != current_player))) 
                        for x in train_examples]

    @torch.compile
    def generate_games(self, num_games, output_path):
        """
        Generate a specified number of self-play games and save them.
        
        Args:
            num_games: Number of games to generate
            output_dir: Directory to save the generated games
        """
        iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)
        
        for i in tqdm(range(num_games), desc="Generating games"):
            # Play one episode
            iterationTrainExamples += self.execute_episode()
        
        log.info(f"Saving {len(iterationTrainExamples)} examples to {output_path}")
        self.saveTrainExamples(iterationTrainExamples, output_path)

    @torch.compiler.disable(recursive=True)
    def saveTrainExamples(self, game_history, output_path):
        folder = os.path.dirname(output_path)
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(output_path, "wb+") as f:
            Pickler(f).dump([game_history])
        
        

def main():
    game = Game()
    generator = Generator(game, args)
    generator.generate_games(200, './data.examples')

if __name__ == '__main__':
    main()