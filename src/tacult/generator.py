import logging
import os
from tqdm import tqdm
import numpy as np
import torch
from tacult.utac_game import UtacGame as Game
from tacult.base.mcts import RawMCTS
from collections import deque
from tacult.utils import dotdict
from pickle import Unpickler, Pickler
import torch.nn.functional as F

log = logging.getLogger(__name__)

args = dotdict({
    'numMCTSSims': 100,
    'numRollouts': 9,
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
        # Pre-allocate tensors
        self.iterationTrainExamples = []

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
            # Convert pi to tensor
            pi_tensor = torch.from_numpy(pi).float()

            sym = self.game.getSymmetries(canonical_board, pi_tensor)
            for b, p in sym:
                train_examples.append([
                    self.game._get_obs(b),
                    current_player,
                    torch.from_numpy(p).float(),
                ])
            
            action_tensor = torch.multinomial(pi_tensor, 1)
            action = int(action_tensor.item()) 
            
            board, current_player = self.game.getNextState(board, current_player, action)
            

            r = self.game.getGameEnded(board, current_player)
            if r != 0:
                results = [
                    (
                        x[0], x[2], r * ((-1) ** (x[1] != current_player))
                    )
                    for x in train_examples
                ]
                return results
                

    def generate_games(self, num_games, output_path):
        """
        Generate a specified number of self-play games and save them.
        
        Args:
            num_games: Number of games to generate
            output_dir: Directory to save the generated games
        """
        
        for i in tqdm(range(num_games), desc="Generating games"):
            # Play one episode
            self.iterationTrainExamples.extend(self.execute_episode())
        
        log.info(f"Saving {len(self.iterationTrainExamples)} examples to {output_path}")
        self.saveTrainExamples(self.iterationTrainExamples, output_path)

    def saveTrainExamples(self, game_history, output_path):
        folder = os.path.dirname(output_path)
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(output_path, "wb+") as f:
            Pickler(f).dump(game_history)
        

def main():
    game = Game()
    generator = Generator(game, args)
    generator.generate_games(100, './temp/data.examples')

if __name__ == '__main__':
    main()
