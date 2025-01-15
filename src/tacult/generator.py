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
import torch.nn.functional as F
import torch._dynamo

torch._dynamo.config.capture_scalar_outputs = True

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
        # Pre-allocate tensors
        self.pi_tensor = torch.zeros(game.getActionSize(), dtype=torch.float32)
        self.action_tensor = torch.zeros(1, dtype=torch.long)

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
            # Convert pi to tensor
            self.pi_tensor.copy_(torch.tensor(pi))
            
            # Use torch.multinomial instead of np.random.choice
            self.action_tensor = torch.multinomial(self.pi_tensor, 1)
            action = int(self.action_tensor.item())  # Convert to int outside the critical path
            
            board, current_player = self.game.getNextState(board, current_player, action)
            
            sym = self.game.getSymmetries(canonical_board, pi)
            for b, p in sym:
                train_examples.append([
                    torch.tensor(self.game._get_obs(b), dtype=torch.float32),
                    torch.tensor(current_player, dtype=torch.int32),
                    torch.tensor(p, dtype=torch.float32),
                ])

            r = self.game.getGameEnded(board, current_player)
            if r != 0:
                # Vectorize the result calculation
                results = torch.full((len(train_examples),), r, dtype=torch.float32)
                player_diff = torch.tensor([x[1].item() != current_player for x in train_examples], dtype=torch.float32)
                results *= torch.pow(-1, player_diff)
                
                return [(x[0], x[1], x[2], r) for x, r in zip(train_examples, results)]

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