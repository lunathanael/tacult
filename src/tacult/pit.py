from __future__ import annotations

import logging
from typing import List, Callable, Dict, Tuple
import numpy as np
from collections import defaultdict
from .base.arena import Arena
from .base.mcts import MCTS
from .base.nn_wrapper import load_network
from .utils import dotdict, Rating, Glicko2
import os
import re


log = logging.getLogger(__name__)




def load_checkpoints(checkpoint_dir) -> list:
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_') and f.endswith('.pt')]
    checkpoint_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

    nets = []
    for file in checkpoint_files:
        nets.append({
            "name": int(file.split('_')[1].split('.')[0]),
            "net": load_network(checkpoint_dir, file)
        })
    return nets



class Pit:
    """
    A class to manage tournament-style evaluation between multiple agents,
    tracking statistics and calculating ELO ratings.
    """

    
    class PitAgent:
        def __init__(self):
            pass

        def __call__(self, x):
            pass
        
        def reset(self):
            pass

    class PitAgentMCTS(PitAgent):
        def __init__(self, game, nnet, temp, args):
            self.game = game
            self.nnet = nnet
            self.temp = temp
            self.args = args
            self.mcts_instance = None
            
        def __call__(self, x):
            return np.argmax(self.mcts_instance.getActionProb(x, temp=self.temp))
        
        def reset(self):
            self.mcts_instance = MCTS(self.game, self.nnet, self.args)
    
    def __init__(
        self,
        agents: List[Callable],
        agent_names: List[str],
        game,
        games_per_match: int = 100,
        num_rounds: int = 1,
        display=None,
        mu: float = 1500,
        phi: float = 350,
        sigma: float = 0.06,
        tau: float = 0.5,
        verbose: int = 1
    ):
        """
        Initialize the Pit with Glicko-2 rating system.
        
        Args:
            agents: List of agent functions that take board state and return actions
            agent_names: Names for the agents (for tracking purposes)
            game: Game object that defines the rules and mechanics
            games_per_match: Number of games to play between each pair of agents
            num_rounds: Number of times to repeat the full tournament
            display: Optional display function for the game state
            mu: Starting rating for all agents (default: 1500)
            phi: Starting rating deviation (default: 350)
            sigma: Starting volatility (default: 0.06)
            tau: System constant for volatility update (default: 0.5)
            verbose: Logging verbosity level (0=none, 1=round stats, 2=match stats)
        """
        if len(agents) != len(agent_names):
            raise ValueError("Number of agents must match number of agent names")
        
        self.agents: list[Pit.PitAgent] = agents
        self.agent_names = agent_names
        self.game = game
        self.games_per_match = games_per_match
        self.num_rounds = num_rounds
        self.display = display
        self.verbose = verbose
        self.log_stats = verbose > 0
        
        # Initialize Glicko2 system
        self.glicko2 = Glicko2(mu=mu, phi=phi, sigma=sigma, tau=tau)
        
        # Initialize statistics tracking
        self.stats = {
            'wins': defaultdict(int),
            'losses': defaultdict(int),
            'draws': defaultdict(int),
            'total_games': defaultdict(int),
            'ratings': {name: self.glicko2.create_rating() for name in agent_names},
            'win_rates': defaultdict(float),
            'loss_rates': defaultdict(float),
            'draw_rates': defaultdict(float)
        }
        
        self.match_history = []

    def _g(self, rd: float) -> float:
        """Glicko-2 g-function."""
        return 1 / np.sqrt(1 + (3 * rd**2) / (np.pi**2))

    def _E(self, rating: float, opp_rating: float, opp_rd: float) -> float:
        """Glicko-2 E-function (expected score)."""
        return 1 / (1 + np.exp(-self._g(opp_rd) * (rating - opp_rating) / 400))

    def _update_glicko2(self, player: str, opponents: List[str], scores: List[float]):
        """
        Update Glicko-2 ratings after a series of games.
        scores should be 1 for win, 0 for loss, 0.5 for draw
        """
        series = [(score, self.stats['ratings'][opp]) for score, opp in zip(scores, opponents)]
        self.stats['ratings'][player] = self.glicko2.rate(self.stats['ratings'][player], series)

    def _play_match(self, agent1_idx: int, agent2_idx: int) -> Tuple[int, int, int]:
        """Play a match between two agents and return results."""
        agent1 = self.agents[agent1_idx]
        agent2 = self.agents[agent2_idx]

        agent1.reset()
        agent2.reset()
        
        arena = Arena(agent1, agent2, self.game, self.display)
        oneWon, twoWon, draws = arena.playGames(self.games_per_match)
        
        return oneWon, twoWon, draws

    def _print_round_stats(self, round_num: int):
        """Print statistics after a round."""
        if not self.log_stats:
            return
            
        log.info(f"\nRound {round_num + 1} Statistics:")
        log.info("-" * 50)
        
        # Print ELO ratings
        log.info("\nELO Ratings:")
        sorted_elo = sorted(self.stats['ratings'].items(), key=lambda x: x[1].mu, reverse=True)
        for name, rating in sorted_elo:
            games_played = self.stats['total_games'][name]
            log.info(f"{name}: {rating.mu:.1f} ({games_played} games played)")
        
        # Print win rates
        log.info("\nWin Rates:")
        for name in self.agent_names:
            total_games = self.stats['total_games'][name]
            win_rate = self.stats['win_rates'][name] * 100
            log.info(f"{name}: {win_rate:.1f}% " +
                    f"(W: {self.stats['wins'][name]}, " +
                    f"L: {self.stats['losses'][name]}, " +
                    f"D: {self.stats['draws'][name]}) " +
                    f"({total_games} games)")
        log.info("-" * 50 + "\n")

    def _print_comprehensive_stats(self, title: str = "Current Statistics"):
        """Print comprehensive statistics in a tabular format."""
        if not self.log_stats:
            return
            
        log.info(f"\n{title}")
        log.info("-" * 100)
        
        # Sort agents by ELO rating
        sorted_agents = sorted(
            self.agent_names,
            key=lambda x: self.stats['ratings'][x].mu,
            reverse=True
        )
        
        # Print header
        header = f"{'Agent':<30} {'Rating':<10} {'Games':<8} {'Wins':<6} {'Losses':<6} {'Draws':<6} {'Win Rate':<8} {'Loss Rate':<8} {'Draw Rate':<8}"
        log.info(header)
        log.info("-" * 100)
        
        # Print each agent's stats
        for name in sorted_agents:
            name = name[:30]
            rating = self.stats['ratings'][name]
            total_games = self.stats['total_games'][name]
            wins = self.stats['wins'][name]
            losses = self.stats['losses'][name]
            draws = self.stats['draws'][name]
            win_rate = self.stats['win_rates'][name] * 100
            loss_rate = self.stats['loss_rates'][name] * 100
            draw_rate = self.stats['draw_rates'][name] * 100
            
            row = (
                f"{name:<30} "
                f"{rating.mu:>9.1f} "
                f"{total_games:>7} "
                f"{wins:>5} "
                f"{losses:>6} "
                f"{draws:>6} "
                f"{win_rate:>7.1f}% "
                f"{loss_rate:>7.1f}% "
                f"{draw_rate:>7.1f}%"
            )
            log.info(row)
        
        log.info("-" * 100 + "\n")

    def _print_match_stats(self, name1: str, name2: str):
        """Print statistics after a match between two agents."""
        if self.verbose < 2:
            return
            
        if self.verbose >= 3:
            self._print_comprehensive_stats(f"Statistics After Match: {name1} vs {name2}")
            return
            
        log.info(f"\nMatch Statistics ({name1} vs {name2}):")
        log.info("-" * 40)
        
        # Print ELO ratings for the two agents
        log.info("\nELO Ratings:")
        for name in [name1, name2]:
            log.info(f"{name}: {self.stats['ratings'][name].mu:.1f}")
        
        # Print win rates for the two agents
        for name in [name1, name2]:
            total_games = (self.stats['wins'][name] + 
                          self.stats['losses'][name] + 
                          self.stats['draws'][name])
            win_rate = self.stats['win_rates'][name] * 100
            log.info(f"{name}: {win_rate:.1f}% " +
                    f"(W: {self.stats['wins'][name]}, " +
                    f"L: {self.stats['losses'][name]}, " +
                    f"D: {self.stats['draws'][name]})" +
                    f"({total_games} games)")
        log.info("-" * 40 + "\n")

    def play_round(self):
        """Play one complete round of matches between all agents."""
        n_agents = len(self.agents)
        
        # Create list of all possible matches
        matches = [(i, j) for i in range(n_agents) 
                  for j in range(i + 1, n_agents)]
        
        # Randomize match order
        np.random.shuffle(matches)
        
        for i, j in matches:
            name1, name2 = self.agent_names[i], self.agent_names[j]
            if self.verbose > 0:
                log.info(f"Playing match: {name1} vs {name2}")
            
            wins1, wins2, draws = self._play_match(i, j)
            total_games = wins1 + wins2 + draws
            
            self.stats['wins'][name1] += wins1
            self.stats['wins'][name2] += wins2
            self.stats['losses'][name1] += wins2
            self.stats['losses'][name2] += wins1
            self.stats['draws'][name1] += draws
            self.stats['draws'][name2] += draws
            
            # Create series of results for both players
            series1 = []
            series2 = []
            
            # Add wins
            series1.extend([(1.0, self.stats['ratings'][name2])] * wins1)
            series2.extend([(0.0, self.stats['ratings'][name1])] * wins1)
            
            # Add losses
            series1.extend([(0.0, self.stats['ratings'][name2])] * wins2)
            series2.extend([(1.0, self.stats['ratings'][name1])] * wins2)
            
            # Add draws
            series1.extend([(0.5, self.stats['ratings'][name2])] * draws)
            series2.extend([(0.5, self.stats['ratings'][name1])] * draws)
            
            # Update ratings once for all games
            if series1:  # Only update if games were played
                self.stats['ratings'][name1] = self.glicko2.rate(self.stats['ratings'][name1], series1)
                self.stats['ratings'][name2] = self.glicko2.rate(self.stats['ratings'][name2], series2)
            
            # Store match result
            self.match_history.append({
                'agent1': name1,
                'agent2': name2,
                'wins1': wins1,
                'wins2': wins2,
                'draws': draws
            })

            # Update win rates and print match stats
            self._update_win_rates()
            self._print_match_stats(name1, name2)

        # Print round stats
        self._print_round_stats(len(self.match_history) // 
                              (len(self.agents) * (len(self.agents) - 1) // 2) - 1)

    def play_tournament(self):
        """Play all rounds of the tournament."""
        log.info(f"Starting tournament with {self.num_rounds} rounds and {len(self.agents)} agents")
        log.info(f"Agents: {'\n\t'.join(self.agent_names)}")
        for round_num in range(self.num_rounds):
            log.info(f"Starting round {round_num + 1}/{self.num_rounds}")
            self.play_round()
    
    def _update_win_rates(self):
        """Update win rate statistics for all agents."""
        for name in self.agent_names:
            total_games = (self.stats['wins'][name] + 
                         self.stats['losses'][name] + 
                         self.stats['draws'][name])
            
            self.stats['total_games'][name] = total_games
            
            if total_games > 0:
                self.stats['win_rates'][name] = self.stats['wins'][name] / total_games
                self.stats['loss_rates'][name] = self.stats['losses'][name] / total_games
                self.stats['draw_rates'][name] = self.stats['draws'][name] / total_games

    def get_results(self) -> Dict:
        """Get the current tournament results."""
        return {
            'elo_ratings': dict(sorted(
                {name: rating.mu for name, rating in self.stats['ratings'].items()}.items(),
                key=lambda x: x[1],
                reverse=True
            )),
            'win_rates': dict(sorted(
                self.stats['win_rates'].items(),
                key=lambda x: x[1],
                reverse=True
            )),
            'loss_rates': dict(self.stats['loss_rates']),
            'draw_rates': dict(self.stats['draw_rates']),
            'wins': dict(self.stats['wins']),
            'losses': dict(self.stats['losses']),
            'draws': dict(self.stats['draws']),
            'total_games': dict(self.stats['total_games']),
            'match_history': self.match_history
        }

    @staticmethod
    def create_mcts_tournament(
        game,
        nnet,
        num_sims_list: List[int],
        prefix: str = "NNMCTS",
        cpuct: float = 1.0,
        games_per_match: int = 100,
        num_rounds: int = 1,
        temperature: float = 0,
        display=None,
        mu: float = 1500,
        phi: float = 350,
        sigma: float = 0.06,
        tau: float = 0.5,
        verbose: int = 1
    ) -> 'Pit':
        """
        Create a tournament between MCTS agents with different simulation counts.
        
        Args:
            game: Game object that defines rules and mechanics
            nnet: Neural network for MCTS (should be pre-loaded with weights)
            num_sims_list: List of MCTS simulation counts to test
            cpuct: Exploration constant for MCTS
            games_per_match: Number of games to play between each pair
            num_rounds: Number of times to repeat the tournament
            temperature: Temperature for action selection (0 = deterministic)
            display: Optional display function
            mu: Starting rating for all agents (default: 1500)
            phi: Starting rating deviation (default: 350)
            sigma: Starting volatility (default: 0.06)
            tau: System constant for volatility update (default: 0.5)
            verbose: Logging verbosity level (0=none, 1=round stats, 2=match stats)
            
        Returns:
            Pit: Configured pit instance ready for tournament play
        """
        # Create agents list
        agents = []
        agent_names = []
        
        for num_sims in num_sims_list:
            # Create MCTS instance for each simulation count
            args = dotdict({
                'numMCTSSims': num_sims,
                'cpuct': cpuct,
            })
            
            agents.append(Pit.PitAgentMCTS(
                game=game,
                nnet=nnet,
                temp=temperature,
                args=args
            ))
            agent_names.append(f"{prefix}_sims{num_sims}")
        
        # Create and return pit instance
        return Pit(
            agents=agents,
            agent_names=agent_names,
            game=game,
            games_per_match=games_per_match,
            num_rounds=num_rounds,
            display=display,
            mu=mu,
            phi=phi,
            sigma=sigma,
            tau=tau,
            verbose=verbose
        )
    
    def load_checkpoints_to_pit(self, checkpoint_dir: str, num_sims_list: list = [2, 32], cpuct: float = 1.0, temperature: float = 0):
        nets = load_checkpoints(checkpoint_dir)

        safe_prefix = re.sub(r"[/\\?%*:|\"<>\x7F\x00-\x1F]", "_", checkpoint_dir).lstrip("_.-")
        for net in nets:
            self += Pit.create_mcts_tournament(
                game=self.game,
                nnet=net["net"],
                prefix=f"{safe_prefix}_iter{net['name']}",
                num_sims_list=num_sims_list,
                cpuct=cpuct,
                temperature=temperature,
            )
        print(self.agent_names)


    def add_agent(self, agent: Callable, agent_name: str, rating: Rating = None, use_average_rating: bool = False):
        """
        Add a new agent to the tournament.
        
        Args:
            agent: Agent function that takes board state and returns actions
            agent_name: Name for the agent (for tracking purposes)
            rating: Starting Rating for the agent. If None, creates new default rating
            use_average_rating: If True, uses average rating of existing agents
        """
        if agent_name in self.agent_names:
            raise ValueError(f"Agent name '{agent_name}' already exists")
            
        self.agents.append(agent)
        self.agent_names.append(agent_name)
        
        # Set initial rating
        if use_average_rating and self.stats['ratings']:
            mu = sum(r.mu for r in self.stats['ratings'].values()) / len(self.stats['ratings'])
            rating = self.glicko2.create_rating(mu=mu)
        elif rating is None:
            rating = self.glicko2.create_rating()
            
        # Initialize all stats
        self.stats['ratings'][agent_name] = rating
        self.stats['wins'][agent_name] = 0
        self.stats['losses'][agent_name] = 0
        self.stats['draws'][agent_name] = 0
        self.stats['total_games'][agent_name] = 0
        self.stats['win_rates'][agent_name] = 0.0

    def __add__(self, other: 'Pit') -> 'Pit':
        """
        Merge two Pit instances into a new one.
        
        Args:
            other: Another Pit instance to merge with
            
        Returns:
            Pit: A new Pit instance containing agents and stats from both pits
            
        Raises:
            ValueError: If the games are different or if there are duplicate agent names
        """
        if self.game != other.game:
            raise ValueError("Cannot merge Pits with different games")
            
        # Check for duplicate names
        duplicate_names = set(self.agent_names) & set(other.agent_names)
        if duplicate_names:
            raise ValueError(f"Duplicate agent names found: {duplicate_names}")
            
        # Create new pit with combined agents
        new_pit = Pit(
            agents=self.agents + other.agents,
            agent_names=self.agent_names + other.agent_names,
            game=self.game,
            games_per_match=self.games_per_match,
            num_rounds=self.num_rounds,
            display=self.display,
            mu=self.glicko2.mu,
            phi=self.glicko2.phi,
            sigma=self.glicko2.sigma,
            tau=self.glicko2.tau,
            verbose=self.verbose
        )
        
        # Merge statistics
        for name in self.agent_names:
            new_pit.stats['ratings'][name] = self.stats['ratings'][name]
            new_pit.stats['wins'][name] = self.stats['wins'][name]
            new_pit.stats['losses'][name] = self.stats['losses'][name]
            new_pit.stats['draws'][name] = self.stats['draws'][name]
            new_pit.stats['total_games'][name] = self.stats['total_games'][name]
            new_pit.stats['win_rates'][name] = self.stats['win_rates'][name]
            new_pit.stats['loss_rates'][name] = self.stats['loss_rates'][name]
            new_pit.stats['draw_rates'][name] = self.stats['draw_rates'][name]
            
        for name in other.agent_names:
            new_pit.stats['ratings'][name] = other.stats['ratings'][name]
            new_pit.stats['wins'][name] = other.stats['wins'][name]
            new_pit.stats['losses'][name] = other.stats['losses'][name]
            new_pit.stats['draws'][name] = other.stats['draws'][name]
            new_pit.stats['total_games'][name] = other.stats['total_games'][name]
            new_pit.stats['win_rates'][name] = other.stats['win_rates'][name]
            new_pit.stats['loss_rates'][name] = other.stats['loss_rates'][name]
            new_pit.stats['draw_rates'][name] = other.stats['draw_rates'][name]
            
        # Merge match history
        new_pit.match_history = self.match_history + other.match_history
        
        return new_pit


    def __iadd__(self, other: 'Pit') -> 'Pit':
        """
        In-place merge of another Pit instance into this one.
        
        Args:
            other: Another Pit instance to merge with
            
        Returns:
            Pit: Self, with agents and stats merged from other pit
            
        Raises:
            ValueError: If the games are different or if there are duplicate agent names
        """
        if self.game != other.game:
            raise ValueError("Cannot merge Pits with different games")
            
        # Check for duplicate names
        duplicate_names = set(self.agent_names) & set(other.agent_names)
        if duplicate_names:
            raise ValueError(f"Duplicate agent names found: {duplicate_names}")
            
        # Add agents and names
        self.agents.extend(other.agents)
        self.agent_names.extend(other.agent_names)
        
        # Merge statistics
        for name in other.agent_names:
            self.stats['ratings'][name] = other.stats['ratings'][name]
            self.stats['wins'][name] = other.stats['wins'][name]
            self.stats['losses'][name] = other.stats['losses'][name]
            self.stats['draws'][name] = other.stats['draws'][name]
            self.stats['total_games'][name] = other.stats['total_games'][name]
            self.stats['win_rates'][name] = other.stats['win_rates'][name]
            self.stats['loss_rates'][name] = other.stats['loss_rates'][name]
            self.stats['draw_rates'][name] = other.stats['draw_rates'][name]
            
        # Merge match history
        self.match_history.extend(other.match_history)
        
        return self