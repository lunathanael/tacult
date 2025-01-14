import logging

from tqdm import tqdm
import time
import numpy as np

log = logging.getLogger(__name__)


from .utils import to_obs


class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1, player2, game, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

    def playGame(self, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [self.player2, None, self.player1]
        curPlayer = 1
        board = self.game.getInitBoard()
        it = 0

        for player in players[0], players[2]:
            if hasattr(player, "startGame"):
                player.startGame()

        while self.game.getGameEnded(board, curPlayer) == 0:
            it += 1
            if verbose:
                assert self.display
                print("Turn ", str(it), "Player ", str(curPlayer))
                self.display(board)
            action = players[curPlayer + 1](self.game.getCanonicalForm(board, curPlayer))
            valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer), 1)

            if valids[action] == 0:
                log.error(f'Action {action} is not valid!')
                log.debug(f'valids = {valids}')
                assert valids[action] > 0

            # Notifying the opponent for the move
            opponent = players[-curPlayer + 1]
            if hasattr(opponent, "notify"):
                opponent.notify(board, action)

            board, curPlayer = self.game.getNextState(board, curPlayer, action)

        for player in players[0], players[2]:
            if hasattr(player, "endGame"):
                player.endGame()

        if verbose:
            assert self.display
            print("Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(board, 1)))
            self.display(board)
        return curPlayer * self.game.getGameEnded(board, curPlayer)

    def playGames(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """

        num = int(num / 2)
        oneWon = 0
        twoWon = 0
        draws = 0
        for _ in tqdm(range(num), desc="Arena.playGames (1)"):
            gameResult = self.playGame(verbose=verbose)
            if gameResult == 1:
                oneWon += 1
            elif gameResult == -1:
                twoWon += 1
            else:
                draws += 1

        self.player1, self.player2 = self.player2, self.player1

        for _ in tqdm(range(num), desc="Arena.playGames (2)"):
            gameResult = self.playGame(verbose=verbose)
            if gameResult == -1:
                oneWon += 1
            elif gameResult == 1:
                twoWon += 1
            else:
                draws += 1

        return oneWon, twoWon, draws


class VectorizedArena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1, player2, game, display, num_envs):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display
        self.num_envs = num_envs

    def playGame(self, verbose=False):
        """
        Executes one episode of a game on all environments.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [self.player2, None, self.player1]
        curPlayer = 1
        boards = [self.game.getInitBoard() for _ in range(self.num_envs)]
        terminal_boards = np.zeros(self.num_envs, dtype=bool)
        it = 0

        for player in players[0], players[2]:
            if hasattr(player, "startGame"):
                player.startGame()
        
        while not np.all(terminal_boards):
            it += 1
            if verbose:
                assert self.display
                # for i, board in enumerate(boards):
                #     if not terminal_boards[i]:
                #         print("Game ", str(i), "Turn ", str(it), "Player ", str(curPlayer))
                #         self.display(board)

            canonicalBoards = [
                self.game.getCanonicalForm(board, curPlayer) for board in boards
            ]
            actions = players[curPlayer + 1](canonicalBoards)

            for i, board in enumerate(boards):
                if terminal_boards[i]:
                    continue
                valids = self.game.getValidMoves(canonicalBoards[i], 1)
                action = actions[i]

                if valids[action] == 0:
                    log.error(f'Action {action} is not valid!')
                    log.debug(f'valids = {valids}')
                    assert valids[action] > 0

                # Notifying the opponent for the move
                opponent = players[-curPlayer + 1]
                if hasattr(opponent, "notify"):
                    opponent.notify(board, action)

                boards[i], nextPlayer = self.game.getNextState(board, curPlayer, action)

                value = self.game.getGameEnded(boards[i], nextPlayer)
                if value != 0:
                    terminal_boards[i] = True
                    if verbose:
                        assert self.display
                        # print("Game ", str(i), "Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(boards[i], 1)))
                        # self.display(board)
                    yield nextPlayer * value
            curPlayer = nextPlayer

        for player in players[0], players[2]:
            if hasattr(player, "endGame"):
                player.endGame()

        return

    def playGames(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """

        num = int(num / 2)
        oneWon = 0
        twoWon = 0
        draws = 0

        num_games_played = 0
        time_started = time.time()
        print(
            tqdm.format_meter(
                n=num_games_played,
                total=num,
                elapsed=0,
                unit="game",
                prefix="Arena.playGames (1): "
            ),
            end="\r"
        )
        while num_games_played < num:
            for result in self.playGame(verbose=verbose):
                if result == 1:
                    oneWon += 1
                elif result == -1:
                    twoWon += 1
                else:
                    draws += 1
                num_games_played += 1
                elapsed = time.time() - time_started
                print(
                    tqdm.format_meter(
                        n=num_games_played,
                        total=num,
                        elapsed=elapsed,
                        unit="game",
                        prefix="Arena.playGames (1): "
                    ),
                    end="\r"
                )
        print()

        self.player1, self.player2 = self.player2, self.player1

        num_games_played = 0
        time_started = time.time()
        print(
            tqdm.format_meter(
                n=num_games_played,
                total=num,
                elapsed=0,
                unit="game",
                prefix="Arena.playGames (2): "
            ),
            end="\r"
        )
        while num_games_played < num:
            for result in self.playGame(verbose=verbose):
                if result == -1:
                    oneWon += 1
                elif result == 1:
                    twoWon += 1
                else:
                    draws += 1
                num_games_played += 1
                elapsed = time.time() - time_started
                print(
                    tqdm.format_meter(
                        n=num_games_played,
                        total=num,
                        elapsed=elapsed,
                        unit="game",
                        prefix="Arena.playGames (2): "
                    ),
                    end="\r"
                )
        print()

        return oneWon, twoWon, draws