from src.base.game import Game
from utac_gym.core import GameState
from utac_gym.core.types import GAMESTATE
import numpy as np

class UtacGame(Game):
    def __init__(self):
        pass

    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        return GameState()

    def getBoardSize(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        return (9, 9, 2)

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        return 81

    def getNextState(self, board: GameState, player: int, action: int):
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """
        board = GameState(board)
        board.make_move(action)
        return board, -player

    def getValidMoves(self, board: GameState, player: int):
        """
        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        if player != self._get_current_player(board):
            return np.zeros(81)
        return board.get_valid_mask()

    def getGameEnded(self, board: GameState, player: int):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.
               
        """
        if not board.is_game_over():
            return 0
        
        return 1 if board.terminal_value() == player else -1

    def getCanonicalForm(self, board: GameState, player: int):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        obs = board.get_obs()
        obs[0] *= player
        return obs

    def getSymmetries(self, board: GameState, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        pi_board = np.reshape(pi, (9, 9))
        symmetries = []
        obs = self._get_obs(board)
        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(obs, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                symmetries.append((newB, newPi.flatten()))
        return symmetries

    def stringRepresentation(self, board: GameState):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        gs: GAMESTATE = board._get_gs()
        board_str = occ_str = ""
        for i in range(9):
            board_str += format(gs.board[i], '09b')
            occ_str += format(gs.occ[i], '09b')
        s = board_str + occ_str + str(gs.side) + str(gs.last_square)
        return s

    def _get_current_player(self, board: GameState):
        return 1 if board.current_player() == 1 else -1
    
    def _get_obs(self, board: GameState):
        obs = board.get_obs()
        obs = np.array(obs).reshape(2, 9, 9).transpose(1, 2, 0)
        return obs
