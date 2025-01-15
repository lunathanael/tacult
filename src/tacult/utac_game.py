from tacult.base.game import Game
from utac_gym.core import GameState
from utac_gym.core.types import GAMESTATE
import numpy as np
import torch


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
        return (9, 9, 4)

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
        if not board.is_terminal():
            return 0

        if board.terminal_value() == 0:
            return 1e-4

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
        if player == 1:
            return board
        else:
            state = GameState(board)._get_gs()
            b = state.board
            for i in range(9):
                b[i] = (~b[i]) & state.occ[i]
            state.main_board = (~state.main_board) & state.main_occ
            state.side = 1
            return GameState(state)

    @staticmethod
    def _getSymmetriesHelper(
        occ_arrays,
        board_arrays,
        game_occ_array,
        main_occ_array,
        main_board_array,
        last_square_array,
        pi_board,
        i,
        j,
    ):
        last_square_array = last_square_array.reshape(3, 3)

        # Rotate arrays
        occ_arrays = np.rot90(occ_arrays, i)
        board_arrays = np.rot90(board_arrays, i)
        game_occ_array = np.rot90(game_occ_array, i)
        main_occ_array = np.rot90(main_occ_array, i)
        main_board_array = np.rot90(main_board_array, i)
        last_square_array = np.rot90(last_square_array, i)
        newPi = np.rot90(pi_board, i)

        if j:
            occ_arrays = np.fliplr(occ_arrays)
            board_arrays = np.fliplr(board_arrays)
            game_occ_array = np.fliplr(game_occ_array)
            main_occ_array = np.fliplr(main_occ_array)
            main_board_array = np.fliplr(main_board_array)
            last_square_array = np.fliplr(last_square_array)
            newPi = np.fliplr(newPi)
        # Convert back to bitboards
        state_occ = array_to_array_of_bitboards(occ_arrays)
        state_board = array_to_array_of_bitboards(board_arrays)

        state_game_occ = array_to_bitboard(game_occ_array.ravel())
        state_main_occ = array_to_bitboard(main_occ_array.ravel())
        state_main_board = array_to_bitboard(main_board_array.ravel())
        state_last_square = np.argmax(last_square_array.ravel())
        newPi = newPi.flatten()

        return (
            state_occ,
            state_board,
            state_game_occ,
            state_main_occ,
            state_main_board,
            state_last_square,
            newPi,
        )

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

        state = GameState(board)._get_gs()
        for i in range(1, 5):
            for j in [True, False]:
                # Extract bitboards from state
                occ_arrays = array_of_bitboards_to_array(state.occ)
                board_arrays = array_of_bitboards_to_array(state.board)

                game_occ_array = bitboard_to_array(state.game_occ).reshape(3, 3)
                main_occ_array = bitboard_to_array(state.main_occ).reshape(3, 3)
                main_board_array = bitboard_to_array(state.main_board).reshape(3, 3)
                last_square_array = np.zeros(9, dtype=bool)
                if state.last_square != -1:
                    last_square_array[state.last_square] = 1

                (
                    state_occ,
                    state_board,
                    state_game_occ,
                    state_main_occ,
                    state_main_board,
                    state_last_square,
                    newPi
                ) = self._getSymmetriesHelper(
                    occ_arrays,
                    board_arrays,
                    game_occ_array,
                    main_occ_array,
                    main_board_array,
                    last_square_array,
                    pi_board,
                    i,
                    j,
                )

                # Convert back to bitboards
                state.occ = state_occ
                state.board = state_board

                state.game_occ = state_game_occ
                state.main_occ = state_main_occ
                state.main_board = state_main_board
                if state.last_square != -1:
                    state.last_square = state_last_square

                newB = GameState(state)
                symmetries.append((newB, newPi.flatten()))
        return symmetries
    
    @staticmethod
    def _stringRepresentationHelper(board, occ, side, last_square):
        board_str = occ_str = ""
        for i in range(9):
            board_str += np.binary_repr(board[i], width=9)
            occ_str += np.binary_repr(occ[i], width=9)
        s = board_str + occ_str + str(side) + str(last_square)
        return s

    def stringRepresentation(self, board: GameState):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        gs: GAMESTATE = board._get_gs()
        return self._stringRepresentationHelper(gs.board, gs.occ, gs.side, gs.last_square)

    def _get_current_player(self, board: GameState):
        return 1 if board.current_player() == 1 else -1


    def _get_obs(self, board: GameState):
        obs = np.array(board.get_obs())
        obs = obs.reshape(4, 9, 9)
        return torch.from_numpy(obs)


def bitboard_to_array(_bitboard: int):
    last_bit = bool(_bitboard & 0x100)
    _bitboard = _bitboard & 0xFF
    bitboard = np.array([_bitboard], dtype=np.uint8)
    arr = np.unpackbits(bitboard, bitorder="little", count=9)
    arr[8] = last_bit
    return arr


def array_to_bitboard(array: np.ndarray):
    return np.packbits(array.ravel(), bitorder="little")[0]



def array_of_bitboards_to_array(array_of_bitboards: np.ndarray):
    arrays = [bitboard_to_array(bitboard) for bitboard in array_of_bitboards]
    result = np.zeros((9, 9), dtype=np.int8)
    for i, arr in enumerate(arrays):
        grid_row = (i // 3) * 3
        grid_col = (i % 3) * 3
        # Place the 9 elements into their corresponding 3x3 subgrid
        result[grid_row : grid_row + 3, grid_col : grid_col + 3] = arr.reshape(3, 3)
    return result



def array_to_array_of_bitboards(array: np.ndarray):
    bitboards = []
    # Process each 3x3 subgrid
    for i in range(3):
        for j in range(3):
            # Extract the 3x3 subgrid
            grid_row = i * 3
            grid_col = j * 3
            subgrid = array[grid_row : grid_row + 3, grid_col : grid_col + 3]
            # Convert the subgrid to a bitboard
            bitboard = array_to_bitboard(subgrid)
            bitboards.append(bitboard)
    return np.array(bitboards)
