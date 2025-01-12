import gymnasium as gym
from gymnasium import spaces
import numpy as np
import utac
from utac.types import Board, SubBoard


def cast_to_board(subboard: SubBoard) -> Board:
    board = np.zeros((9, 9), dtype=np.int8)
    for i in range(3):
        for j in range(3):
            if subboard.board[i, j] != 0:
                row_start = subboard.start_row + (i * 3)
                col_start = subboard.start_col + (j * 3)
                board[row_start : row_start + 3, col_start : col_start + 3] = (
                    subboard.board[i, j]
                )
    return board


class UtacEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        # Observation space: 3 binary planes of 9x9
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(3, 9, 9), dtype=np.int8
        )

        # Action space: 81 possible moves (9x9 grid)
        self.action_space = spaces.Discrete(81)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None
        self.state: utac.UtacState = None

    def _get_obs(self):
        # Convert UtacState to 3 binary planes
        boardX: Board = self.state.boardX.astype(np.int8)
        boardO: Board = self.state.boardO.astype(np.int8)

        current_subboard_index: int = self.state.current_subboard_index
        board_mask: Board = np.zeros((9, 9), dtype=np.int8)
        if current_subboard_index == -1:
            board_mask = np.ones((9, 9), dtype=np.int8)
        else:
            for i in range(3):
                for j in range(3):
                    move = (current_subboard_index, i, j)
                    move_index = utac.utils.move_to_index(move)
                    move_coord = utac.utils.index_to_board_coord(move_index)
                    board_mask[move_coord] = True

        if self.state.current_player == "O":
            boardX, boardO = boardO, boardX

        observation = np.stack([boardX, boardO, board_mask], axis=0)
        return observation

    def _get_features(self):
        # Convert UtacState to 3 binary planes
        boardX: Board = self.state.boardX.astype(np.int8)
        boardO: Board = self.state.boardO.astype(np.int8)
        main_boardX: SubBoard = self.state.main_boardX.astype(np.int8)
        main_boardO: SubBoard = self.state.main_boardO.astype(np.int8)
        main_boardDraw: SubBoard = self.state.main_boardDraw.astype(np.int8)

        main_boardX = cast_to_board(main_boardX)
        main_boardO = cast_to_board(main_boardO)
        main_boardDraw = cast_to_board(main_boardDraw)

        current_subboard_index: int = self.state.current_subboard_index
        board_mask: Board = np.zeros((9, 9), dtype=np.int8)
        if current_subboard_index == -1:
            board_mask = np.ones((9, 9), dtype=np.int8)
        else:
            for i in range(3):
                for j in range(3):
                    move = (current_subboard_index, i, j)
                    move_index = utac.utils.move_to_index(move)
                    move_coord = utac.utils.index_to_board_coord(move_index)
                    board_mask[move_coord] = True

        if self.state.current_player == "O":
            boardX, boardO = boardO, boardX
            main_boardX, main_boardO = main_boardO, main_boardX

        observation = np.stack(
            [boardX, boardO, board_mask, main_boardX, main_boardO, main_boardDraw],
            axis=0,
        )
        return observation

    def _get_info(self):
        return {
            "current_subboard_index": self.state.current_subboard_index,
            "current_player": self.state.current_player,
            "winner": self.state.winner,
            "game_over": self.state.game_over,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize new game state
        self.state = utac.UtacState(current_subboard_index=-1)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        current_player = self.state.current_player
        self.state.make_move(action)

        # Check if game is over
        terminated = self.state.game_over

        # Reward: 1 for win, 0 for ongoing, -1 for invalid/loss
        reward = 1 if terminated and self.state.winner == current_player else 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def apply_move_index(self, move_index: int):
        self.state.make_move_index(move_index)

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        # Implement rendering logic here if needed
        pass

    def close(self):
        if self.window is not None:
            pass

    def get_legal_actions(self):
        return self.state.get_legal_moves()

    def is_terminal(self):
        return self.state.game_over

    def clone(self):
        clone = UtacEnv()
        clone.state = self.state.copy()
        return clone

    @property
    def current_player(self):
        return self.state.current_player == "X"

    def get_reward(self, current_player: int):
        players = ["O", "X"]
        if self.state.winner not in ("X", "O"):
            return 0
        if self.state.winner == players[current_player]:
            return 1
        return -1
