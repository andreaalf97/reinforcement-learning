import math
from random import choice
from typing import Callable, Dict, Tuple

import numpy as np


def generate_max_moves_list(args) -> list:
    # Create the list of max moves per episode following a sigmoid trend between min and max moves
    def sigmoid(x):
        return (1 / (1 + pow(math.e, -x))) * (
            args.max_moves_end - args.max_moves_start
        ) + args.max_moves_start

    max_moves_per_episode = [
        sigmoid(i / 100) for i in range(-600, 600, int(1200 / args.episodes))
    ][: args.episodes]
    while len(max_moves_per_episode) < args.episodes:
        max_moves_per_episode.append(max_moves_per_episode[-1])
    return max_moves_per_episode


class GameSimulator:
    def __is_game_over(self, board: np.ndarray) -> bool:
        if board.tolist() != self.__right(board, check_game_over=False)[0].tolist():
            return False
        if board.tolist() != self.__left(board, check_game_over=False)[0].tolist():
            return False
        if board.tolist() != self.__up(board, check_game_over=False)[0].tolist():
            return False
        if board.tolist() != self.__down(board, check_game_over=False)[0].tolist():
            return False
        return True

    def get_available_actions(self) -> list:
        available_actions = []
        if (
            self.board.tolist()
            != self.__right(self.board, check_game_over=False)[0].tolist()
        ):
            available_actions.append("right")
        if (
            self.board.tolist()
            != self.__left(self.board, check_game_over=False)[0].tolist()
        ):
            available_actions.append("left")
        if (
            self.board.tolist()
            != self.__up(self.board, check_game_over=False)[0].tolist()
        ):
            available_actions.append("up")
        if (
            self.board.tolist()
            != self.__down(self.board, check_game_over=False)[0].tolist()
        ):
            available_actions.append("down")
        return available_actions

    def __init__(
        self,
        max_moves: float,
        wrong_move_reward: int = 0,
        game_end_reward: int = -1000,
        is_conv: bool = False,
    ) -> None:
        assert isinstance(
            max_moves, int
        ), f"Expected type int for `max_moves`, received {type(max_moves)}"

        self.max_moves = max_moves
        self.wrong_move_reward = wrong_move_reward
        self.game_end_reward = game_end_reward
        self.is_conv = is_conv

        self.board = np.zeros([4, 4], "int64")
        self.board = self.__add_random(self.board)

        self.actions: Dict[str, Callable] = {
            "right": self.right,
            "left": self.left,
            "up": self.up,
            "down": self.down,
        }

    def __add_random(self, board_, init=2) -> np.ndarray:
        board = board_.copy()
        available = []
        for row in range(len(board)):
            for col in range(len(board[0])):
                if board[row][col] == 0:
                    available.append((row, col))

        selected = choice(available)
        board[selected[0]][selected[1]] = init
        return board

    def __collapse_row_right(self, row_: list) -> Tuple[list, int]:
        row = row_.copy()

        points = 0

        i = len(row) - 1
        while i > 0:
            if row[i] != 0:
                for j in range(i - 1, -1, -1):
                    if row[i] == row[j]:
                        row[i] = row[i] + row[j]
                        points += row[i]
                        row[j] = 0
                        i = j
                        break
                    if row[j] != 0:
                        break
            i -= 1

        found_item = False
        for i in range(len(row)):

            if row[i] == 0 and found_item:
                for j in range(i, 0, -1):
                    row[j] = row[j - 1]
                row[0] = 0

            found_item = row[i] != 0

        return row, points

    def __right(
        self, b_: np.ndarray, check_game_over=True
    ) -> Tuple[np.ndarray, int, bool]:
        board = b_.copy()
        points = 0
        for row_num in range(len(board)):
            board[row_num], p = self.__collapse_row_right(list(board[row_num]))
            points += p
        if board.tolist() != b_.tolist():
            board = self.__add_random(board)
        else:
            points = self.wrong_move_reward
        if check_game_over:
            if self.__is_game_over(board):
                return board, self.game_end_reward, True
        return board, points, False

    def __left(
        self, b_: np.ndarray, check_game_over=True
    ) -> Tuple[np.ndarray, int, bool]:
        board = b_.copy()
        points = 0
        for row_num in range(len(board)):
            new_row, p = self.__collapse_row_right(list(board[row_num])[::-1])
            new_row = new_row[::-1]
            board[row_num] = new_row
            points += p
        if board.tolist() != b_.tolist():
            board = self.__add_random(board)
        else:
            points = self.wrong_move_reward
        if check_game_over:
            if self.__is_game_over(board):
                return board, self.game_end_reward, True
        return board, points, False

    def __down(
        self, b_: np.ndarray, check_game_over=True
    ) -> Tuple[np.ndarray, int, bool]:
        board = b_.copy()
        points = 0
        for col_num in range(len(board[0])):
            board[:, col_num], p = self.__collapse_row_right(list(board[:, col_num]))
            points += p
        if board.tolist() != b_.tolist():
            board = self.__add_random(board)
        else:
            points = self.wrong_move_reward
        if check_game_over:
            if self.__is_game_over(board):
                return board, self.game_end_reward, True
        return board, points, False

    def __up(
        self, b_: np.ndarray, check_game_over=True
    ) -> Tuple[np.ndarray, int, bool]:
        board = b_.copy()
        points = 0
        for col_num in range(len(board[0])):
            # display(board[:, col_num])
            new_col, p = self.__collapse_row_right(list(board[:, col_num])[::-1])
            new_col = new_col[::-1]
            board[:, col_num] = new_col
            points += p
        if board.tolist() != b_.tolist():
            board = self.__add_random(board)
        else:
            points = self.wrong_move_reward
        if check_game_over:
            if self.__is_game_over(board):
                return board, self.game_end_reward, True
        return board, points, False

    def board_to_state(self) -> np.ndarray:
        if self.is_conv:
            # Convolutions expect shape like [batch_size, n_channels, x, y]
            out = self.board[np.newaxis, ...]
            return out
        # Linear models expect shape like [batch_size, x]
        out = self.board.flatten()
        return out

    def right(self) -> Tuple[int, bool]:
        self.board, reward, done = self.__right(self.board, check_game_over=True)
        return reward, done

    def left(self) -> Tuple[int, bool]:
        self.board, reward, done = self.__left(self.board, check_game_over=True)
        return reward, done

    def up(self) -> Tuple[int, bool]:
        self.board, reward, done = self.__up(self.board, check_game_over=True)
        return reward, done

    def down(self) -> Tuple[int, bool]:
        self.board, reward, done = self.__down(self.board, check_game_over=True)
        return reward, done

    def move(self, action: str) -> Tuple[int, bool]:
        return self.actions[action]()
