import numpy as np
from random import choice
from typing import Tuple


WRONG_MOVE_REWARD = 0
GAME_END_REWARD = -1000

def start() -> np.array:
    board = np.zeros([4, 4], "int64")
    return __add_random(board)

def get_available_actions(board: np.array) -> list:
    available_actions = []
    if board.tolist() != right(board, check_game_over=False)[0].tolist():
        available_actions.append("right")
    if board.tolist() != left(board, check_game_over=False)[0].tolist():
        available_actions.append("left")
    if board.tolist() != up(board, check_game_over=False)[0].tolist():
        available_actions.append("up")
    if board.tolist() != down(board, check_game_over=False)[0].tolist():
        available_actions.append("down")
    return available_actions



def __is_game_over(board: np.array) -> bool:
    if board.tolist() != right(board, check_game_over=False)[0].tolist():
        return False
    if board.tolist() != left(board, check_game_over=False)[0].tolist():
        return False
    if board.tolist() != up(board, check_game_over=False)[0].tolist():
        return False
    if board.tolist() != down(board, check_game_over=False)[0].tolist():
        return False
    return True

def __add_random(b_: np.array, init=2) -> np.array:
    board = b_.copy()
    available = []
    for row in range(len(board)):
        for col in range(len(board[0])):
            if board[row][col] == 0:
                available.append((row, col))
    
    selected = choice(available)
    board[selected[0]][selected[1]] = init
    return board

def __collapse_row_right(row_: list) -> Tuple[list, int]:
    row = row_.copy()

    points = 0

    i = len(row) - 1
    while i > 0:
        if row[i] != 0:
            for j in range(i-1, -1, -1):
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
                row[j] = row[j-1]
            row[0] = 0

        found_item = row[i] != 0
    
    return row, points

def right(b_: np.array, check_game_over=True) -> Tuple[np.array, int, bool]:
    board = b_.copy()
    points = 0
    for row_num in range(len(board)):
        board[row_num], p = __collapse_row_right(list(board[row_num]))
        points += p
    if board.tolist() != b_.tolist():
        board = __add_random(board)
    else:
        points = WRONG_MOVE_REWARD
    if check_game_over:
        if __is_game_over(board):
            return board, GAME_END_REWARD, True
    return board, points, False

def left(b_: np.array, check_game_over=True) -> Tuple[np.array, int, bool]:
    board = b_.copy()
    points = 0
    for row_num in range(len(board)):
        new_row, p = __collapse_row_right(list(board[row_num])[::-1])
        new_row = new_row[::-1]
        board[row_num] = new_row
        points += p
    if board.tolist() != b_.tolist():
        board = __add_random(board)
    else:
        points = WRONG_MOVE_REWARD
    if check_game_over:
        if __is_game_over(board):
            return board, GAME_END_REWARD, True
    return board, points, False

def down(b_: np.array, check_game_over=True) -> Tuple[np.array, int, bool]:
    board = b_.copy()
    points = 0
    for col_num in range(len(board[0])):
        board[:, col_num], p = __collapse_row_right(list(board[:, col_num]))
        points += p
    if board.tolist() != b_.tolist():
        board = __add_random(board)
    else:
        points = WRONG_MOVE_REWARD
    if check_game_over:
        if __is_game_over(board):
            return board, GAME_END_REWARD, True
    return board, points, False

def up(b_: np.array, check_game_over=True) -> Tuple[np.array, int, bool]:
    board = b_.copy()
    points = 0
    for col_num in range(len(board[0])):
        # display(board[:, col_num])
        new_col, p = __collapse_row_right(list(board[:, col_num])[::-1])
        new_col = new_col[::-1]
        board[:, col_num] = new_col
        points += p
    if board.tolist() != b_.tolist():
        board = __add_random(board)
    else:
        points = WRONG_MOVE_REWARD
    if check_game_over:
        if __is_game_over(board):
            return board, GAME_END_REWARD, True
    return board, points, False