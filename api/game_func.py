import numpy as np
from typing import Tuple

WRONG_MOVE_REWARD = -1

def init_board() -> np.array:
    board = np.zeros([4, 4], "int64")
    return add_random(board.copy())

def is_game_over(board) -> bool:
    if board.tolist() != shift_right(board, check_game_over=False)[0].tolist():
        return False
    if board.tolist() != shift_left(board, check_game_over=False)[0].tolist():
        return False
    if board.tolist() != shift_up(board, check_game_over=False)[0].tolist():
        return False
    if board.tolist() != shift_left(board, check_game_over=False)[0].tolist():
        return False
    return True

def add_random(b_: np.array, init=2) -> np.array:
    board = b_.copy()
    from random import choice
    available = []
    for row in range(len(board)):
        for col in range(len(board[0])):
            if board[row][col] == 0:
                available.append((row, col))
    
    selected = choice(available)
    board[selected[0]][selected[1]] = init
    return board

def collapse_row_right(row_: list) -> Tuple[list, int]:
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

def shift_right(b_: np.array, check_game_over=True) -> Tuple[np.array, int]:
    board = b_.copy()
    points = 0
    for row_num in range(len(board)):
        board[row_num], p = collapse_row_right(list(board[row_num]))
        points += p
    if board.tolist() != b_.tolist():
        board = add_random(board)
    if check_game_over:
        if is_game_over(board):
            return board, -1
    return board, points

def shift_left(b_: np.array, check_game_over=True) -> Tuple[np.array, int]:
    board = b_.copy()
    points = 0
    for row_num in range(len(board)):
        new_row, p = collapse_row_right(list(board[row_num])[::-1])
        new_row = new_row[::-1]
        board[row_num] = new_row
        points += p
    if board.tolist() != b_.tolist():
        board = add_random(board)
    if check_game_over:
        if is_game_over(board):
            return board, -1
    return board, points

def shift_down(b_: np.array, check_game_over=True) -> Tuple[np.array, int]:
    board = b_.copy()
    points = 0
    for col_num in range(len(board[0])):
        board[:, col_num], p = collapse_row_right(list(board[:, col_num]))
        points += p
    if board.tolist() != b_.tolist():
        board = add_random(board)
    if check_game_over:
        if is_game_over(board):
            return board, -1
    return board, points

def shift_up(b_: np.array, check_game_over=True) -> Tuple[np.array, int]:
    board = b_.copy()
    points = 0
    for col_num in range(len(board[0])):
        # display(board[:, col_num])
        new_col, p = collapse_row_right(list(board[:, col_num])[::-1])
        new_col = new_col[::-1]
        board[:, col_num] = new_col
        points += p
    if board.tolist() != b_.tolist():
        board = add_random(board)
    if check_game_over:
        if is_game_over(board):
            return board, -1
    return board, points