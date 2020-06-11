#! /usr/bin/python3

import json


def save_board(game, path):

    game_board = {
        "width": game.width,
        "height": game.height,
        "board_state": game.board_state.tolist(),
    }

    with open(path, "w") as f:
        json.dump(game_board, f)


def load_board(game, path):

    with open(path, "r") as f:
        board = json.load(f)

    board = game.from_json(board)

    return board
