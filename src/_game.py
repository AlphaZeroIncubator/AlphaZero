#! /usr/bin/python3

import torch
from typing import Union


class Game:

    def __init__(self, *args):
        """
        Construct an instance of the game class.
        """
        raise NotImplementedError

    def make_move(self, move):
        """
        Make a move on the board. Should be a valid move and return a valid
        game board.
        """
        raise NotImplementedError

    @staticmethod
    def get_initial_board():
        """
        Get the initial game board for this game. For Connect 4, for instance,
        this should be an empty board of width `width` and height `height`, but
        for chess this may be the initial position of the pieces.
        """
        raise NotImplementedError

    def get_legal_moves(self) -> torch.Tensor:
        """
        Get a dynamic list of legal moves for the current game position. Should
        return a `torch.Tensor` full of booleans that represent whether a move
        is valid or not.
        """
        raise NotImplementedError

    @staticmethod
    def is_game_over(board) -> bool:
        """
        Check if the board state represents a game that is over. This should
        just return a boolean True/False.
        """
        raise NotImplementedError

    @staticmethod
    def result(board) -> Union[None, int]:
        """
        Get the result of the game. This should be 1 for a win for the player
        whose perspective we're looking for, 0 for a draw, -1 for a loss, or
        None for undetermined, when the game is not over yet.
        """
        raise NotImplementedError

    @property
    def width(self) -> int:
        """
        Get the width of the game board. For tic-tac-toe, for instance, this
        should return 3; for chess, 8.
        """
        raise NotImplementedError

    @property
    def height(self) -> int:
        """
        Get the height of the game board. For tic-tac-toe, this should return
        3; for chess, 8.
        """
        raise NotImplementedError

    @property
    def board_state(self) -> torch.Tensor:
        """
        Get the current board state as a torch.Tensor.
        """
        raise NotImplementedError

    @classmethod
    def from_json(cls, params):
        """
        Create an instance of class `cls` starting from a JSON style dict
        `params`. This dict should contain all the width, height, board state
        parameters that may be needed to create an instance of this game class.
        """
        raise NotImplementedError

        # can we just return a generic thing like this?
        # return cls(params)

    def reset(self) -> None:
        """
        Reset the game to the initial board position (as given by
        `get_initial_board()`). Reset move lists, move counters, anything else.
        """
        raise NotImplementedError

    def is_valid(self) -> bool:
        """
        Check if the current board position is valid.
        """
        raise NotImplementedError

    @staticmethod
    def mirror_h(board):
        """
        Mirror the board position horizontally. Should be implemented per
        subclass.
        """
        raise NotImplementedError

    @staticmethod
    def mirror_v(board):
        """
        Mirror the board position vertically. Should be implemented per
        subclass.

        For some games, like Connect 4, this may not be a valid transformation.
        This may require handling downstream in whatever uses this function.
        """
        raise NotImplementedError
