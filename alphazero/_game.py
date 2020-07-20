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
    def board_after_move(board, move):
        """
        Make a move on the board and return it. Does not affect any instances
        of the class. Should be a valid move and return a valid game board.
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

    @staticmethod
    def get_legal_moves(board) -> torch.Tensor:
        """
        Get a list of legal moves for the given board game position. Should
        return a `torch.Tensor` full of booleans that represent whether a move
        is valid or not. Does not affect internal state of any instances of
        this class.
        """
        raise NotImplementedError

    def current_legal_moves(self) -> torch.Tensor:
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

    @staticmethod
    def n_players():
        """
        Get the number of players in the game.
        """
        raise NotImplementedError


class TicTacToe(Game):
    def __init__(self, width=3, height=3, board_state=None):
        """
        Construct an instance of the game class.
        """
        if board_state is None:
            self._board = self.get_initial_board(width, height)
        else:
            tensor = torch.Tensor(board_state)
            if width * height != tensor.numel():
                raise ValueError(
                    f"width and height dimensions {width}, {height} do not "
                    f"match board state size {tensor.shape}"
                )
            self._board = tensor.view(width, height)
        self._move_count = 0
        self._players = [0, 1]

    def make_move(self, move: (int, int)):
        """
        Make a move on the board. Should be a valid move and return a valid
        game board.
        """
        self._board = self.board_after_move(
            self._board, self._players[self._move_count % 2], move
        )
        self._move_count += 1

        if not self.is_valid():
            raise ValueError(f"Board state {self._board} is not valid.")

    @staticmethod
    def board_after_move(board, player: int, move: (int, int)):
        """
        Make a move on the board and return it. Does not affect any instances
        of the class. Should be a valid move and return a valid game board.
        """
        if not isinstance(move, tuple):
            raise TypeError(f"move is of type {type(move)} but must be tuple")
        if not len(move) == 2:
            raise ValueError(f"move is of length {len(move)} but must be 3")

        if player not in (0, 1):
            raise ValueError(f"invalid player for move {move}")

        if (
            move[0] > board.shape[0]
            or move[1] > board.shape[1]
            or any([m < 0 for m in move[1:]])
        ):
            raise IndexError(
                f"invalid move {move} for board shaped {board.shape}"
            )

        if board[move] != -1:
            raise ValueError(
                f"invalid move {move}: space already occupied in board {board}"
            )

        new_board = board.clone()
        new_board[move] = player

        return new_board

    @staticmethod
    def get_initial_board(width=3, height=3):
        """
        Get the initial game board for this game. For Connect 4, for instance,
        this should be an empty board of width `width` and height `height`, but
        for chess this may be the initial position of the pieces.
        """
        return torch.full((width, height), -1)

    @staticmethod
    def get_legal_moves(board) -> torch.Tensor:
        """
        Get a list of legal moves for the given board game position. Should
        return a `torch.Tensor` full of booleans that represent whether a move
        is valid or not. Does not affect internal state of any instances of
        this class.
        """
        return board == -1

    def current_legal_moves(self) -> torch.Tensor:
        """
        Get a dynamic list of legal moves for the current game position. Should
        return a `torch.Tensor` full of booleans that represent whether a move
        is valid or not.
        """
        return self.get_legal_moves(self._board)

    @staticmethod
    def get_game_status(board) -> tuple:
        verts_0 = any((board == 0).sum(dim=0) == 3)
        horizontals_0 = any((board == 0).sum(dim=1) == 3)

        verts_1 = any((board == 1).sum(dim=0) == 3)
        horizontals_1 = any((board == 1).sum(dim=1) == 3)

        n = board.shape[0]

        first_diag = [board[i, i].item() for i in range(n)]

        first_diag_0 = all([d == 0 for d in first_diag])
        first_diag_1 = all([d == 1 for d in first_diag])

        second_diag = [board[i, n - i - 1] for i in range(n)]

        second_diag_0 = all([d == 0 for d in second_diag])
        second_diag_1 = all([d == 1 for d in second_diag])

        return (
            verts_0,
            horizontals_0,
            first_diag_0 or second_diag_0,
            verts_1,
            horizontals_1,
            first_diag_1 or second_diag_1,
        )

    @staticmethod
    def is_game_over(board) -> bool:
        """
        Check if the board state represents a game that is over. This should
        just return a boolean True/False.
        """
        result = TicTacToe.result(board, 0)

        if result is None:
            return False
        # implicit else
        return True

    @staticmethod
    def result(board, player) -> Union[None, int]:
        """
        Get the result of the game. This should be 1 for a win for the player
        whose perspective we're looking for, 0 for a draw, -1 for a loss, or
        None for undetermined, when the game is not over yet.
        """

        status = TicTacToe.get_game_status(board)
        zero_win = any(status[:3])
        one_win = any(status[3:])

        if zero_win:
            result = 1 if player == 0 else -1
        elif one_win:
            result = 1 if player == 1 else -1
        elif torch.any(board.eq(-1)):
            return None
        else:
            result = 0

        return result

    @property
    def width(self) -> int:
        """
        Get the width of the game board. For tic-tac-toe, for instance, this
        should return 3; for chess, 8.
        """
        return self._board.shape[0]

    @property
    def height(self) -> int:
        """
        Get the height of the game board. For tic-tac-toe, this should return
        3; for chess, 8.
        """
        return self._board.shape[1]

    @property
    def board_state(self) -> torch.Tensor:
        """
        Get the current board state as a torch.Tensor.
        """
        return self._board

    @classmethod
    def from_json(cls, **params):
        """
        Create an instance of class `cls` starting from a JSON style dict
        `params`. This dict should contain all the width, height, board state
        parameters that may be needed to create an instance of this game class.
        """
        return cls(**params)

    def reset(self) -> None:
        """
        Reset the game to the initial board position (as given by
        `get_initial_board()`). Reset move lists, move counters, anything else.
        """
        self._board = self.get_initial_board()
        self._move_count = 0
        assert self.is_valid()

    def is_valid(self) -> bool:
        """
        Check if the current board position is valid.
        """
        valid = True

        zeroes = (self._board == 0).sum().item()
        ones = (self._board == 1).sum().item()

        if zeroes + ones != self._move_count:
            valid = False

        if abs(zeroes - ones) > 1:
            valid = False

        return valid

    @staticmethod
    def mirror_h(board):
        """
        Mirror the board position horizontally. Should be implemented per
        subclass.
        """
        return board.flip(1)

    @staticmethod
    def mirror_v(board):
        """
        Mirror the board position vertically. Should be implemented per
        subclass.

        For some games, like Connect 4, this may not be a valid transformation.
        This may require handling downstream in whatever uses this function.
        """
        return board.flip(0)

    @staticmethod
    def n_players():
        """
        Get the number of players in the game.
        """
        return 2


class Connect4(Game):
    def __init__(self, width=7, height=6, board_state=None):
        """
        Construct an instance of the game class.
        """
        if board_state is None:
            self._board = self.get_initial_board(height, width)
        else:
            tensor = torch.Tensor(board_state)
            if width * height != tensor.numel():
                raise ValueError(
                    f"width and height dimensions {width}, {height} do not "
                    f"match board state size {tensor.shape}"
                )
            self._board = tensor.view(height, width)
        self._move_count = 0
        self._players = [0, 1]

    def make_move(self, move: int):
        """
        Make a move on the board. Should be a valid move and return a valid
        game board.
        """
        self._board = self.board_after_move(
            self._board, self._players[self._move_count % 2], move
        )
        self._move_count += 1

        if not self.is_valid():
            raise ValueError(f"Board state {self._board} is not valid.")

    @staticmethod
    def board_after_move(board, player: int, move: int):
        """
        Make a move on the board and return it. Does not affect any instances
        of the class. Should be a valid move and return a valid game board.
        """
        if not isinstance(move, int):
            raise TypeError(f"move is of type {type(move)} but must be an int")

        if player not in (0, 1):
            raise ValueError(f"invalid player for move {move}")

        if not any((board[:, move] == -1)):
            raise ValueError(f"Column is full")
        if move > board.shape[0]:
            raise IndexError(
                f"invalid move {move} for board shaped {board.shape}"
            )

        row_index = board.shape[0] - 1
        while row_index >= 0 and board[row_index, move] != -1:
            row_index -= 1

        new_board = board.clone()
        new_board[row_index, move] = player

        return new_board

    @staticmethod
    def get_initial_board(height=6, width=7):
        """
        Get the initial game board for this game. For Connect 4, for instance,
        this should be an empty board of width `width` and height `height`, but
        for chess this may be the initial position of the pieces.
        """
        return torch.full((height, width), -1)

    @staticmethod
    def get_legal_moves(board) -> torch.Tensor:
        """
        Get a list of legal moves for the given board game position. Should
        return a `torch.Tensor` full of booleans that represent whether a move
        is valid or not. Does not affect internal state of any instances of
        this class.
        """
        return board[0, :] == -1

    def current_legal_moves(self) -> torch.Tensor:
        """
        Get a dynamic list of legal moves for the current game position. Should
        return a `torch.Tensor` full of booleans that represent whether a move
        is valid or not.
        """
        return self.get_legal_moves(self._board)

    @staticmethod
    def get_game_status(board) -> tuple:
        """
        Checks to see if any player has reached Victory. Stores all the
        columns, rows, and diagonals of the tensor in a list of lists,
        and then checks every subarray to see if there are 4 pieces in a
        row. Works for all rectangular boards
        """
        possible_rows = []
        m = board.shape[0]
        n = board.shape[1]

        if n > m:
            board = board.transpose(-1, 0)
            m = board.shape[0]
            n = board.shape[1]

        print(board)
        for i in range(m):
            possible_rows.append(board[i, :].tolist())

        for i in range(n):
            possible_rows.append(board[:, i].tolist())

        # h_index starts incrementing once you reach the base
        h_index = 0
        for i in range(2 * min(m, n) - 7 + abs(m - n)):
            if 4 + i < m:
                possible_rows.append(
                    [board[3 + i - j, j].item() for j in range(4 + i)]
                )
                possible_rows.append(
                    [board[m - 4 - i + j, j].item() for j in range(4 + i)]
                )
            elif (4 + i) >= m and 4 + i <= n:
                possible_rows.append(
                    [board[m - 1 - j, j + h_index].item() for j in range(m)]
                )
                possible_rows.append(
                    [board[j, j + h_index].item() for j in range(m)]
                )
                h_index += 1
            elif 4 + i > n:
                possible_rows.append(
                    [
                        board[m - 1 - j, j + h_index].item()
                        for j in range(max(4, n - h_index))
                    ]
                )
                possible_rows.append(
                    [
                        board[j, j + h_index].item()
                        for j in range(max(4, n - h_index))
                    ]
                )
                h_index += 1
        if m > n:
            board = board.transpose(-1, 0)

        check_0_win = []
        check_1_win = []

        for i in possible_rows:
            temp_0 = [j == 0 for j in i]
            temp_1 = [m == 1 for m in i]

            for k in range(len(i) - 3):
                # check every subarray of length 4 to if there are 4 connected
                check_0_win.append(sum(temp_0[k : k + 4]))
                check_1_win.append(sum(temp_1[k : k + 4]))

        return (4 in check_0_win, 4 in check_1_win)

    @staticmethod
    def is_game_over(board) -> bool:
        """
        Check if the board state represents a game that is over. This should
        just return a boolean True/False.
        """
        result = Connect4.result(board)
        if result is None:
            return False
        # implicit else
        return True

    @staticmethod
    def result(board) -> Union[None, int]:
        """
        Get the result of the game. This should be 1 for a win for the player
        whose perspective we're looking for, 0 for a draw, -1 for a loss, or
        None for undetermined, when the game is not over yet.
        """
        status = Connect4.get_game_status(board)

        zero_win = status[0]
        one_win = status[1]

        if zero_win:
            result = 1
        elif one_win:
            result = -1
        elif torch.any(board.eq(-1)):
            return None
        else:
            result = 0

        return result

    @property
    def width(self) -> int:
        """
        Get the width of the game board. For tic-tac-toe, for instance, this
        should return 3; for chess, 8.
        """
        return self._board.shape[1]

    @property
    def height(self) -> int:
        """
        Get the height of the game board. For tic-tac-toe, this should return
        3; for chess, 8.
        """
        return self._board.shape[0]

    @property
    def board_state(self) -> torch.Tensor:
        """
        Get the current board state as a torch.Tensor.
        """
        return self._board

    @classmethod
    def from_json(cls, **params):
        """
        Create an instance of class `cls` starting from a JSON style dict
        `params`. This dict should contain all the width, height, board state
        parameters that may be needed to create an instance of this game class.
        """
        return cls(**params)

    def reset(self) -> None:
        """
        Reset the game to the initial board position (as given by
        `get_initial_board()`). Reset move lists, move counters, anything else.
        """
        self._board = self.get_initial_board()
        self._move_count = 0
        assert self.is_valid()

    def is_valid(self) -> bool:
        """
        Check if the current board position is valid.
        """
        valid = True

        zeroes = (self._board == 0).sum().item()
        ones = (self._board == 1).sum().item()

        if zeroes + ones != self._move_count:
            valid = False

        if abs(zeroes - ones) > 1:
            valid = False

        return valid

    @staticmethod
    def mirror_h(board):
        """
        Mirror the board position horizontally. Should be implemented per
        subclass.
        """
        return board.flip(1)

    @staticmethod
    def mirror_v(board):
        """
        Mirror the board vertically. Not implemented for Connect4.
        """
        raise NotImplementedError("Not Valid Operation for Connect4")

    @staticmethod
    def n_players():
        """
        Get the number of players in the game.
        """
        return 2
