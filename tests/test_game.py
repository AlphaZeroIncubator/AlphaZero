#! /usr/bin/python3

import pytest
from alphazero import TicTacToe, Connect4
import torch


class Test_TicTacToe:
    def setup_class(self):
        self.ttt = TicTacToe()
        self.dummy_tensor = torch.Tensor(
            [[0, -1, -1], [-1, -1, -1], [-1, -1, -1]]
        )

    def test_dims_default(self):
        ttt = self.ttt

        assert ttt.board_state.shape == (3, 3)

    def test_move_count_default(self):
        assert self.ttt._move_count == 0

    def test_board_state_default(self):
        assert torch.all(self.ttt.board_state.eq(-1))

    def test_dims_custom(self):
        ttt = TicTacToe(width=4, height=5)

        assert ttt.board_state.shape == (4, 5)
        assert ttt.width == 4
        assert ttt.height == 5

    def test_board_state_custom(self):
        board = self.dummy_tensor
        ttt = TicTacToe(board_state=board)

        assert torch.all(ttt.board_state.eq(board))

    def test_make_move(self):
        self.ttt.make_move((0, 0))

        result = self.dummy_tensor

        assert torch.all(self.ttt.board_state.eq(result))

        with pytest.raises(TypeError):
            self.ttt.make_move(1)

        with pytest.raises(TypeError):
            self.ttt.make_move([1, 1])

        with pytest.raises(ValueError):
            self.ttt.make_move((1, 1, 1))

        with pytest.raises(IndexError):
            self.ttt.make_move((3, 1))

        self.ttt.reset()

    def test_board_after_move(self):
        initial = TicTacToe.get_initial_board()
        board = TicTacToe.board_after_move(initial, 0, (0, 0))

        assert torch.all(board.eq(self.dummy_tensor))

        board = TicTacToe.board_after_move(board, 1, (0, 1))

        next_state = torch.Tensor([[0, 1, -1], [-1, -1, -1], [-1, -1, -1]])

        assert torch.all(board.eq(next_state))

        with pytest.raises(TypeError):
            TicTacToe.board_after_move(initial, 0, 1)

        with pytest.raises(TypeError):
            TicTacToe.board_after_move(initial, 0, [1, 1])

        with pytest.raises(ValueError):
            TicTacToe.board_after_move(initial, 0, (1, 1, 1))

        with pytest.raises(ValueError):
            TicTacToe.board_after_move(initial, 2, (1, 1))

        with pytest.raises(IndexError):
            TicTacToe.board_after_move(initial, 0, (3, 1))

    def test_get_initial_board(self):
        board = TicTacToe.get_initial_board()
        assert torch.all(board.eq(-1))
        assert board.shape == (3, 3)

    def test_1(self):
        legal_moves = TicTacToe.get_legal_moves(self.dummy_tensor)

        assert torch.all(legal_moves.eq(self.dummy_tensor == -1))

        legal_moves = TicTacToe.get_legal_moves(TicTacToe.get_initial_board())

        assert torch.all(legal_moves.eq(True))

    def test_current_legal_moves(self):
        self.ttt.make_move((0, 0))

        self.ttt.current_legal_moves()
        self.ttt.reset()

    def test_get_game_status(self):
        board = torch.Tensor([[-0, -1, -1], [-1, -1, -1], [-1, -1, -1]])

        status = TicTacToe.get_game_status(board)

        assert status == (False, False, False, False, False, False)

        board = torch.Tensor([[0, -1, -1], [0, -1, -1], [0, -1, -1]])

        status = TicTacToe.get_game_status(board)

        assert status == (True, False, False, False, False, False)

        board = torch.Tensor([[0, 0, 0], [-1, -1, -1], [-1, -1, -1]])

        status = TicTacToe.get_game_status(board)

        assert status == (False, True, False, False, False, False)

        board = torch.Tensor([[0, -1, -1], [-1, 0, -1], [-1, -1, 0]])

        status = TicTacToe.get_game_status(board)

        assert status == (False, False, True, False, False, False)

        board = torch.Tensor([[-1, -1, 0], [-1, 0, -1], [0, -1, -1]])

        status = TicTacToe.get_game_status(board)

        assert status == (False, False, True, False, False, False)

        board = torch.Tensor([[1, -1, -1], [1, -1, -1], [1, -1, -1]])

        status = TicTacToe.get_game_status(board)

        assert status == (False, False, False, True, False, False)

        board = torch.Tensor([[1, 1, 1], [-1, -1, -1], [-1, -1, -1]])

        status = TicTacToe.get_game_status(board)

        assert status == (False, False, False, False, True, False)

        board = torch.Tensor([[1, -1, -1], [-1, 1, -1], [-1, -1, 1]])

        status = TicTacToe.get_game_status(board)

        assert status == (False, False, False, False, False, True)

        board = torch.Tensor([[-1, -1, 1], [-1, 1, -1], [1, -1, -1]])

        status = TicTacToe.get_game_status(board)

        assert status == (False, False, False, False, False, True)

    def test_is_game_over(self):
        board = torch.Tensor([[0, -1, -1], [0, -1, -1], [0, -1, -1]])

        assert TicTacToe.is_game_over(self.dummy_tensor) is False
        assert TicTacToe.is_game_over(board) is True

    def test_result(self):
        board = torch.Tensor([[0, -1, -1], [0, -1, -1], [0, -1, -1]])

        board_2 = torch.Tensor([[1, -1, -1], [1, -1, -1], [1, -1, -1]])

        draw = torch.Tensor([[0, 1, 0], [1, 0, 0], [1, 0, 1]])

        assert TicTacToe.result(self.dummy_tensor, 0) is None
        assert TicTacToe.result(board, 0) == 1
        assert TicTacToe.result(board_2, 0) == -1
        assert TicTacToe.result(draw, 0) == 0

    def test_width(self):
        assert self.ttt.width == 3

    def test_height(self):
        assert self.ttt.height == 3

    def test_board_state(self):
        assert torch.all(self.ttt.board_state.eq(-1))

    def test_from_json(self):
        board = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]

        json_dict = {"height": 4, "width": 5, "board_state": board}

        ttt = TicTacToe(width=5, height=4, board_state=board)

        ttt_json = TicTacToe.from_json(**json_dict)

        assert torch.all(ttt.board_state.eq(ttt_json.board_state))

    def test_reset_board(self):
        self.ttt.make_move((0, 0))
        self.ttt.reset()
        assert torch.all(self.ttt.board_state.eq(-1))
        assert self.ttt._move_count == 0

    def test_is_valid(self):
        self.ttt._board = torch.full((3, 3), -2)
        self.ttt._board[0] = 1

        assert self.ttt.is_valid() is False
        self.ttt.reset()

    def test_mirror_h(self):
        board = torch.Tensor([[0, 1, 0], [1, 0, 0], [1, 0, 1]])

        mirrored = torch.Tensor([[0, 1, 0], [0, 0, 1], [1, 0, 1]])

        assert torch.all(TicTacToe.mirror_h(board).eq(mirrored))

    def test_mirror_v(self):
        # how is this more readable than having the three lists in
        # separate lines? dang it, black
        board = torch.Tensor([[0, 1, 0], [1, 0, 0], [1, 0, 1]])

        mirrored = torch.Tensor([[1, 0, 1], [1, 0, 0], [0, 1, 0]])

        assert torch.all(TicTacToe.mirror_v(board).eq(mirrored))

    def test_make_move_same_spot(self):
        self.ttt.make_move((0, 0))

        with pytest.raises(ValueError):
            self.ttt.make_move((0, 0))

    def test_board_state_list_of_lists_accepted(self):
        board = [
            [-1, -1, -1],
            [-1, -1, -1],
            [-1, -1, -1],
        ]

        ttt = TicTacToe(board_state=board)

        assert ttt.width == 3
        assert ttt.height == 3
        assert torch.all(ttt.board_state.eq(-1))

    def test_board_state_list_accepted(self):
        board = [
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
        ]

        ttt = TicTacToe(board_state=board)

        assert ttt.width == 3
        assert ttt.height == 3
        assert torch.all(ttt.board_state.eq(-1))

    def test_json_accepts_any_order_of_params(self):
        board = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]

        json_dict = {"board_state": board, "height": 4, "width": 5}

        ttt = TicTacToe(width=5, height=4, board_state=board)

        ttt_json = TicTacToe.from_json(**json_dict)

        assert torch.all(ttt.board_state.eq(ttt_json.board_state))

    def test_tie_board_is_game_over(self):
        board = torch.Tensor([[1, 0, 1], [1, 0, 1], [0, 1, 0]])

        assert TicTacToe.is_game_over(board) is True

    def test_n_players(self):
        assert TicTacToe.n_players() == 2


class Test_Connect4:
    def setup_class(self):
        self.C4 = Connect4()
        self.dummy_tensor = torch.Tensor(
            [
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [0, -1, -1, -1, -1, -1, -1],
            ]
        )

    def test_dims_default(self):
        C4 = self.C4

        assert C4.board_state.shape == (6, 7)

    def test_move_count_default(self):
        assert self.C4._move_count == 0

    def test_board_state_default(self):
        assert torch.all(self.C4.board_state.eq(-1))

    def test_dims_custom(self):
        C4 = Connect4(width=8, height=7)

        assert C4.board_state.shape == (7, 8)
        assert C4.width == 8
        assert C4.height == 7

    def test_board_state_custom(self):
        board = self.dummy_tensor
        C4 = Connect4(board_state=board)

        assert torch.all(C4.board_state.eq(board))

    def test_make_move(self):
        self.C4.make_move(0)

        result = self.dummy_tensor

        assert torch.all(self.C4.board_state.eq(result))

        with pytest.raises(TypeError):
            self.C4.make_move([3, 3])

        with pytest.raises(TypeError):
            self.C4.make_move((1, 1))

        with pytest.raises(IndexError):
            self.C4.make_move(10)

        self.C4.reset()

    def test_board_after_move(self):
        initial = Connect4.get_initial_board()
        board = Connect4.board_after_move(initial, 0, 0)

        assert torch.all(board.eq(self.dummy_tensor))

        board = Connect4.board_after_move(board, 1, 1)

        next_state = torch.Tensor(
            [
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [0, 1, -1, -1, -1, -1, -1],
            ]
        )

        final_state = torch.Tensor(
            [
                [0, -1, -1, -1, -1, -1, -1],
                [1, -1, -1, -1, -1, -1, -1],
                [0, -1, -1, -1, -1, -1, -1],
                [1, -1, -1, -1, -1, -1, -1],
                [-0, -1, -1, -1, -1, -1, -1],
                [1, -1, -1, -1, -1, -1, -1],
            ]
        )

        assert torch.all(board.eq(next_state))

        with pytest.raises(TypeError):
            Connect4.board_after_move(initial, 0, (1, 1))

        with pytest.raises(TypeError):
            Connect4.board_after_move(initial, 0, [1, 1])

        with pytest.raises(ValueError):
            Connect4.board_after_move(final_state, 0, 0)

        with pytest.raises(ValueError):
            Connect4.board_after_move(initial, 2, 1)

        with pytest.raises(IndexError):
            Connect4.board_after_move(initial, 0, 10)

    def test_get_initial_board(self):
        board = Connect4.get_initial_board()
        assert torch.all(board.eq(-1))
        assert board.shape == (6, 7)

    def test_1(self):
        legal_moves = Connect4.get_legal_moves(self.dummy_tensor)

        assert torch.all(legal_moves.eq(self.dummy_tensor[0, :] == -1))

        legal_moves = Connect4.get_legal_moves(Connect4.get_initial_board())

        assert torch.all(legal_moves.eq(True))

    def test_current_legal_moves(self):
        self.C4.make_move(0)

        self.C4.current_legal_moves()

        self.C4.reset()

    def test_get_game_status(self):
        board = torch.Tensor(
            [
                [0, 0, 1, 0, 1, 0, 0],
                [1, 1, 0, 0, 1, 1, 0],
                [1, 1, 0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0, 1, 1],
                [1, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 1, 1, 1, 0],
            ]
        )

        status = Connect4.get_game_status(board)

        assert status == (False, False)

        board = torch.Tensor(
            [
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [0, -1, -1, -1, -1, -1, -1],
                [0, -1, -1, -1, -1, -1, -1],
                [0, -1, -1, -1, -1, -1, -1],
                [0, -1, -1, -1, -1, -1, -1],
            ]
        )

        status = Connect4.get_game_status(board)

        assert status == (True, False)

        board = torch.Tensor(
            [
                [-1, -1, -1, -1, -1, -1, 0],
                [-1, -1, -1, -1, -1, -1, 0],
                [-1, -1, -1, -1, -1, -1, 0],
                [-1, -1, -1, -1, -1, -1, 0],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
            ]
        )

        status = Connect4.get_game_status(board)

        assert status == (True, False)

        board = torch.Tensor(
            [
                [0, 0, 0, 0, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
            ]
        )

        status = Connect4.get_game_status(board)

        assert status == (True, False)

        board = torch.Tensor(
            [
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, 0, 0, 0, 0],
            ]
        )

        status = Connect4.get_game_status(board)

        assert status == (True, False)

        board = torch.Tensor(
            [
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [0, -1, -1, -1, -1, -1, -1],
                [-1, 0, -1, -1, -1, -1, -1],
                [-1, -1, 0, -1, -1, -1, -1],
                [-1, -1, -1, 0, -1, -1, -1],
            ]
        )

        status = Connect4.get_game_status(board)

        assert status == (True, False)

        board = torch.Tensor(
            [
                [-1, -1, -1, 0, -1, -1, -1],
                [-1, -1, -1, -1, 0, -1, -1],
                [-1, -1, -1, -1, -1, 0, -1],
                [-1, -1, -1, -1, -1, -1, 0],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
            ]
        )

        status = Connect4.get_game_status(board)

        assert status == (True, False)

        board = torch.Tensor(
            [
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, 0, -1, -1, -1, -1, -1],
                [-1, -1, 0, -1, -1, -1, -1],
                [-1, -1, -1, 0, -1, -1, -1],
                [-1, -1, -1, -1, 0, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
            ]
        )

        status = Connect4.get_game_status(board)
        assert status == (True, False)

        board = torch.Tensor(
            [
                [-1, -1, -1, 0, -1, -1, -1],
                [-1, -1, 0, -1, -1, -1, -1],
                [-1, 0, -1, -1, -1, -1, -1],
                [0, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
            ]
        )

        status = Connect4.get_game_status(board)

        assert status == (True, False)

        board = torch.Tensor(
            [
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, 0],
                [-1, -1, -1, -1, -1, 0, -1],
                [-1, -1, -1, -1, 0, -1, -1],
                [-1, -1, -1, 0, -1, -1, -1],
            ]
        )
        status = Connect4.get_game_status(board)

        assert status == (True, False)

        board = torch.Tensor(
            [
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, 0, -1, -1],
                [-1, -1, -1, 0, -1, -1, -1],
                [-1, -1, 0, -1, -1, -1, -1],
                [-1, 0, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
            ]
        )
        status = Connect4.get_game_status(board)

        board = torch.Tensor(
            [
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [1, -1, -1, -1, -1, -1, -1],
                [1, -1, -1, -1, -1, -1, -1],
                [1, -1, -1, -1, -1, -1, -1],
                [1, -1, -1, -1, -1, -1, -1],
            ]
        )

        status = Connect4.get_game_status(board)

        assert status == (False, True)

        board = torch.Tensor(
            [
                [-1, -1, -1, -1, -1, -1, 1],
                [-1, -1, -1, -1, -1, -1, 1],
                [-1, -1, -1, -1, -1, -1, 1],
                [-1, -1, -1, -1, -1, -1, 1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
            ]
        )

        status = Connect4.get_game_status(board)

        assert status == (False, True)

        board = torch.Tensor(
            [
                [1, 1, 1, 1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
            ]
        )

        status = Connect4.get_game_status(board)

        assert status == (False, True)

        board = torch.Tensor(
            [
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, 1, 1, 1, 1],
            ]
        )

        status = Connect4.get_game_status(board)

        assert status == (False, True)

        board = torch.Tensor(
            [
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [1, -1, -1, -1, -1, -1, -1],
                [-1, 1, -1, -1, -1, -1, -1],
                [-1, -1, 1, -1, -1, -1, -1],
                [-1, -1, -1, 1, -1, -1, -1],
            ]
        )

        status = Connect4.get_game_status(board)

        assert status == (False, True)

        board = torch.Tensor(
            [
                [-1, -1, -1, 1, -1, -1, -1],
                [-1, -1, -1, -1, 1, -1, -1],
                [-1, -1, -1, -1, -1, 1, -1],
                [-1, -1, -1, -1, -1, -1, 1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
            ]
        )

        status = Connect4.get_game_status(board)

        assert status == (False, True)

        board = torch.Tensor(
            [
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, 1, -1, -1, -1, -1, -1],
                [-1, -1, 1, -1, -1, -1, -1],
                [-1, -1, -1, 1, -1, -1, -1],
                [-1, -1, -1, -1, 1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
            ]
        )

        status = Connect4.get_game_status(board)
        assert status == (False, True)

        board = torch.Tensor(
            [
                [-1, -1, -1, 1, -1, -1, -1],
                [-1, -1, 1, -1, -1, -1, -1],
                [-1, 1, -1, -1, -1, -1, -1],
                [1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
            ]
        )

        status = Connect4.get_game_status(board)

        assert status == (False, True)

        board = torch.Tensor(
            [
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, 1],
                [-1, -1, -1, -1, -1, 1, -1],
                [-1, -1, -1, -1, 1, -1, -1],
                [-1, -1, -1, 1, -1, -1, -1],
            ]
        )

        status = Connect4.get_game_status(board)

        assert status == (False, True)

        board = torch.Tensor(
            [
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, 1, -1, -1],
                [-1, -1, -1, 1, -1, -1, -1],
                [-1, -1, 1, -1, -1, -1, -1],
                [-1, 1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
            ]
        )

        status = Connect4.get_game_status(board)
        assert status == (False, True)

        status = Connect4.get_game_status(board)
        board = torch.Tensor(
            [
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [1, 0, -1, -1, -1, -1, -1],
                [1, 0, -1, -1, -1, -1, -1],
                [1, 0, -1, -1, -1, -1, -1],
                [1, 0, -1, -1, -1, -1, -1],
            ]
        )

        status = Connect4.get_game_status(board)
        assert status == (True, True)

    def test_is_game_over(self):
        board = torch.Tensor(
            [
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, 1, -1, -1],
                [-1, -1, -1, 1, -1, -1, -1],
                [-1, -1, 1, -1, -1, -1, -1],
                [-1, 1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
            ]
        )

        assert Connect4.is_game_over(self.dummy_tensor) is False
        assert Connect4.is_game_over(board) is True

    def test_result(self):
        board = torch.Tensor(
            [
                [-1, -1, -1, 0, -1, -1, -1],
                [-1, -1, 0, -1, -1, -1, -1],
                [-1, 0, -1, -1, -1, -1, -1],
                [0, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
            ]
        )

        board_2 = torch.Tensor(
            [
                [-1, -1, -1, 1, -1, -1, -1],
                [-1, -1, 1, -1, -1, -1, -1],
                [-1, 1, -1, -1, -1, -1, -1],
                [1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
            ]
        )

        draw = torch.Tensor(
            [
                [0, 0, 1, 0, 1, 0, 0],
                [1, 1, 0, 0, 1, 1, 0],
                [1, 1, 0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0, 1, 1],
                [1, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 1, 1, 1, 0],
            ]
        )

        assert Connect4.result(self.dummy_tensor) is None
        assert Connect4.result(board) == 1
        assert Connect4.result(board_2) == -1
        assert Connect4.result(draw) == 0

    def test_width(self):
        assert self.C4.width == 7

    def test_height(self):
        assert self.C4.height == 6

    def test_board_state(self):
        assert torch.all(self.C4.board_state.eq(-1))

    def test_from_json(self):
        board = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]

        json_dict = {"height": 5, "width": 4, "board_state": board}

        C4 = Connect4(width=4, height=5, board_state=board)

        C4_json = Connect4.from_json(**json_dict)

        assert torch.all(C4.board_state.eq(C4_json.board_state))

    def test_reset_board(self):
        self.C4.make_move(0)
        self.C4.reset()
        assert torch.all(self.C4.board_state.eq(-1))
        assert self.C4._move_count == 0

    def test_is_valid(self):
        self.C4._board = torch.full((6, 7), -2)
        self.C4._board[0] = 1

        assert self.C4.is_valid() is False
        self.C4.reset()

    def test_mirror_h(self):
        board = torch.Tensor(
            [
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [0, -1, -1, -1, -1, -1, -1],
                [0, -1, -1, -1, -1, -1, -1],
                [0, -1, -1, -1, -1, -1, -1],
                [0, -1, -1, -1, -1, -1, -1],
            ]
        )

        mirrored = torch.Tensor(
            [
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, 0],
                [-1, -1, -1, -1, -1, -1, 0],
                [-1, -1, -1, -1, -1, -1, 0],
                [-1, -1, -1, -1, -1, -1, -0],
            ]
        )

        assert torch.all(Connect4.mirror_h(board).eq(mirrored))

    def test_board_state_list_of_lists_accepted(self):
        board = torch.Tensor(
            [
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1],
            ]
        )

        C4 = Connect4(board_state=board)

        assert C4.width == 7
        assert C4.height == 6
        assert torch.all(C4.board_state.eq(-1))

    def test_board_state_list_accepted(self):
        board = [
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
        ]

        C4 = Connect4(board_state=board)

        assert C4.width == 7
        assert C4.height == 6
        assert torch.all(C4.board_state.eq(-1))

    def test_json_accepts_any_order_of_params(self):
        board = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]

        json_dict = {"board_state": board, "height": 5, "width": 4}

        C4 = Connect4(width=4, height=5, board_state=board)

        C4_json = Connect4.from_json(**json_dict)

        assert torch.all(C4.board_state.eq(C4_json.board_state))

    def test_n_players(self):
        assert Connect4.n_players() == 2
