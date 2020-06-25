import torch
from alphazero import TicTacToeConverter


class TestBoardConverters:
    def test_tictactoe_converter(self):
        # Check if output of standard tictactoe boardconverter is OK
        board = torch.Tensor([[-1, 0, 1], [1, -1, -1], [0, 0, 1]])
        player = 0
        out = TicTacToeConverter.board_to_tensor(board, player)

        assert torch.allclose(
            out,
            torch.FloatTensor(
                [
                    [[0, 1, 0], [0, 0, 0], [1, 1, 0]],
                    [[0, 0, 1], [1, 0, 0], [0, 0, 1]],
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                ]
            ),
        )
