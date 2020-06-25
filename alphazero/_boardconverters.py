import torch


class BoardConverter:
    @staticmethod
    def board_to_tensor(board):
        raise NotImplementedError


class TicTacToeConverter(BoardConverter):
    @staticmethod
    def board_to_tensor(board: torch.Tensor, current_player: int):
        player_1 = board == 0
        player_2 = board == 1
        player_layer = torch.full(
            board.size(), current_player, dtype=torch.float
        )
        print(torch.stack((player_1.float(), player_2.float(), player_layer)))
        return torch.stack((player_1.float(), player_2.float(), player_layer))
