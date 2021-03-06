import torch


class BoardConverter:
    @staticmethod
    def board_to_tensor(
        board: torch.Tensor, current_player: int
    ) -> torch.Tensor:
        """Convert a board to a tensor.

        Converts a game board to a tensor
            that can be input into a model.
        Args:
            board (torch.Tensor): Input board
            current_player (int): The player whose turn it is

        Returns:
            tensor (torch.Tensor): Output tensor
        """
        raise NotImplementedError


class TicTacToeConverter(BoardConverter):
    @staticmethod
    def board_to_tensor(
        board: torch.Tensor, current_player: int
    ) -> torch.Tensor:

        player_1: torch.Tensor = board == 0
        player_2: torch.Tensor = board == 1

        player_layer = torch.full(
            board.size(), current_player, dtype=torch.float
        )

        # return stacked: player_1, player_2, current_player layers
        return torch.stack(
            (player_1.float(), player_2.float(), player_layer)
        ).unsqueeze(0)
