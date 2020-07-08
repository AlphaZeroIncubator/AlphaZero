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
        print(torch.stack((player_1.float(), player_2.float(), player_layer)))
        return torch.stack(
            (player_1.float(), player_2.float(), player_layer)
        ).unsqueeze(0)



class LeoConverter(BoardConverter):
    @staticmethod
    def board_to_tensor(
        board: torch.Tensor, current_player: int
    ) -> torch.Tensor:
        player_1: torch.Tensor = board == current_player
        player_2: torch.Tensor = board == 1 - current_player
        board_tensor = torch.stack((player_1.float(), player_2.float()))
        board_tensor.unsqueeze_(0)
        return board_tensor
