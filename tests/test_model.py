from alphazero import model
import torch


class test_model:
    def __init__(self, game):
        self.game = game

    def test_model_input(self):
        encoding_net = model.SidNet(
            model.ResidualBlock,
            in_channels=17,
            enc_channels=256,
            num_blocks=19,
            activation=torch.nn.ReLU(inplace=False),
            batch_on=True,
        )
        test_input = torch.zeros(1, 17, self.game.width, self.game.height)
        output = encoding_net(test_input)

        assert list(output.size()) == [
            1,
            256,
            self.game.width,
            self.game.height,
        ]

        assert torch.allclose(
            output, torch.zeros(1, 256, self.game.width, self.game.height)
        )

        print("Enconding Network Test Complete")

        policy = model.PolicyHead(
            enc_channels=256, game=self.game, pass_move=False
        )
        value = model.ValueHead(enc_channels=256, game=self.game)
        probas = policy(output)
        win_value = value(output)

        assert list(probas.size()) == [self.game.width, self.game.height]

        assert list(win_value.size()) == [1, 1]

        print("Policy and Value Network Test Complete")
