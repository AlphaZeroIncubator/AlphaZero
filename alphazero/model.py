import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=true)

        def forward(self, x):
            """
			Create One Residual Block where we save the input as the residual and add it after two convolutions
			"""
            residual = x

            """
			first convolution
			"""
            out = self.conv(x)
            out = self.batch_norm(out)
            out = self.relu(out)

            """
			second convolution, add input after batchnorm step
			"""
            out = self.conv(out)
            out = self.batch_norm(out)
            out += residual
            out = self.relu(out)

            return out


class Net(nn.Module):
    def __init__(
        self,
        block,
        in_channels: int,
        out_channels: int,
        game_play: Game,
        residual_layers: int,
    ):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.resblock = self.make_layer(block, out_channels, residual_layers)
        self.relu = nn.ReLU(inplace=true)

        self.policyconv = nn.Conv2D(
            out_channels, out_channels=2, kernel_size=2, stride=1
        )
        self.policybn = nn.BatchNorm2d(2)
        self.policyfc = nn.Linear(
            in_channels=out_channels,
            out_channels=game_play.width * game_play.length + 1,
        )

        self.valueconv = nn.Conv2D(
            out_channels, out_channels=1, kernel_size=1, stride=1
        )
        self.valuebn = nn.BatchNorm2d(1)

        # figuring out what the in_channel should be for this
        self.valuefc1 = nn.Linear(in_channels=number, out_channels=256)
        self.valuefc2 = nn.Linear(in_channels=256, out_channels=1)

    def make_layer(self, block, out_channels, num_layers, stride=1):

        layers = []
        for i in range(0, num_layers):
            layers.append(block(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)

        out = self.resblock(out)

        policy = self.policyconv(out)
        policy = self.policybn(policy)
        policy = self.relu(policy)
        # a step here I am not completely sure yet
        policy = self.policyfc1(policy)

        value = self.valueconv(out)
        value = self.valuebn(value)
        value = self.relu(value)
        # a step here I am not completely sure yet
        value = self.valuefc1(value)
        value = self.relu(value)
        value = self.policyfc2(value)
        win_value = F.tanh(value)

        return policy, value
