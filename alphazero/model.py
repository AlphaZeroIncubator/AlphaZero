import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation=nn.ReLU(inplace=False),
        batch_on=True,
    ):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1)
        self.bn = nn.BatchNorm2d(out_channels) if batch_on else None
        self.activation = activation

    def forward(self, x):
        """
        Create One Residual Block where we save the input 
        as the residual and add it after two convolutions
        """
        residual = x
        # first convolution
        out = self.conv1(x)
        if self.batchnorm is not None:
            out = self.bn(out)
        out = self.activation(out)

        # second convolution, add input after batchnorm step
        out = self.conv2(out)
        if self.bn is not None:
            out = self.bn(out)
        out += residual
        out = self.activation(out)

        return out


class SidNet(nn.Module):
    """
    Creates the Encoding Neural Network. Takes
            block: type of blocks to construct. In this case we use ResidualBlocks as defined above
            in_channels: 3-D tensor of game.width*game.height*(number of previous turns and player positions)
            enc_channels: number of channels to encode 
            num_blocks: number of residual blocks to create
    """

    def __init__(
        self,
        block,
        in_channels: int,
        enc_channels: int,
        num_blocks: int,
        activation=nn.ReLU(inplace=False),
        batch_on=True,
    ):
        super(SidNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, enc_channels, kernel_size=3, stride=1)
        self.bn = nn.BatchNorm2d(enc_channels) if batch_on else None
        self.resblock = self.make_layer(block, enc_channels, num_blocks)
        self.activation = activation

    def make_layer(self, block, out_channels, num_blocks, stride=1):

        layers = []
        for i in range(0, num_layers):
            layers.append(block(enc_channels, enc_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):

        encoding = self.conv1(x)
        if self.bn is not None:
            encoding = self.bn(encoding)
        encoding = self.activation(encoding)
        encoding = self.resblock(encoding)

        return encoding


class PolicyHead(nn.Module):
    """
    Create the PolicyHead Network, which take the encoded features and the game being played 
    from the previous SidNet Network and returns a vector of length game.width * game.height of 
    move probabilities on each specific position. Following the architecture of the AlphaGoZero Paper,
    only requires the number of channels from the previous residual nets and the game
    being played to obtain board positions and applies the transformations mentioned below.
    """

    def __init__(
        self,
        enc_channels: int,
        game: Game,
        activation=nn.ReLU(inplace=False),
        batch_on=True,
    ):

        super(PolicyHead, self).__init__()
        self.policyconv = nn.Conv2D(
            enc_channels, out_channels=2, kernel_size=2, stride=1
        )
        self.policybn = nn.BatchNorm2d(2) if batch_on else None
        self.policyfc = nn.Linear(
            in_channels=enc_channels, out_channels=game.width * game.height,
        )
        self.activation = activation

    def forward(self, encoding):
        policy = self.policyconv(encoding)
        if self.policybn is not None:
            policy = self.policybn(policy)
        policy = self.activation(policy)
        # a step here I am not completely sure yet
        policy = self.policyfc(policy)
        return policy


class ValueHead(nn.Module):
    """
    Create the ValueHead Network, which take the encoded features from the previous SidNet
    Network and returns a value between -1 and 1 predicting the probability of winning. Following
    the architecture of the AlphaGoZero Paper, only requires the number of channels from the
    previous residual nets and applies the transformations mentioned below.
    """

    def __init__(
        self, enc_channels: int, activation=nn.ReLU(inplace=False), batch_on=True
    ):

        super(ValueHead, self).__init__()
        self.valueconv = nn.Conv2D(
            enc_channels, out_channels=1, kernel_size=1, stride=1
        )
        self.valuebn = nn.BatchNorm2d(1) if batch_on else None

        # figuring out what the in_channel should be for this
        self.valuefc1 = nn.Linear(in_channels=number, out_channels=256)
        self.valuefc2 = nn.Linear(in_channels=256, out_channels=1)
        self.activation = activation

        self.tanh = nn.tanh(inplace=False)

    def forward(self, encoding):
        value = self.valueconv(encoding)
        if self.valuebn is not None:
            value = self.valuebn(value)
        value = self.relu(value)
        # a step here I am not completely sure yet
        value = self.valuefc1(value)
        value = self.relu(value)
        value = self.valuefc2(value)
        win_value = self.tanh(value)
        return win_value
