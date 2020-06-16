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
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels) if batch_on else None
        self.bn2 = nn.BatchNorm2d(out_channels) if batch_on else None
        self.activation = activation

    def forward(self, x):
        """
        Create One Residual Block where we save the input
        as the residual and add it after two convolutions
        """
        residual = x
        # first convolution
        out = self.conv1(x)
        if self.bn1 is not None:
            out = self.bn1(out)
        out = self.activation(out)

        # second convolution, add input after batchnorm step
        out = self.conv2(out)
        if self.bn2 is not None:
            out = self.bn2(out)
        out += residual
        out = self.activation(out)

        return out


class SidNet(nn.Module):
    """
    Creates the Encoding Neural Network. Takes
            block: type of blocks to construct.(ResidualBlocks)
            in_channels: # of game states(previous turns and positions)
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
        self.conv1 = nn.Conv2d(
            in_channels, enc_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn = nn.BatchNorm2d(enc_channels) if batch_on else None
        self.resblock = self.make_layer(block, enc_channels, num_blocks)
        self.activation = activation

    def make_layer(self, block, out_channels, num_blocks):

        layers = []
        for i in range(0, num_blocks):
            layers.append(block(out_channels, out_channels))
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
    Create the PolicyHead Network, which take the encoded features and the game
    being played from the previous SidNet Network and returns a vector of
    length game.width * game.height of move probabilities for each specific
    position. Following the architecture of the AlphaGoZero Paper, requires the
    number of channels from the previous residual nets, the game being played,
    the activation function, and if bathcnorm is on or off.
    """

    def __init__(
        self,
        enc_channels: int,
        game,
        activation=nn.ReLU(inplace=False),
        batch_on=True,
        pass_move=True,
    ):

        super(PolicyHead, self).__init__()
        self.policyconv = nn.Conv2d(
            enc_channels, out_channels=2, kernel_size=1, stride=1
        )
        self.policybn = nn.BatchNorm2d(2) if batch_on else None
        self.poss_moves = game.width * game.height

        if pass_move:
            self.policyfc = nn.Linear(
                in_features=self.poss_moves * 2,
                out_features=self.poss_moves + 1,
            )
        else:
            self.policyfc = nn.Linear(
                in_features=self.poss_moves * 2, out_features=self.poss_moves
            )
        self.activation = activation
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, encoding):
        policy = self.policyconv(encoding)
        if self.policybn is not None:
            policy = self.policybn(policy)
        policy = self.activation(policy)
        policy = policy.view(-1, self.poss_moves * 2)
        policy = self.policyfc(policy)
        probas = self.logsoftmax(policy).exp()
        return probas


class ValueHead(nn.Module):
    """
    Create the ValueHead Network, which take the encoded features from the
    SidNet Network and returns a value between -1 and 1 predicting the
    probability of winning. Requres number of encoded features, the game being
    played, the activation function, and if bathcnorm is on or off
    """

    def __init__(
        self,
        enc_channels: int,
        game,
        activation=nn.ReLU(inplace=False),
        batch_on=True,
    ):

        super(ValueHead, self).__init__()
        self.valueconv = nn.Conv2d(
            enc_channels, out_channels=1, kernel_size=1, stride=1
        )
        self.valuebn = nn.BatchNorm2d(1) if batch_on else None
        self.game_area = game.width * game.height
        # figuring out what the in_channel should be for this
        self.valuefc1 = nn.Linear(in_features=self.game_area, out_features=256)
        self.valuefc2 = nn.Linear(in_features=256, out_features=1)
        self.activation = activation

        self.tanh = nn.Tanh()

    def forward(self, encoding):
        value = self.valueconv(encoding)
        if self.valuebn is not None:
            value = self.valuebn(value)
        value = self.activation(value)
        value = value.view(-1, self.game_area)
        value = self.valuefc1(value)
        value = self.activation(value)
        value = self.valuefc2(value)
        win_value = self.tanh(value)
        return win_value
