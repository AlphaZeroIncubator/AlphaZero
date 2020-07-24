import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        enc_channels: int,
        activation=nn.ReLU(inplace=False),
        batch_on=True,
    ):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, enc_channels, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels, enc_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn1 = nn.BatchNorm2d(enc_channels) if batch_on else nn.Identity()
        self.bn2 = nn.BatchNorm2d(enc_channels) if batch_on else nn.Identity()
        self.activation = activation

    def forward(self, x):
        """
        Create One Residual Block where we save the input
        as the residual and add it after two convolutions
        """
        residual = x
        # first convolution
        out = self.conv1(x)

        out = self.bn1(out)
        out = self.activation(out)

        # second convolution, add input after batchnorm step
        out = self.conv2(out)

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
        self.bn = nn.BatchNorm2d(enc_channels) if batch_on else nn.Identity()
        self.resblock = self.make_layer(
            block, enc_channels, num_blocks, activation, batch_on
        )
        self.activation = activation

    def make_layer(
        self, block, out_channels, num_blocks, activation, batch_on
    ):

        layers = []
        for i in range(0, num_blocks):
            layers.append(
                block(out_channels, out_channels, activation, batch_on)
            )
        return nn.Sequential(*layers)

    def forward(self, x):

        encoding = self.conv1(x)
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
        pass_move=False,
    ):

        super(PolicyHead, self).__init__()
        self.policyconv = nn.Conv2d(
            enc_channels, out_channels=2, kernel_size=1, stride=1
        )
        self.policybn = nn.BatchNorm2d(2) if batch_on else nn.Identity()

        self.game_width = game.width
        self.game_height = game.height
        self.board_pos = game.width * game.height
        self.final_shape = game.get_legal_moves(game.get_initial_board()).shape
        self.pass_move = pass_move
        self.policyfc = nn.Linear(
            in_features=self.board_pos * 2,
            out_features=self.final_shape.numel() + self.pass_move,
        )

        self.activation = activation
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, encoding):
        policy = self.policyconv(encoding)
        policy = self.policybn(policy)
        policy = self.activation(policy)
        policy = policy.view(-1, self.board_pos * 2)
        policy = self.policyfc(policy)
        probas = self.logsoftmax(policy).exp()
        if self.pass_move:
            return probas
        else:
            return probas.view(self.final_shape)


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
        self.valuebn = nn.BatchNorm2d(1) if batch_on else nn.Identity()
        self.board_pos = game.width * game.height
        # figuring out what the in_channel should be for this
        self.valuefc1 = nn.Linear(in_features=self.board_pos, out_features=256)
        self.valuefc2 = nn.Linear(in_features=256, out_features=1)
        self.activation = activation
        self.tanh = nn.Tanh()

    def forward(self, encoding):
        value = self.valueconv(encoding)
        value = self.valuebn(value)
        value = self.activation(value)
        value = value.view(-1, self.board_pos)
        value = self.valuefc1(value)
        value = self.activation(value)
        value = self.valuefc2(value)
        win_value = self.tanh(value)
        return win_value


class PhilipNet(nn.Module):
    """
    Consolidates all the networks into one. Takes into account
        game: game being played
        block: type of Block being used(ResidualBlock)
        num_block: int for number of blocks in network
        in_channels: number of game states being loaded
        enc_channels: number of channels in hidden layers for residual block
        activation: activation function
        batch_on: turns on or off batch_norm
        pass_move: whether the game into consideration has a pass move
    """

    def __init__(
        self,
        game,
        block=ResidualBlock,
        num_blocks=40,
        in_channels=1,
        enc_channels=256,
        activation=nn.ReLU(inplace=False),
        batch_on=True,
        pass_move=False,
    ):

        super(PhilipNet, self).__init__()
        self.EncodingNet = SidNet(
            block, in_channels, enc_channels, num_blocks, activation, batch_on
        )
        self.PolicyNet = PolicyHead(
            enc_channels, game, activation, batch_on, pass_move
        )
        self.ValueNet = ValueHead(enc_channels, game, activation, batch_on)

    def forward(self, x):

        encoding = self.EncodingNet(x)
        policy = self.PolicyNet(encoding)
        value = self.ValueNet(encoding)

        return policy, value
