from torch import nn


class ConvBlock(nn.Module):
    """
    Convolutional Block for Discriminator

    Attributes
    ----------
    in_channel:
        number of input channels for Conv2d Layer
    out_channel:
        number of output channels for Conv2d Layer
    kernel:
        kernel size for Conv2d Layer
    stride:
        stride for Conv2d Layer
    padding:
        padding for Conv2d Layer
    """

    def __init__(self, in_channel=3, out_channel=3, kernel=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(num_features=out_channel),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.block(x)
