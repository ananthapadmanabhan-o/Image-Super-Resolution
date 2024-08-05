from torch import nn 

class ConvBlock(nn.Module):
    ''' Convolutional Block for Discriminator '''
    def __init__(
            self,
            in_channel=3,
            out_channel=3,
            kernel=3,
            stride=2
        ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=kernel,
                stride=stride
            ),
            nn.BatchNorm2d(
                num_features=out_channel
            ),
            nn.LeakyReLU()
        )

    def forward(self,x):
        return self.block(x)
