from torch import nn


class UpsampleBlock(nn.Module):
    def __init__(self, in_channel=64, out_channel=256, up_sample=2):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channel,
                out_channel * up_sample**2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.PixelShuffle(up_sample),
            nn.PReLU(),
        )

    def forward(self, x):
        return self.block(x)
