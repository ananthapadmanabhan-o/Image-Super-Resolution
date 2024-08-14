from torch import nn 
from .conv_block import ConvBlock

class Discriminator(nn.Module):
    ''' Class for the Discriminator '''

    def __init__(self):
        super().__init__()
        self.block_0 = nn.Sequential(
            nn.Conv2d(3,64,3,1),
            nn.LeakyReLU()
        )

        self.block_1 = nn.Sequential(
            ConvBlock(64,64,3,2),
            ConvBlock(64,128,3,1),
            ConvBlock(128,128,3,2),
            ConvBlock(128,256,3,1),
            ConvBlock(256,256,3,2),
            ConvBlock(256,512,3,1),
            ConvBlock(512,512,3,2),
        )

        self.block_2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self,x):
        x = self.block_0(x)
        x = self.block_1(x)
        x = self.block_2(x)

        return x