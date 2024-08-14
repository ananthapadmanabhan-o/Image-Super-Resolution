from torch import nn
from .residual_block import ResidualBlock
from .upsample_block import UpsampleBlock
from math import log2
    
    
class Generator(nn.Module): 
    def __init__(self,scale_factor=4):
        super().__init__()
        num_upsample_block = int(log2(scale_factor))

        self.input_block = nn.Sequential(
            nn.Conv2d(3,64,9,1,padding=4),
            nn.PReLU()
        )

        self.residual_blocks = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
        )

        self.middle_block = nn.Sequential(
            nn.Conv2d(64,64,3,1,padding=1),
            nn.BatchNorm2d(num_features=64)
        )

        # upsample_blocks = [UpsampleBlock(64,256,2) for _ in range(num_upsample_block)]

        self.upsample_block = nn.Sequential(
            UpsampleBlock(64,256,2),
            UpsampleBlock(256,256,2)
        )

        self.output_block = nn.Sequential(
                    nn.Conv2d(256,3,9,1,padding=4),
                    nn.Tanh()
                )

    def forward(self,x):
        input_block_out = self.input_block(x)
        residual_block_out = self.residual_blocks(input_block_out)
        middle_block_out = self.middle_block(residual_block_out)

        input_to_upsample = middle_block_out + input_block_out

        

        upsample_block_out = self.upsample_block(input_to_upsample)

        out = self.output_block(upsample_block_out)
        return out