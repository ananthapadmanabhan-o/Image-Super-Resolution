from torch import nn
from residual_block import ResidualBlock
from upsample_block import UpsampleBlock
    
class Generator(nn.Module): 
    def __init__(self):
        super().__init__()

        self.input_block = nn.Sequential(
            nn.Conv2d(3,64,9,1),
            nn.PReLU(64)
        )

        self.residual_blocks = nn.Sequential(
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
        )

        self.middle_block = nn.Sequential(
            nn.Conv2d(64,64,3,1),
            nn.BatchNorm2d(num_features=64)
        )

        self.upsample_block = nn.Sequential(
            UpsampleBlock(64,256,3,1),
            UpsampleBlock(256,256,3,1)
        )

        self.output_block = nn.Conv2d(256,3,9,1)


        def forward(self,x):
            input_block_out = self.input_block(x)
            residual_block_out = self.residual_blocks(input_block_out)
            middle_block_out = self.middle_block(residual_block_out)
            middle_block_sum = middle_block_out + input_block_out
            upsample_block_01 = self.upsample_block(middle_block_sum)
            upsample_block_02 = self.upsample_block(upsample_block_01)
            out = self.output_block(upsample_block_02)
            return out