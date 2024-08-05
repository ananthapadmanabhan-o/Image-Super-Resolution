from torch import nn

class UpsampleBlock(nn.Module): 
    def __init__(
        self,
        in_channel=64,
        out_channel=64,
        kernel=3,
        stride=1
    ): 
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,kernel,stride),
            nn.BatchNorm2d(out_channel)
        )

    def forward(self,x):
        return self.block(x)