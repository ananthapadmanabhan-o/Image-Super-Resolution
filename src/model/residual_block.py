from torch import nn

class ResidualBlock(nn.Module):
    '''Residual Block for Generator'''
    def __init__(self,channels=64):
        super().__init__()
        
        self.channels = channels

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=self.channels,out_channels=self.channels,kernel_size=3,stride=1),
            nn.BatchNorm2d(num_features=self.channels),
            nn.PReLU(),
            nn.Conv2d(in_channels=self.channels,out_channels=self.channels,kernel_size=3,stride=1),
            nn.BatchNorm2d(num_features=channels)
        )

    def forward(self,x):
        out = self.block(x)
        return out+x 
        
