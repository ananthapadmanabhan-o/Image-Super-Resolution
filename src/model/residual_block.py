from torch import nn

class ResidualBlock(nn.Module):
    '''Residual Block for Generator'''
    def __init__(
        self,
        in_channel=64, 
        out_channel=64, 
        kernel_size=3,
        stride=1
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=kernel_size,
                stride=stride
            ),
            nn.BatchNorm2d(num_features=out_channel),
            nn.PReLU(num_parameters=out_channel),
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=kernel_size,
                stride=stride
            ),
            nn.BatchNorm2d(num_features=out_channel)
            
        )

        def forward(self,x):
            out = self.block(x)
            return out+x 
        
        