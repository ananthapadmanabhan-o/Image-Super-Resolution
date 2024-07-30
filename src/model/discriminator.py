from torch import nn 

class ConvBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride
        ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU()
        )


    def forward(self,x):
        x = self.block(x)
        return x




class Discriminator(nn.Module):
    '''
    Discriminator class for gan

    '''

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=3,
                stride=1
            ),
            nn.LeakyReLU()            
        )
        
        
        nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1
        )

        self.blocks = nn.Sequential(
            ConvBlock(64,64,3,2),
            ConvBlock(64,128,3,1),
            ConvBlock(128,128,3,2),
            ConvBlock(128,256,3,1),
            ConvBlock(256,256,3,2),
            ConvBlock(256,512,3,1),
            ConvBlock(512,512,3,2)    
        )

        self.dense = nn.Sequential(
            nn.Linear(512,1024),
            nn.LeakyReLU(),
            nn.Linear(512,1),
            nn.Sigmoid()
        )


        
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.dense(x)

        return x
    