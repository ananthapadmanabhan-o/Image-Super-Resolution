from torch import nn
from torchvision import models

class VggLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.vgg = models.vgg19(pretrained = True).features[:36]
        self.vgg.eval()
        self.loss = nn.MSELoss()

    def forward(self,generated_hr,hr):
        vgg_hr = self.vgg(hr)
        vgg_gen_hr = self.vgg(generated_hr)

        return self.loss(vgg_gen_hr,vgg_hr)



