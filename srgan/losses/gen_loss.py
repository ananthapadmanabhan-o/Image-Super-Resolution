from torch import nn
from torchvision import models

class GenLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.vgg = models.vgg19(weights='DEFAULT').features[:20]
        self.vgg.eval()
        for param in self.vgg.parameters():
            param.require_grad = False
        self.mse = nn.MSELoss()

        self.bce = nn.BCEWithLogitsLoss()
        self.content = nn.MSELoss()


    def forward(self,generated_hr,original_hr,disc_out_generated,real_label):
        vgg_hr = self.vgg(original_hr)
        vgg_gen_hr = self.vgg(generated_hr)

        perceptual_loss = self.mse(vgg_gen_hr,vgg_hr)
        content_loss = self.content(generated_hr,original_hr)
        adversarial_loss = self.bce(disc_out_generated,real_label)

        total_loss = content_loss + 0.006 * perceptual_loss + 0.001 * adversarial_loss

        return total_loss
