import torch
from torch import nn 

class DiscLoss(nn.Module):
    def __init__(self,):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self,generated_hr,original_hr):

        original_label = torch.ones_like(original_hr)
        generated_label = torch.zeros_like(generated_hr)

        original_loss = self.bce(original_hr,original_label)
        generated_loss = self.bce(generated_hr,generated_label)

        total_loss = 0.5*(original_loss+generated_loss)
        return total_loss
