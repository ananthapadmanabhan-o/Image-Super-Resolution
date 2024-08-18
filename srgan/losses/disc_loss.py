from torch import nn 

class DiscLoss(nn.Module):
    def __init__(self,):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self,disc_output_real,disc_ouput_generated,real_label,generated_label):

        real_loss = self.bce(disc_output_real,real_label)
        generated_loss = self.bce(disc_ouput_generated,generated_label)

        total_loss = 0.5*(real_loss+generated_loss)
        return total_loss
