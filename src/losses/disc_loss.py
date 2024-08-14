from torch import nn 

class DiscLoss(nn.Module):
    def __init__(self,):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self,real_disc_ouput,genereated_disc_output,real_label,generated_label):

        original_loss = self.bce(real_disc_ouput,real_label)
        generated_loss = self.bce(genereated_disc_output,generated_label)

        total_loss = 0.5*(original_loss+generated_loss)
        return total_loss
