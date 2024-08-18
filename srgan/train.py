import torch
from data import Div2kDataset
from model import Generator, Discriminator
from losses import GenLoss, DiscLoss 
from torch.optim import Adam 



device = 'cuda' if torch.cuda.is_available() else 'cpu'

lr = 0.0001
num_epochs = 1


gen = Generator().to(device)
disc = Discriminator().to(device)

gen_optim = Adam(gen.parameters(),lr=lr)
disc_optim = Adam(disc.parameters(),lr=lr)

genloss = GenLoss()
discloss = DiscLoss()

