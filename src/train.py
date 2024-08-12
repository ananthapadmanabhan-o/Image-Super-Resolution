from torch import nn
from model import Generator, Discriminator
from torch.optim import Adam 


gen = Generator()
disc = Discriminator()

lr = 0.0001

gen_optim = Adam(gen.parameters(),lr=lr)
disc_optim = Adam(disc.parameters(),lr=lr)

