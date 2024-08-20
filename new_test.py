import torch
from srgan.data import SrganDataset
from srgan.model import Generator, Discriminator
from srgan.losses import GenLoss, DiscLoss
from torch.optim import Adam 
from torchvision import transforms

from srgan.train import SrganTrainer


data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4112, 0.4435, 0.4562],std=[0.1613, 0.1644, 0.1737]),
])


dataset = SrganDataset(
  root_dir='image_hr',
  downscale=4,
  transform=data_transforms
)



gen = Generator(num_blocks=2,scale_factor=4)
disc = Discriminator()
g_loss = GenLoss()
d_loss = DiscLoss()


g_optim = Adam(gen.parameters(),lr = 0.01)
d_optim = Adam(disc.parameters(),lr = 0.01)

model_trainer = SrganTrainer(
    generator=gen,
    generator_loss=g_loss,
    discriminator=disc,
    discriminator_loss=d_loss,
    device='cuda'
)



model_trainer.train(
    dataset=dataset,
    batch_size=1,
    epochs=1,
    gen_optimizer=g_optim,
    disc_optimizer=d_optim
)