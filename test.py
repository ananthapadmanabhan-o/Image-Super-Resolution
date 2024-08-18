import torch
from srgan.data.dataset import Div2kDataset
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from srgan.model.generator import Generator
from srgan.model.discriminator import Discriminator

from srgan.losses.disc_loss import DiscLoss
from srgan.losses.gen_loss import GenLoss

from torch.optim import Adam 

from tqdm import tqdm

# torch.autograd.set_detect_anomaly(True)



# Mean: tensor([0.4112, 0.4435, 0.4562])
# Std: tensor([0.1613, 0.1644, 0.1737])


data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4112, 0.4435, 0.4562],std=[0.1613, 0.1644, 0.1737]),
])


dataset = Div2kDataset(
    root_dir='image_hr',
    downscale=4,
    transform=data_transform
)


dataloader = DataLoader(dataset,batch_size=1,num_workers=2,shuffle=True)



device = 'cuda' if torch.cuda.is_available() else 'cpu'

lr = 0.001
num_epochs = 1


generator = Generator().to(device)
discriminator = Discriminator().to(device)

generator_optim = Adam(generator.parameters(),lr=lr)
discriminator_optim = Adam(discriminator.parameters(),lr=lr)

generator_loss = GenLoss().to(device)
discriminator_loss = DiscLoss().to(device)

print(f'Training Device: {device}')

for epoch in range(1,num_epochs+1):

    for bch_idx, (lr_img,hr_img) in enumerate(tqdm(dataloader)):

        lr_img = lr_img.to(device)
        hr_img = hr_img.to(device)

        
        # generate hr image
        generated_hr_img = generator(lr_img)

        # discriminator training

        discriminator_optim.zero_grad()

        

        disc_generated_out = discriminator(generated_hr_img.detach())
        disc_real_out = discriminator(hr_img)

        real_label = torch.ones_like(disc_real_out.detach()).to(device)
        generated_label = torch.zeros_like(disc_generated_out.detach()).to(device)

        d_loss = discriminator_loss(disc_real_out,disc_generated_out,real_label,generated_label)

        d_loss.backward(retain_graph=True)
        discriminator_optim.step()


        # Generator Training

        generator_optim.zero_grad()

        generated_hr_img = generator(lr_img)
        disc_generated_out = discriminator(generated_hr_img.detach())


        g_loss = generator_loss(generated_hr_img,hr_img,disc_generated_out,real_label)
        g_loss.backward()

        generator_optim.step()


    print(f"Epoch [{epoch}/{num_epochs}], Step [{bch_idx}/{len(dataloader)}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")