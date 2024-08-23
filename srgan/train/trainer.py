from torch import ones_like,zeros_like
from torch.utils.data import DataLoader
from srgan import logger
from tqdm import tqdm
import pandas as pd
import torch

class SrganTrainer:
    def __init__(
            self,
            generator,
            generator_loss,
            discriminator,
            discriminator_loss,
            path,
            device='cuda'
    ) -> None:
        self.device = device
        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)
        self.generator_loss = generator_loss.to(self.device)
        self.discriminator_loss = discriminator_loss.to(self.device)
        self.path = path

      
    
        
    def train(self,dataset,batch_size,epochs,gen_optimizer,disc_optimizer):
        
        train_dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=2,
            shuffle=True
        )

        G_Loss = []
        D_Loss = []

        logger.info(f'Training started on {self.device}')

        for epoch in range(1,epochs+1):
            for bch_idx, (lr_img,hr_img) in enumerate(tqdm(train_dataloader),start=1):

                lr_img = lr_img.to(self.device)
                hr_img = hr_img.to(self.device)

                '''Discriminator Training'''
                disc_optimizer.zero_grad()

                generated_hr_img = self.generator(lr_img)

                disc_out_generated = self.discriminator(generated_hr_img.detach())
                disc_out_real = self.discriminator(hr_img)

                real_label = ones_like(disc_out_real.detach()).to(self.device)
                generated_label= zeros_like(disc_out_generated.detach()).to(self.device)

                disc_loss = self.discriminator_loss(disc_out_real,disc_out_generated,real_label,generated_label)

                disc_loss.backward(retain_graph=True)
                disc_optimizer.step()


                '''Generator training'''
                gen_optimizer.zero_grad()

                generated_hr_img = self.generator(lr_img)
                disc_out_generated = self.discriminator(generated_hr_img)

                gen_loss = self.generator_loss(generated_hr_img,hr_img,disc_out_generated,real_label)
                gen_loss.backward()
                gen_optimizer.step()
            
            G_Loss.append(gen_loss.item())
            D_Loss.append(disc_loss.item())


            if epoch%1==0:
                print(f"Epoch [{epoch}/{epochs}], Step [{bch_idx}/{len(train_dataloader)}], D Loss: {disc_loss.item():.4f}, G Loss: {gen_loss.item():.4f}")
        
        

        torch.save(self.generator,self.path)
        Loss_dataframe = pd.DataFrame({
            'epoch':epoch,
            'G_Loss':G_Loss,
            'D_Loss':D_Loss
        })
        Loss_dataframe.to_csv('model_log.csv')
