from srgan.data import SrganDataset
from srgan.model import Generator, Discriminator
from srgan.losses import GenLoss, DiscLoss
from torch.optim import Adam 
from torchvision import transforms
from srgan.utils import read_yaml, create_directories
from srgan.train import SrganTrainer



def main():
    config = read_yaml('config.yaml')

    
    model_config = config.MODEL
    data_config = config.DATA_TRANSFORMATION


    create_directories([model_config.ROOT_DIR])

    
    
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4112, 0.4435, 0.4562],std=[0.1613, 0.1644, 0.1737]),
    ])


    dataset = SrganDataset(
    root_dir=data_config.VALID_DATA_TANSFORMED,
    downscale=int(data_config.DOWN_SCALE),
    transform=data_transforms
    )

    gen = Generator(num_blocks=int(model_config.NUM_BLOCKS),scale_factor=int(model_config.SCALE_FACTOR))
    disc = Discriminator()
    g_loss = GenLoss()
    d_loss = DiscLoss()
    g_optim = Adam(gen.parameters(),lr = float(model_config.LR))
    d_optim = Adam(disc.parameters(),lr = float(model_config.LR))

    model_trainer = SrganTrainer(
        generator=gen,
        generator_loss=g_loss,
        discriminator=disc,
        discriminator_loss=d_loss,
        device= model_config.DEVICE
    )

    model_trainer.train(
        dataset=dataset,
        batch_size=int(model_config.BATCH_SIZE),
        epochs=int(model_config.EPOCHS),
        gen_optimizer=g_optim,
        disc_optimizer=d_optim
    )


if __name__ == '__main__':
    main()