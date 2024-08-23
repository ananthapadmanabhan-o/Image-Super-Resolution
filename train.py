from srgan.data import SrganDataset
from srgan.model import Generator, Discriminator
from srgan.losses import GenLoss, DiscLoss
from torch.optim import Adam 
from srgan.utils import read_yaml, create_directories
from srgan.train import SrganTrainer



def main():
    config = read_yaml('config.yaml')

    model_config = config.MODEL
    data_config = config.DATA


    create_directories([model_config.ROOT_DIR])


    dataset = SrganDataset(
    root_dir=data_config.TRAIN_SOURCE_DIR,
    crop_size=data_config.CROP_SIZE,
    downscale=int(data_config.DOWN_SCALE)
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
        path=model_config.MODEL_PATH,
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