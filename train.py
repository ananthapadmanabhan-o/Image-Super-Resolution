from srgan.data import SrganDataset
from srgan.model import Generator, Discriminator
from srgan.losses import GenLoss, DiscLoss
from srgan.utils import read_yaml, create_directories
from srgan.train import SrganTrainer



def main():
    config = read_yaml('config.yaml')

    model_config = config.MODEL
    data_config = config.DATA


    create_directories([model_config.ROOT_DIR])


    '''Dataset parameters'''
    dataset_root_dir = data_config.TRAIN_SOURCE_DIR
    crop_size = int(data_config.CROP_SIZE)
    downscale = int(data_config.DOWN_SCALE)

    dataset = SrganDataset(
    root_dir=dataset_root_dir,
    crop_size=crop_size,
    downscale=downscale
    )


    '''Model parameters'''
    res_block_num = int(model_config.NUM_BLOCKS)
    scale_factor = int(model_config.SCALE_FACTOR)
    lr = float(model_config.LR)
    model_path = model_config.MODEL_PATH
    device = model_config.DEVICE

    batch_size = int(model_config.BATCH_SIZE)
    epochs = int(model_config.EPOCHS)



    gen = Generator(num_blocks=res_block_num,scale_factor=scale_factor)
    disc = Discriminator()

    g_loss = GenLoss()
    d_loss = DiscLoss()

    model_trainer = SrganTrainer(
        generator=gen,
        generator_loss=g_loss,
        discriminator=disc,
        discriminator_loss=d_loss,
        path=model_path,
        device= device
    )

    model_trainer.train(
        dataset=dataset,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr
    )


if __name__ == '__main__':
    main()