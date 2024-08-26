from srgan.data import SrganDataset
from srgan.model import Generator, Discriminator
from srgan.losses import GenLoss, DiscLoss
from srgan.train import SrganTrainer
from srgan.utils import read_yaml, create_directories
import mlflow

"""
Training Pipeling Script
------------------------ 
This scripts initialises the training parameters 
and runs the training pipeline

First collects the parameters and confugurations
from the config.yaml file and initialises models
Then Srgan Trainer is imported and used to train
"""


mlflow_uri = "http://127.0.0.1:8080"

mlflow.set_registry_uri(mlflow_uri)
mlflow.set_experiment(experiment_name='SRGAN')


def main():
    """Reading configurations from config.yaml"""
    config = read_yaml("config.yaml")

    model_config = config.MODEL
    data_config = config.DATA

    model_path = model_config.ROOT_DIR
    create_directories([model_path])

    """Dataset parameters"""
    dataset_root_dir = data_config.TRAIN_SOURCE_DIR
    crop_size = int(data_config.CROP_SIZE)
    downscale = int(data_config.DOWN_SCALE)

    """Dataset initialisation"""
    dataset = SrganDataset(
        root_dir=dataset_root_dir, crop_size=crop_size, downscale=downscale
    )

    """Model parameters"""
    res_block_num = int(model_config.NUM_BLOCKS)
    scale_factor = int(model_config.SCALE_FACTOR)
    lr = float(model_config.LR)
    

    """Training Parameters"""
    device = model_config.DEVICE
    batch_size = int(model_config.BATCH_SIZE)
    epochs = int(model_config.EPOCHS)

    """Model inititalisation"""
    gen = Generator(num_blocks=res_block_num, scale_factor=scale_factor)
    disc = Discriminator()

    """Loss initialisation"""
    g_loss = GenLoss()
    d_loss = DiscLoss()

    """Trainer initialisation"""
    model_trainer = SrganTrainer(
        generator=gen,
        generator_loss=g_loss,
        discriminator=disc,
        discriminator_loss=d_loss,
        path=model_path,
        device=device,
    )
    """Traning"""
    with mlflow.start_run() as run:
        model_trainer.train(dataset=dataset, batch_size=batch_size, epochs=epochs, lr=lr)


# if __name__ == "__main__":
#     main()
