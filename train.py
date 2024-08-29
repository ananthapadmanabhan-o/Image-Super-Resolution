from srgan.data import SrganDataset
from srgan.model import Generator, Discriminator
from srgan.losses import GenLoss, DiscLoss
from srgan.train import SrganTrainer
from srgan.utils import read_yaml, create_directories
import mlflow
from datetime import datetime


"""
Training Pipeling Script
------------------------ 
This scripts initialises the training parameters 
and runs the training pipeline

First collects the parameters and confugurations
from the config.yaml file and initialises models
Then Srgan Trainer is imported and used to train
"""



import dagshub
dagshub.init(repo_owner='ananthapadmanabhan-o', repo_name='Image-Super-Resolution', mlflow=True)


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
    

    """MlFlow Experiment"""
    mlflow.set_experiment(experiment_name=f'SRGAN-{scale_factor}x')
    run_name = f'srgan-{scale_factor}x-version-{datetime.now():%Y-%m-%d %H:%M}'

    mlflow.start_run(run_name=run_name)

    # mlflow_uri = "http://127.0.0.1:8080"
    # mlflow.set_tracking_uri(mlflow_uri)


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
    model, history = model_trainer.train(dataset=dataset, batch_size=batch_size, epochs=epochs, lr=lr)


    """Mlflow Logging"""
    model_params = {
        'Res Block':res_block_num,
        'LR':lr,
        'Epochs':epochs,
        'Batch Size': batch_size
    }

    mlflow.log_params(model_params)

    gloss = history['G_Loss']
    dloss = history['D_Loss']
    
    for i in range(len(history)):
        mlflow.log_metric('G Loss',gloss.iloc[i],step=i)
        mlflow.log_metric('D Loss',dloss.iloc[i],step=i)

    model_name = f"srgan{res_block_num}_{scale_factor}x"

    mlflow.pytorch.log_model(model,model_name)


    mlflow.end_run()


if __name__ == "__main__":
    main()
