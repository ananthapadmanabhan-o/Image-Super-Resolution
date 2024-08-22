from srgan.data import DataTransformation
from srgan.utils import read_yaml

def main():
        
    config = read_yaml('config.yaml')
    data_config = config.DATA_TRANSFORMATION


    # train_data_transformation = DataTransformation(
    #     source_dir=data_config.TRAIN_SOURCE_DIR,
    #     destination_dir=data_config.TRAIN_DATA_TRANSFORMED
    # )

    # train_data_transformation.transform_data()


    valid_data_transformation = DataTransformation(
        source_dir=data_config.VALID_SOURCE_DIR,
        destination_dir=data_config.VALID_DATA_TANSFORMED
    )

    valid_data_transformation.transform_data()

if __name__=='__main__':
    main()
