from srgan.utils import read_yaml
from srgan.data import DataIngestion

def main():

    config = read_yaml('config.yaml')
    data_ingestion_config = config.DATA_INGESTION


    # train_data_ingestion = DataIngestion(
    #     url=data_ingestion_config.TRAIN_SOURCE_URL,
    #     dir_path=data_ingestion_config.UNZIP_DIR
    # )

    # train_data_ingestion.ingest_data()


    valid_data_ingestion = DataIngestion(
        url=data_ingestion_config.VALID_SOURCE_URL,
        dir_path=data_ingestion_config.UNZIP_DIR
    )

    valid_data_ingestion.ingest_data()

if __name__ == '__main__':
    main()