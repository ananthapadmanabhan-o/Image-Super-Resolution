from srgan.utils import read_yaml
from srgan.data import DataIngestion

def main():

    config = read_yaml('config.yaml')
    data_ingestion_config = config.DATA_INGESTION

    '''Data Ingestion parameters'''
    train_data_url = data_ingestion_config.TRAIN_SOURCE_URL
    validation_data_url = data_ingestion_config.VALID_SOURCE_URL
    dir_path = data_ingestion_config.ROOT_DIR



    '''Training Data ingestion'''
    train_data_ingestion = DataIngestion(
        url=train_data_url,
        dir_path=dir_path
    )
    train_data_ingestion.ingest_data()



    '''Validation data ingestion'''
    valid_data_ingestion = DataIngestion(
        url=validation_data_url,
        dir_path=dir_path
    )

    valid_data_ingestion.ingest_data()




if __name__ == '__main__':
    main()