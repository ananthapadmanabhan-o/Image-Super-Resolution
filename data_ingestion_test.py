from srgan.utils import read_yaml

from srgan.data import DataIngestion

config = read_yaml('config.yaml')
data_ingestion_config = config.data_ingestion


# train_data_ingestion = DataIngestion(
#     url=data_ingestion_config.train_source_url,
#     dir_path=data_ingestion_config.unzip_dir
# )

# train_data_ingestion.ingest_data()


valid_data_ingestion = DataIngestion(
    url=data_ingestion_config.valid_source_url,
    dir_path=data_ingestion_config.unzip_dir
)

valid_data_ingestion.ingest_data()