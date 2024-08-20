from srgan.utils import read_yaml, create_directories

from srgan.data import DataIngestion

config = read_yaml('config.yaml')
data_ingestion_config = config.data_ingestion



print(data_ingestion_config.valid_source_url)

create_directories([data_ingestion_config.unzip_dir])

data_ingestion = DataIngestion(
    url=data_ingestion_config.valid_source_url,
    dir_path=data_ingestion_config.unzip_dir
)

data_ingestion.ingest_data()