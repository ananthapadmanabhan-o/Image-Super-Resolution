from srgan.utils import read_yaml

configurations = read_yaml('config.yaml')
parameters = read_yaml('params.yamls')

CONFIG_DATA_INGESTION = configurations.data_ingestion