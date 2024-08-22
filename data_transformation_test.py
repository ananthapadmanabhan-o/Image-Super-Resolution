from srgan.data import DataTransformation
from srgan.utils import read_yaml, create_directories


config = read_yaml('config.yaml')
data_transformation_config = config.data_transformation


train_data_transformation = DataTransformation(
    source_dir=data_transformation_config.train_source_dir,
    destination_dir=data_transformation_config.train_data_transformed
)

train_data_transformation.transform_data()


# valid_data_transformation = DataTransformation(
#     source_dir=data_transformation_config.valid_source_dir,
#     destination_dir=data_transformation_config.valid_data_transformed
# )

# valid_data_transformation.transform_data()
