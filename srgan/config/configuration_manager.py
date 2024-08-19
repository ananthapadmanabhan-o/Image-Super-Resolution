from srgan.utils import read_yaml, create_directories


class ConfigManager:
    def __init__(
            self,
            config_path,
            params_path
            ) -> None:
        
        self.config = read_yaml(config_path)
        self.params = read_yaml(params_path)
        pass
    def get_data_ingestion_config(self):
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        pass 
    def data_transformation_config(self):
        pass