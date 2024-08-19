from download import download
from srgan import logger

class DataIngestion:
    def __init__(self,url,dir_path,download,kind='zip') -> None:
        self.url = url 
        self.dir_path = dir_path
        self.download = download
        self.kind = kind
        
        
    def ingest_data(self):
        path = download(
            url=self.url,
            path=self.dir_path,
            kind=self.kind,
            verbose=False
        )
        logger.info(f'Data ingested successfull to {path}')
        


"""
data_ingestion:
  root_dir: artifacts/data_ingestion
  train_source_url: http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
  valid_source_url: http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip
  unzip_dir: artifacts/data_ingestion
"""
