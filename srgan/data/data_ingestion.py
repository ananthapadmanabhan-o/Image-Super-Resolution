from download import download
from srgan import logger

class DataIngestion:
    def __init__(self,url,dir_path,kind='zip') -> None:
        self.url = url 
        self.dir_path = dir_path
        self.kind = kind
        
        
    def ingest_data(self):
        logger.info(f'Data Ingestion Started')
        path = download(
            url=self.url,
            path=self.dir_path,
            kind=self.kind,
            progressbar=True,
            verbose=True,
            replace=True
        )
        logger.info(f'Data ingested successfull')
        
