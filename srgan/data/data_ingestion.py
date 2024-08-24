from download import download
from srgan import logger
from srgan.utils import create_directories


class DataIngestion:
    """
    DataIngestion downloads the data
    from url and saves it in local folder

    Attributes
    ----------
    url:
        url of source data
    dir_path:
        local path for storing data
    kind:
        file type

    Methods
    -------
    ingest_data:
        downloads and unzips the data
    """

    def __init__(self, url, dir_path, kind="zip") -> None:
        self.url = url
        self.dir_path = dir_path
        self.kind = kind

    def ingest_data(self):
        """
        Downloads the data to source folder
        """
        create_directories([self.dir_path])
        logger.info(f"Data Ingestion Started")
        path = download(
            url=self.url,
            path=self.dir_path,
            kind=self.kind,
            progressbar=True,
            verbose=True,
            replace=True,
        )
        logger.info(f"Data ingested successfull to {self.dir_path}")
