from . import *

import os 
import sys
import logging


log_format = '[%(asctime)s: %(levelname)s: %(module)s: %(message)s]'

log_dir = 'logs'
log_file_path = 'logs/runnig_logs.log'

os.makedirs(log_dir,exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger('ImageSuperResolution')


