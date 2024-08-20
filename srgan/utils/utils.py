import os 
import yaml
from pathlib import Path
from box import ConfigBox
from srgan import logger 
from ensure import ensure_annotations



# @ensure_annotations
def read_yaml(yaml_path: Path) -> ConfigBox:
    '''
    reads the yaml file from the path and
    returns the content as ConfigBox

    Args:
        yaml_path (Path)
    
    Returns:
        ConfigBox type of contents inside the yaml file
    '''
    try:
        with open(yaml_path) as yaml_file:
            contents = yaml.safe_load(yaml_file)
            logger.info(f'{yaml_path} loaded Successfully')
            return ConfigBox(contents)
    except Exception as e:
        raise e


# @ensure_annotations  
def create_directories(dir_path: list) -> None:
    '''
    creates the directories in the given list

    Args:
        dir_path (list): list of directories to be created

    '''
    for path in dir_path:
        os.makedirs(path,exist_ok=True)
        logger.info(f'Created directory {path}')
