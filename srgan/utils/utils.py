import os
import yaml
from box import ConfigBox
from srgan import logger


def read_yaml(yaml_path) -> ConfigBox:
    """
    reads the yaml file from the path and
    returns the content as ConfigBox

    Args:
        yaml_path (Path)

    Returns:
        ConfigBox type of contents inside the yaml file
    """
    try:
        with open(yaml_path) as yaml_file:
            contents = yaml.safe_load(yaml_file)
            logger.info(f"{yaml_path} loaded Successfully")
            return ConfigBox(contents)
    except Exception as e:
        raise e


def create_directories(dir_path) -> None:
    """
    creates the directories in the given list

    Args:
        dir_path (list): list of directories to be created

    """
    for path in dir_path:
        os.makedirs(path, exist_ok=True)
        logger.info(f"Created directory {path}")
