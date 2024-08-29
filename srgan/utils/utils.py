import os
import yaml
from box import ConfigBox
from srgan import logger
from PIL import Image
import torch
from torchvision.transforms.functional import pil_to_tensor,to_pil_image

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
 model = torch.load(model_path).to(device)
    Args:
        dir_path (list): list of directories to be created

    """
    for path in dir_path:
        os.makedirs(path, exist_ok=True)
        logger.info(f"Created directory {path}")



def predict(img_path,model_path,device='cpu'):

    img = Image.open(img_path)
    img_tensor = pil_to_tensor(img) / 255
    input_tensor = img_tensor.unsqueeze(0).to(device)
    
    model = torch.load(model_path).to(device)
    model.eval()
    with torch.no_grad():
        output_tensor = model(input_tensor)
        
    out_img = to_pil_image(output_tensor.squeeze(0))

    return out_img