import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class SrganDataset(Dataset):
    """
    Custom Dataset for srgan

    Attributes
    ----------
    root_dir:
        root folder of dataset
    crop_size:
        crop size for downsizing image
    downscale:
        for downscaling the low resulotion image from high resolution image


    Returns:
        low resolution and high resolution image
    """
    def __init__(
        self,
        root_dir,
        crop_size=100,
        downscale=4,
    ):
        super().__init__()

        self.root_dir = root_dir
        self.crop_size = crop_size
        self.downscale = downscale
        self.images = os.listdir(self.root_dir)

        self.hr_transform = transforms.Compose(
            [
                transforms.PILToTensor(),
                transforms.CenterCrop(self.crop_size),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=0, std=1),
            ]
        )

        self.lr_transform = transforms.Compose(
            [transforms.Resize(self.crop_size // self.downscale)]
        )

    def __getitem__(self, index):

        img_file_path = os.path.join(self.root_dir, self.images[index])
        hr_img = Image.open(img_file_path)

        hr_img = self.hr_transform(hr_img)
        lr_img = self.lr_transform(hr_img)

        return lr_img, hr_img

    def __len__(self):
        return len(self.images)
