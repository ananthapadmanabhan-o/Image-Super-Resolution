import os 
from torch.utils.data import Dataset
import cv2

class Div2kDataset(Dataset):
    def __init__(self,root_dir,downscale=4,transform=None):
        super().__init__() 

        self.root_dir = root_dir
        self.downscale = downscale
        self.transform = transform
        self.images = os.listdir(self.root_dir)
    
    def __getitem__(self, index):

        hr_img = cv2.imread(self.images[index],1)
        hr_img = cv2.cvtColor(hr_img,cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = hr_img.size

        crop_h = (img_h//self.downscale)*self.downscale
        crop_w = (img_w//self.downscale)*self.downscale

        hr_img = hr_img[:crop_h,:crop_w,:]
        lr_img = cv2.resize(hr_img,(crop_w//self.downscale,crop_h//self.downscale))


        if self.transform:
            hr_img = self.transform(hr_img)
            lr_img = self.transform(lr_img)


        return lr_img, hr_img
    
    
    def __len__(self):
        return len(self.images)