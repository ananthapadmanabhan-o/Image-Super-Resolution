import os
from patchify import patchify
from srgan import logger
from PIL import Image
import numpy as np
from tqdm import tqdm



class DataTransformation:
    def __init__(
            self,
            source_dir,
            destination_dir
    ) -> None:
        self.source_dir = source_dir
        self.destination_dir = destination_dir
        

    def transform_data(self,patch_size=256):
        logger.info('Data Transformation started')
        image_list = os.listdir(self.source_dir)

        count = 1

        for img in tqdm(image_list):
            img_path = os.path.join(self.source_dir,img)
            img = np.asarray(Image.open(img_path))



            img_height, img_width, img_channel = img.shape
            
            crop_height = (img_height//patch_size)*patch_size
            crop_width = (img_width//patch_size)*patch_size
            cropped_img = img[:crop_height,:crop_width,:]

            patched_imgs = patchify(
                image=cropped_img,
                patch_size=(patch_size,patch_size,img_channel),
                step=patch_size
            )

            row, col = patched_imgs.shape[:2]


            for i in range(row):
                for j in range(col):
                    img_patch = patched_imgs[i,j,0,:,:]
                    img = Image.fromarray(img_patch)
                    img_save_path = os.path.join(self.destination_dir,f'{count}_hr.png')
                    img.save(img_save_path)
                    count+=1
                    
        logger.info('Data Transformation ended')
