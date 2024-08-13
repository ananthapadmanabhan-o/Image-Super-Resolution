from src.dataset.dataset import Div2kDataset


from torch.utils.data import DataLoader

dataset = Div2kDataset(
    root_dir='DIV2K_train_HR',
    downscale=4,

)

dataloader = DataLoader(dataset,batch_size=32,num_workers=2)

for batch_idx, (lr,hr) in enumerate(dataloader):
    print('batch number',batch_idx)
    print('\n\n\n',lr)
    break

print(dataset.__len__())


# import os 
# import cv2


# dir_name = 'DIV2K_train_HR'

# img_file = os.listdir(dir_name)

# for i in range(10):
#     img_file_path = os.path.join(dir_name,img_file[i])
#     # print(img_file_path)
#     img = cv2.imread(img_file_path,1)
#     print(img.shape)