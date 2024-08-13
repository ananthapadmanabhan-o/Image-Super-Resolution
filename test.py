from src.dataset.dataset import Div2kDataset
from torch.utils.data import DataLoader

dataset = Div2kDataset(
    root_dir='DIV2K_train_HR',
    downscale=4,

)

print(dataset)