from srgan.utils import read_yaml

config = read_yaml('config.yaml')
model = config.MODEL

print(model.LR)