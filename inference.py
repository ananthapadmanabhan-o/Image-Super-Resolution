from PIL import Image
import torch
from torchvision.transforms.functional import pil_to_tensor

img_path = '13_hr.png'
model_path = 'models/model.pth'

img = Image.open(img_path)

img_tensor = pil_to_tensor(img)

input_tensor = torch.unsqueeze(img_tensor,0)


print(input_tensor.shape)

model = torch.load(model_path)

with torch.inference_mode():
    output_tensor = model(input_tensor)

print(output_tensor.shape)