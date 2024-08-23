from PIL import Image
import torch
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
import warnings
warnings.filterwarnings('ignore')




img_path = '13_hr.png'
model_path = 'models/model.pth'
output_path = 'runs/output.png'
device = 'cuda'




img = Image.open(img_path)
img_tensor = pil_to_tensor(img)/255
input_tensor = torch.unsqueeze(img_tensor,0).to(device)

model = torch.load(model_path).to(device)
model.eval()
print(f'Model loaded from {model_path}')

with torch.no_grad():
    output_tensor = model(input_tensor)

out_img = to_pil_image(torch.squeeze(output_tensor,0))

out_img.save(output_path)

print(f'Output saved in {output_path}')
