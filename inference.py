import os
from PIL import Image
import torch
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

"""
Model inferencing pipeline

Loads model from local and input from runs/inputs
model predicts and saves the output to runs/outputs
"""


def main():

    model_path = "demo-models/model2x-2.pth"
    device = "cuda"

    img_dir = os.path.join("runs", "inputs")
    out_dir = os.path.join("runs", "outputs")
    img_files = os.listdir(img_dir)

    model = torch.load(model_path).to(device)
    print(f"Model loaded from {model_path}")
    model.eval()
    with torch.no_grad():

        for img_file_name in tqdm(img_files):

            img_path = os.path.join(img_dir, img_file_name)
            img = Image.open(img_path)

            img_tensor = pil_to_tensor(img) / 255
            input_tensor = torch.unsqueeze(img_tensor, 0).to(device)

            output_tensor = model(input_tensor)
            out_img = to_pil_image(torch.squeeze(output_tensor, 0))

            output_path = os.path.join(out_dir, img_file_name)
            out_img.save(output_path)

    print(f"Outputs saved in {out_dir}")


if __name__ == "__main__":

    main()
