import os
from srgan.utils import predict
from tqdm import tqdm
import warnings


warnings.filterwarnings("ignore")

"""
Model inferencing pipeline

Loads model from local and input from runs/inputs
model predicts and saves the output to runs/outputs
"""


def main():

    model_path = "models/srgan6_4x.pth"

    img_dir = os.path.join("runs", "inputs")
    out_dir = os.path.join("runs", "outputs")
    img_files = os.listdir(img_dir)

    print(f"Model loaded from {model_path}")


    for img_file_name in tqdm(img_files):
        img_path = os.path.join(img_dir, img_file_name)

        out_img = predict(img_path, model_path)
        output_path = os.path.join(out_dir, img_file_name)
        out_img.save(output_path)

    print(f"Outputs saved in {out_dir}")


if __name__ == "__main__":
    main()
