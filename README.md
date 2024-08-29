# Image-Super-Resolution-GAN


## Introduction 🚨

One of the fundamental challenge in signal processing is the presence of noise. Random nature of noise makes it difficult to process the data and distructively affects the quality of data collected. The field of computer vision is growing large with its vast applications accross different fields. Advancements in computer vision demands high quality image data for building robust state of the art models. 

This project aims at exploring autoencoder architecture for image denoising. AutoEncoder arcitecture is a neural netwoek architecture widely used for denoising and reconstruction of images and video files.

check out [GAN](https://en.wikipedia.org/wiki/Autoencoder)
GitHub link [Project-Repo](https://github.com/ananthapadmanabhan-o/Image-Denoising-AutoEncoder)



One of the fundamental challenge in signal processing is the presence of noise. Random nature of noise makes it difficult to process the data and distructively affects the quality of data collected. The field of computer vision is growing large with its vast applications accross different fields. Advancements in computer vision demands high quality image data for building robust state of the art models. 

This project aims at exploring autoencoder architecture for image denoising. AutoEncoder arcitecture is a neural netwoek architecture widely used for denoising and reconstruction of images and video files.

check out [AutoEncoders](https://en.wikipedia.org/wiki/Autoencoder)
learn more about [Image-Noise](https://en.wikipedia.org/wiki/Image_noise#:~:text=Image%20noise%20is%20random%20variation,of%20an%20ideal%20photon%20detector.)
GitHub link [Project-Repo](https://github.com/ananthapadmanabhan-o/Image-Denoising-AutoEncoder)
Data set linke [link](https://www.kaggle.com/datasets/huaiyingu/bsd100)

##### check out the app at [streallit app link](https://image-denoising-autoencoder.streamlit.app/)


## 🚀 Installation and Setup 🔥

### STEP 01
#### 💻Clone the repository
```bash 
git clone https://github.com/ananthapadmanabhan-o/Image-Super-Resolution.git
```

### STEP 02
#### Create a virtual environment
```bash 
cd Image-Super-Resolution
python3 -m venv venv
```


### STEP 03
#### Activate the virtual environment

```bash
source venv/bin/activate
```


### STEP 04
#### Install the requirements 🔧
```bash 
pip install -r requirements.txt
```


### STEP 05
#### Model Configurations setup ⚙️. 
- Model parameters like epochs, batch size etc can be modified in the config.yaml file before training

### STEP 06
#### To Train custom model run this command
```bash
python3 train.py
```


### For Running Streamlit UI
#### Run Home.py
```bash 
streamlit run Home.py
```
