import streamlit as st

st.set_page_config(
    page_title="Documentation",
    layout='centered',
    page_icon="📄",
    initial_sidebar_state="expanded",
)

st.header('Welcome to Documentation! 👋', divider='rainbow')




body = '''
# Image-Super-Resolution-GAN


## Introduction 🚨

![Alt text](assets/srgan_cover.png)


check out [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802v5)
GitHub link [Project-Repo](https://github.com/ananthapadmanabhan-o/Image-Super-Resolution)

Data set linke [link](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

## SRGAN Architecture

![Alt text](assets/srgan_block.png)

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

### STEP 07
#### For Running Streamlit UI
```bash 
streamlit run Home.py
```

### STEP 08
#### MlFlow Experiment Tracking

![Alt text](assets/mlflow.png)
```bash
mlflow server --host 127.0.0.1 --port 8080
```
'''


st.markdown(body)