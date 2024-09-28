# conditioning-transformer

[![size](https://img.shields.io/github/languages/code-size/MarcoParola/conditioning-transformer?style=plastic)]()
[![license](https://img.shields.io/static/v1?label=OS&message=Linux&color=green&style=plastic)]()
[![Python](https://img.shields.io/static/v1?label=Python&message=3.10&color=blue&style=plastic)]()


Design of a transformer-based architecture for object detection conditioned by metadata:
- DEtection TRanformer (DETR)
- You Only Look at One Sequence (YOLOS)


## **Installation**

To install the project, simply clone the repository and get the necessary dependencies. Then, create a new project on [Weights & Biases](https://wandb.ai/site). Log in and paste your API key when prompted.
```sh
# clone repo
git clone https://github.com/MarcoParola/conditioning-transformer.git
cd conditioning-transformer
mkdir models data

# Create virtual environment and install dependencies 
python -m venv env
. env/bin/activate
python -m pip install -r requirements.txt 

# Weights&Biases login 
wandb login 
```

## **Usage**

To perform a training run by setting `model` parameter that can assume the following value `detr`, `early-sum-detr`, `early-concat-detr`, `yolos`, `early-sum-yolos`, `early-concat-yolos`
```sh
python train.py model=detr
```
The command could also be run specifying the `cropBackground` option by setting it at `true` or `false` resulting on the following training image.

Whole image             |  Cropped image
:-------------------------:|:-------------------------:
![entire_img_2](https://github.com/user-attachments/assets/8895703b-8920-4bea-9394-48fc00a577ed)  |  ![cropped_img_2](https://github.com/user-attachments/assets/5c6e5cce-2f33-4412-9b9e-007afaa81e1b)


 

To run inference on test set to compute some metrics, specify the weight model path by setting `weight` parameter (I ususally download it from wandb and I copy it in `checkpoint` folder).
```sh
python test.py model=detr weight=checkpoint/best.pt
```

## **Training params**
Training hyperparams can be edited in the [config file](./config/config.yaml) or ovewrite by shell
Params             |  Value
:-------------------------:|:-------------------------:
batchSize  |  16
lr  |  1e-6

## Acknowledgement
Special thanks to [@clive819](https://github.com/clive819) for making an implementation of DETR public [here](https://github.com/clive819/Modified-DETR). Special thanks to [@hustvl](https://github.com/hustvl) for YOLOS [original implementation](https://github.com/hustvl/YOLOS)
