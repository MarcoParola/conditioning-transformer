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

To run inference on test set to compute some metrics, specify the weight model path by setting `weight` parameter (I ususally download it from wandb and I copy it in `checkpoint` folder).
```sh
python test.py model=detr weight=checkpoint/best.pt
```

## Acknowledgement
Special thanks to [@clive819](https://github.com/clive819) for making an implementation of DETR public [here](https://github.com/clive819/Modified-DETR). Special thanks to [@hustvl](https://github.com/hustvl) for YOLOS [original implementation](https://github.com/hustvl/YOLOS)
