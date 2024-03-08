# conditioning-transformer

[![size](https://img.shields.io/github/languages/code-size/MarcoParola/conditioning-transformer?style=plastic)]()
[![license](https://img.shields.io/static/v1?label=OS&message=Linux&color=green&style=plastic)]()
[![Python](https://img.shields.io/static/v1?label=Python&message=3.10&color=blue&style=plastic)]()


Design of a transformer-based architecture for object detection conditioned by metadata:
- DEtection TRanformer (DETR)
- Vision Transformer (ViT) ???

## **Metadata integration strategies**

We develop the following strategies to incorporate metadata information into image processing:
- Baseline (no metadata)
- Early concatenation
- Early summation


## **Installation**

To install the project, simply clone the repository and get the necessary dependencies:
```sh
git clone https://github.com/MarcoParola/conditioning-transformer.git
cd conditioning-transformer
mkdir models data
```

Create and activate virtual environment, then install dependencies. 
```sh
python -m venv env
. env/bin/activate
python -m pip install -r requirements.txt 
```

Next, create a new project on [Weights & Biases](https://wandb.ai/site). Log in and paste your API key when prompted.
```sh
wandb login 
```

## **Usage**



To perform a training run by setting `model` parameter:
```sh
python train.py model=detr
```
`model` can assume the following value `detr`, `early-sum-detr`, `early-concat-detr`, `early-shift-detr`.

To run inference on test set to compute some metrics, specify the weight model path by setting `weight` parameter (I ususally download it from wandb and I copy it in `checkpoint` folder).
```sh
python test.py model=detr weight=checkpoint/best.pt
```

## Acknowledgement
Special thanks to [@clive819](https://github.com/clive819) for making an implementation of DETR public [here](https://github.com/clive819/Modified-DETR). Special thanks to [@hustvl](https://github.com/hustvl) for YOLOS [original implementation](https://github.com/hustvl/YOLOS)