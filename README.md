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

```sh
python -m scripts.dataset-stats --dataset data/Train.json # training set
python -m scripts.dataset-stats --dataset data/Valid.json # validation set
python -m scripts.dataset-stats --dataset data/Test.json # test set
```