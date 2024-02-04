from argparse import ArgumentParser, Namespace
from collections import defaultdict
from time import time
from typing import Dict, List, Any
import numpy as np
import torch
from torch import nn, Tensor

from .boxOps import boxCxcywh2Xyxy


def baseParser() -> ArgumentParser:
    parser = ArgumentParser('Detection Transformer', add_help=False)

    # MARK: - model parameters
    # backbone
    parser.add_argument('--numGroups', default=8, type=int)
    parser.add_argument('--growthRate', default=32, type=int)
    parser.add_argument('--numBlocks', default=[6] * 4, type=list)

    # transformer
    parser.add_argument('--hiddenDims', default=512, type=int)
    parser.add_argument('--numHead', default=8, type=int)
    parser.add_argument('--numEncoderLayer', default=6, type=int)
    parser.add_argument('--numDecoderLayer', default=6, type=int)
    parser.add_argument('--dimFeedForward', default=2048, type=int)
    parser.add_argument('--dropout', default=.1, type=float)
    parser.add_argument('--numQuery', default=80, type=int)
    parser.add_argument('--numClass', default=4, type=int)

    # MARK: - dataset
    parser.add_argument('--targetHeight', default=608, type=int)
    parser.add_argument('--targetWidth', default=608, type=int)

    # MARK: - miscellaneous
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    parser.add_argument('--weight', default='checkpoint/mango.pt', type=str)
    parser.add_argument('--seed', default=4, type=int)

    # MARK: - training config
    parser.add_argument('--lr', default=1e-6, type=float)
    parser.add_argument('--lrBackbone', default=1e-5, type=float)
    parser.add_argument('--batchSize', default=8, type=int)
    parser.add_argument('--weightDecay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('--lrDrop', default=1000, type=int)
    parser.add_argument('--clipMaxNorm', default=.1, type=float)

    # MARK: - loss
    parser.add_argument('--classCost', default=1., type=float)
    parser.add_argument('--bboxCost', default=5., type=float)
    parser.add_argument('--giouCost', default=2., type=float)
    parser.add_argument('--eosCost', default=.1, type=float)

    # MARK: - dataset
    parser.add_argument('--dataDir', default='./data', type=str)
    parser.add_argument('--trainAnnFile', default='./data/Train.json', type=str)
    parser.add_argument('--valAnnFile', default='./data/Valid.json', type=str)
    parser.add_argument('--testAnnFile', default='./data/Test.json', type=str)

    # MARK: - miscellaneous
    parser.add_argument('--outputDir', default='./checkpoint', type=str)
    parser.add_argument('--taskName', default='mango', type=str)
    parser.add_argument('--numWorkers', default=1, type=int)
    parser.add_argument('--multi', default=False, type=bool)
    parser.add_argument('--amp', default=False, type=bool)

    # MARK: - wandb
    parser.add_argument('--wandbEntity', default='marcoparola', type=str)
    parser.add_argument('--wandbProject', default='conditioning-transformer', type=str)

    return parser


class PostProcess(nn.Module):
    def __init__(self):
        super(PostProcess, self).__init__()

    @torch.no_grad()
    def forward(self, x: dict, imgSize: Tensor) -> List[Dict[str, Tensor]]:
        logits, bboxes = x['class'], x['bbox']

        prob = nn.functional.softmax(logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        boxes = boxCxcywh2Xyxy(bboxes)

        imgW, imgH = imgSize.unbind(1)

        scale = torch.stack([imgW, imgH, imgW, imgH], 1).unsqueeze(1)
        boxes *= scale

        return [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]



def logMetrics(metrics: Dict[str, Tensor]):
    log = '[ '
    log += ' ] [ '.join([f'{k} = {v.cpu().item():.4f}' for k, v in metrics.items()])
    log += ' ]'
    print(log)


def cast2Float(x):
    if isinstance(x, list):
        return [cast2Float(y) for y in x]
    elif isinstance(x, dict):
        return {k: cast2Float(v) for k, v in x.items()}
    return x.float()
