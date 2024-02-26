from argparse import ArgumentParser, Namespace
from collections import defaultdict
from time import time
from typing import Dict, List, Any
import numpy as np
import torch
from torch import nn, Tensor

from .boxOps import boxCxcywh2Xyxy


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
