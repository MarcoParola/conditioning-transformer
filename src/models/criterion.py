from typing import Dict, List, Tuple
import torch
from torch import nn, Tensor

from src.utils.boxOps import boxCxcywh2Xyxy, gIoU, boxIoU
from .matcher import HungarianMatcher
from torch.nn.functional import cross_entropy, one_hot
import math


def SetCriterion(args, model):
    if args.model == "yolov8":
        return YoloCriterion(args, model)
    else:
        return TransformerCriterion(args, model)

### For Transformer
class TransformerCriterion(nn.Module):
    def __init__(self, args, model):
        super(SetCriterion, self).__init__()

        self.matcher = HungarianMatcher(args.classCost, args.bboxCost, args.giouCost)
        self.numClass = args.numClass

        self.classCost = args.classCost
        self.bboxCost = args.bboxCost
        self.giouCost = args.giouCost

        emptyWeight = torch.ones(args.numClass + 1)
        # emptyWeight[0] = 5841139 / 5841139
        # emptyWeight[1] = 5841139 / 293280
        # emptyWeight[2] = 5841139 / 32393
        # emptyWeight[3] = 5841139 / 701255
        emptyWeight[-1] = args.eosCost
        self.register_buffer('emptyWeight', emptyWeight)

    def forward(self, x: Dict[str, Tensor], y: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        ans = self.computeLoss(x, y)

        for i, aux in enumerate(x['aux']):
            ans.update({f'{k}_aux{i}': v for k, v in self.computeLoss(aux, y).items()})

        return ans

    def computeLoss(self, x: Dict[str, Tensor], y: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        """
        :param x: a dictionary containing:
            'class': a tensor of shape [batchSize, numQuery * numDecoderLayer, numClass + 1]
            'bbox': a tensor of shape [batchSize, numQuery * numDecoderLayer, 4]

        :param y: a list of dictionaries containing:
            'labels': a tensor of shape [numObjects] that stores the ground-truth classes of objects
            'boxes': a tensor of shape [numObjects, 4] that stores the ground-truth bounding boxes of objects
            represented as [centerX, centerY, w, h]

        :return: a dictionary containing classification loss, bbox loss, and gIoU loss
        """
        ids = self.matcher(x, y)
        idx = self.getPermutationIdx(ids)

        # MARK: - classification loss
        logits = x['class']

        targetClassO = torch.cat([t['labels'] for t, (_, J) in zip(y, ids)])
        targetClass = torch.full(logits.shape[:2], self.numClass, dtype=torch.int64, device=logits.device)
        targetClass[idx] = targetClassO

        classificationLoss = nn.functional.cross_entropy(logits.transpose(1, 2), targetClass, self.emptyWeight)
        classificationLoss *= self.classCost

        # MARK: - bbox loss
        # ignore boxes that has no object
        mask = targetClassO != self.numClass
        boxes = x['bbox'][idx][mask]
        targetBoxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(y, ids)], 0)[mask]

        numBoxes = len(targetBoxes) + 1e-6

        bboxLoss = nn.functional.l1_loss(boxes, targetBoxes, reduction='none')
        bboxLoss = bboxLoss.sum() / numBoxes
        bboxLoss *= self.bboxCost

        # MARK: - giou loss
        giouLoss = 1 - torch.diag(gIoU(boxCxcywh2Xyxy(boxes), boxCxcywh2Xyxy(targetBoxes)))
        giouLoss = giouLoss.sum() / numBoxes
        giouLoss *= self.giouCost

        # MARK: - compute statistics
        with torch.no_grad():
            predClass = nn.functional.softmax(logits[idx], -1).max(-1)[1]
            classMask = (predClass == targetClassO)[mask]
            iou = torch.diag(boxIoU(boxCxcywh2Xyxy(boxes), boxCxcywh2Xyxy(targetBoxes))[0])
            iou_th = [50, 75, 95]
            map_th = []
            ap = []
            for threshold in range(50, 100, 5):
                ap_th = ((iou >= threshold / 100) * classMask).sum() / numBoxes
                ap.append(ap_th)
                if threshold in iou_th:
                    map_th.append(ap_th)

            ap = torch.mean(torch.stack(ap))

        return {'classification loss': classificationLoss,
                'bbox loss': bboxLoss,
                'gIoU loss': giouLoss,
                'mAP': ap,
                'mAP_50': map_th[0],
                'mAP_75': map_th[1],
                'mAP_95': map_th[2]
                }

    @staticmethod
    def getPermutationIdx(indices: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
        batchIdx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        srcIdx = torch.cat([src for (src, _) in indices])
        return batchIdx, srcIdx

### FOR YOLO
def make_anchors(x, strides, offset=0.5):
    """
    Generate anchors from features
    """
    assert x is not None
    anchor_points, stride_tensor = [], []
    for i, stride in enumerate(strides):
        _, _, h, w = x[i].shape
        sx = torch.arange(end=w, dtype=x[i].dtype, device=x[i].device) + offset  # shift x
        sy = torch.arange(end=h, dtype=x[i].dtype, device=x[i].device) + offset  # shift y
        sy, sx = torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=x[i].dtype, device=x[i].device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)

def wh2xy(x):
    y = x.clone()
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

class YoloCriterion(nn.Module):
    def __init__(self, args, model):
        super().__init__()

        device = next(model.parameters()).device  # get model device

        m = model.head  # Head() module
        self.bce = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.device = device

        # task aligned assigner
        self.top_k = 10
        self.alpha = 0.5
        self.beta = 6.0
        self.eps = 1e-9

        self.bs = 1
        self.num_max_boxes = 0

        #Loss weights
        self.classCost = args.classCost
        self.bboxCost = args.bboxCost
        self.dlfCost = args.dlfCost

        # DFL Loss params
        self.dfl_ch = m.dfl.ch
        self.project = torch.arange(self.dfl_ch, dtype=torch.float, device=device)

    def forward(self, x: Dict[str, Tensor], y: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        return self.compute_loss(x,y)
    
    def compute_loss(self, x: Dict[str, Tensor], y: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        
        # MARK: - compute statistics
        # with torch.no_grad():

        #     print(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        #     ious = boxIoU(pred_bboxes[fg_mask], target_bboxes[fg_mask])[0]
        #     iou = torch.diag(ious)
        #     print(len(iou), len(fg_mask))
        #     iou_th = [50, 75, 95]
        #     map_th = []
        #     ap = []
        #     for threshold in range(50, 100, 5):
        #         ap_th = ((iou >= threshold / 100) * fg_mask).sum() / (len(ious) + 1e-6)
        #         ap.append(ap_th)
        #         if threshold in iou_th:
        #             map_th.append(ap_th)

        #     ap = torch.mean(torch.stack(ap))
        return {'classification loss': None, #loss_cls,
                'bbox loss': None, #loss_box,
                'DFL loss': None, #loss_dfl,
                'mAP': None, #ap,
                'mAP_50': None, #map_th[0],
                'mAP_75': None, #map_th[1],
                'mAP_95': None, #map_th[2]
                }