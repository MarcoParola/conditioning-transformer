from src.utils import boxOps
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from functools import partial
from typing import Dict, Union, List, Tuple

from .backbone import *
from src.models.mlp import MLP


class Yolos(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.yolos.backboneName == 'tiny':
            self.backbone, hidden_dim = tiny(in_chans=args.inChans, pretrained=args.yolos.pre_trained)
        elif args.yolos.backboneName == 'small':
            self.backbone, hidden_dim = small(in_chans=args.inChans, pretrained=args.yolos.pre_trained)
        elif args.yolos.backboneName == 'base':
            self.backbone, hidden_dim = base(in_chans=args.inChans, pretrained=args.yolos.pre_trained)
        elif args.yolos.backboneName == 'small_dWr':
            self.backbone, hidden_dim = small_dWr(in_chans=args.inChans, pretrained=args.yolos.pre_trained)
        else:
            raise ValueError(f'backbone {args.yolos.backboneName} not supported')
        
        self.backbone.finetune_det(
            det_token_num=args.yolos.detTokenNum, 
            img_size=args.yolos.init_pe_size, 
            mid_pe_size=args.yolos.mid_pe_size, 
            use_checkpoint=args.yolos.use_checkpoint)
        
        self.class_embed = MLP(hidden_dim, hidden_dim, args.numClass + 1, 3)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

    
    def forward(self, x: Tensor, meta=None) -> Dict[str, Union[Tensor, List[Dict[str, Tensor]]]]:
        x = self.backbone(x)
        x = x.unsqueeze(0)
        outputs_class = self.class_embed(x)
        outputs_coord = self.bbox_embed(x).sigmoid()
        
        return {'class': outputs_class[-1],
                'bbox': outputs_coord[-1],
                'aux': [{'class': oc, 'bbox': ob} for oc, ob in zip(outputs_class[:-1], outputs_coord[:-1])]}

    '''
    def forward_return_attention(self, samples: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        attention = self.backbone(samples.tensors, return_attention=True)
        return attention
    '''

class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = boxOps.boxCxcywh2Xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results

