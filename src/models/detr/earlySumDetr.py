import os
from typing import Dict, Union, List, Tuple
import torch
from torch import nn, Tensor
from torch.quantization import quantize_dynamic

from src.utils.misc import PostProcess
from .backbone import buildBackbone
from src.models.transformer import Transformer
from src.models.mlp import MLP


class EarlySummationDETR(nn.Module):
    def __init__(self, args):
        super(EarlySummationDETR, self).__init__()

        self.backbone = buildBackbone(args)

        self.reshape = nn.Conv2d(self.backbone.backbone.outChannels, args.hiddenDims, 1)

        self.metaProjection = nn.Linear(args.numMetadata, args.hiddenDims * 12 * 9) # TODO parameterize
        self.projection_size = (args.hiddenDims, 9, 12)

        self.transformer = Transformer(args.hiddenDims, args.numHead, args.numEncoderLayer, args.numDecoderLayer,
                                       args.dimFeedForward, args.dropout)

        self.dummy = args.dummy

        self.queryEmbed = nn.Embedding(args.numQuery, args.hiddenDims)
        self.classEmbed = nn.Linear(args.hiddenDims, args.numClass + 1)
        self.bboxEmbed = MLP(args.hiddenDims, args.hiddenDims, 4, 3)

    def forward(self, x: Tensor, meta: Tensor) -> Dict[str, Union[Tensor, List[Dict[str, Tensor]]]]:
        """
        :param x: tensor of shape [batchSize, 3, imageHeight, imageWidth].

        :param meta: tensor of shape [batchSize, n_channels].

        :return: a dictionary with the following elements:
            - class: the classification results for all queries with shape [batchSize, numQuery, numClass + 1].
                     +1 stands for no object class.
            - bbox: the normalized bounding box for all queries with shape [batchSize, numQuery, 4],
                    represented as [centerX, centerY, width, height].

        mask: provides specified elements in the key to be ignored by the attention.
              the positions with the value of True will be ignored
              while the position with the value of False will be unchanged.
              Since I am only training with images of the same shape, the mask should be all False.
              Modify the mask generation method if you would like to enable training with arbitrary shape.
        """
        features, (pos, mask) = self.backbone(x)
        features = self.reshape(features)
        if self.dummy:
            # set each element of the meta to 0
            meta = torch.ones_like(meta)
        meta = self.metaProjection(meta)
        meta = meta.view(-1, *self.projection_size)

        # implementing the early summation strategy
        features = features + meta
        
        N = features.shape[0]
        features = features.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        pos = pos.flatten(2).permute(2, 0, 1)
        query = self.queryEmbed.weight
        query = query.unsqueeze(1).repeat(1, N, 1)

        out = self.transformer(features, mask, query, pos)

        outputsClass = self.classEmbed(out)
        outputsCoord = self.bboxEmbed(out).sigmoid()

        return {'class': outputsClass[-1],
                'bbox': outputsCoord[-1],
                'aux': [{'class': oc, 'bbox': ob} for oc, ob in zip(outputsClass[:-1], outputsCoord[:-1])]}


class EarlySummationDETRWrapper(nn.Module):
    """ A simple EarlySummationDETR wrapper that allows torch.jit to trace the module since dictionary output is not supported yet """

    def __init__(self, early_sum_detr, postProcess):
        super(EarlySummationDETRWrapper, self).__init__()

        self.early_sum_detr = early_sum_detr
        self.postProcess = postProcess

    def forward(self, x: Tensor, imgSize: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        :param x: batch images of shape [batchSize, 3, args.targetHeight, args.targetWidth] where batchSize equals to 1
        If tensor with batchSize larger than 1 is passed in, only the first image prediction will be returned

        :param imgSize: tensor of shape [batchSize, imgWidth, imgHeight]

        :return: the first image prediction in the following order: scores, labels, boxes.
        """

        out = self.early_sum_detr(x)
        out = self.postProcess(out, imgSize)[0]
        return out['scores'], out['labels'], out['boxes']


@torch.no_grad()
def buildInferenceModel(args, quantize=False):
    assert os.path.exists(args.weight), 'inference model should have pre-trained weight'
    device = torch.device(args.device)

    model = EarlySummationDETR(args).to(device)
    model.load_state_dict(torch.load(args.weight, map_location=device))

    postProcess = PostProcess().to(device)

    wrapper = EarlySummationDETRWrapper(model, postProcess).to(device)
    wrapper.eval()

    if quantize:
        wrapper = quantize_dynamic(wrapper, {nn.Linear})

    print('optimizing model for inference...')
    return torch.jit.trace(wrapper, (torch.rand(1, 3, args.targetHeight, args.targetWidth).to(device),
                                     torch.as_tensor([args.targetWidth, args.targetHeight]).unsqueeze(0).to(device)))
