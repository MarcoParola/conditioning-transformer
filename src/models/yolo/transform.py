import copy
import math
import random

import torch
import torch.nn.functional as F
from torch import nn


class Transformer(nn.Module):
    def __init__(self, min_size, max_size, stride=32):
        super().__init__()
        self.min_size = min_size
        self.max_size = max_size
        self.stride = stride
        
    def forward(self, images, targets=None):
        if targets is None:
            transformed = [self.transforms(img, targets) for img in images]
        else:
            targets = copy.deepcopy(targets)
            transformed = [self.transforms(img, tgt) for img, tgt in zip(images, targets)]
        
        images, targets, scale_factors = zip(*transformed)
            
        image_shapes = [img.shape[1:] for img in images]
            
        images = self.batch_images(images)
        return images, targets, scale_factors, image_shapes
    
    def transforms(self, image, target):
        image, target, scale_factor = self.resize(image, target)
        return image, target, scale_factor
        
    def resize(self, image, target):
        orig_image_shape = image.shape[1:]
        min_size = min(orig_image_shape)
        max_size = max(orig_image_shape)
        scale_factor = min(self.min_size / min_size, self.max_size / max_size)
        
        if scale_factor != 1:
            size = [round(s * scale_factor) for s in orig_image_shape]
            image = F.interpolate(image[None], size=size, mode="bilinear", align_corners=False)[0]

            if target is not None:
                box = target["boxes"]
                box[:, [0, 2]] *= size[1] / orig_image_shape[1]
                box[:, [1, 3]] *= size[0] / orig_image_shape[0]
        return image, target, scale_factor
    
    def batch_images(self, images):
        max_size = tuple(max(s) for s in zip(*(img.shape[1:] for img in images)))
        batch_size = tuple(math.ceil(m / self.stride) * self.stride for m in max_size)

        batch_shape = (len(images), 3,) + batch_size
        batched_imgs = images[0].new_full(batch_shape, 0)
        for img, pad_img in zip(images, batched_imgs):
            pad_img[:, :img.shape[1], :img.shape[2]].copy_(img)

        return batched_imgs
        
        
def sort_images(shapes, out, dim):
    shapes.sort(key=lambda x: x[dim])
    out.append(shapes.pop()[2])
    if dim == 0:
        out.append(shapes.pop()[2])
        out.append(shapes.pop(1)[2])
    else:
        out.append(shapes.pop(1)[2])
        out.append(shapes.pop()[2])
    out.append(shapes.pop(0)[2])
    if shapes:
        sort_images(shapes, out, dim)