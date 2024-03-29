import os
from glob import glob
from typing import List, Tuple, Dict
import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch import Tensor
from torch.utils.data.dataset import Dataset
import utils.transforms as T


class YOLODataset(Dataset):
    def __init__(self, root: str, targetHeight: int, targetWidth: int, numClass: int, train: bool = True):
        """
        :param root: should contain .jpg files and corresponding .txt files
        :param targetHeight: desired height for model input
        :param targetWidth: desired width for model input
        :param numClass: number of classes in the given dataset
        """
        self.cache = {}

        imagePaths = glob(os.path.join(root, '*.jpg'))
        for path in imagePaths:
            name = path.split('/')[-1].split('.jpg')[0]
            self.cache[path] = os.path.join(root, f'{name}.txt')

        self.paths = list(self.cache.keys())

        self.targetHeight = targetHeight
        self.targetWidth = targetWidth
        self.numClass = numClass

        if train:
            self.transforms = T.Compose([
                T.RandomOrder([
                    T.RandomHorizontalFlip(),
                    T.RandomVerticalFlip(),
                    T.RandomSizeCrop(numClass)
                ]),
                T.Resize((targetHeight, targetWidth)),
                T.ColorJitter(brightness=.2, contrast=.1, saturation=.1, hue=0),
                T.Normalize()
            ])
        else:
            self.transforms = T.Compose([
                T.Resize((targetHeight, targetWidth)),
                T.Normalize()
            ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[Tensor, dict]:
        imgPath = self.paths[idx]
        annPath = self.cache[imgPath]

        image = Image.open(imgPath).convert('RGB')
        annotations = self.loadAnnotations(annPath)

        if len(annotations) == 0:
            targets = {
                'boxes': torch.zeros(1, 4, dtype=torch.float32),
                'labels': torch.as_tensor([self.numClass], dtype=torch.int64),
            }
        else:
            targets = {
                'boxes': torch.as_tensor(annotations[..., :-1], dtype=torch.float32),
                'labels': torch.as_tensor(annotations[..., -1], dtype=torch.int64),
            }

        image, targets = self.transforms(image, targets)

        # # MARK - debug
        # from PIL import ImageDraw
        # import torchvision
        # mean = torch.as_tensor([0.485, 0.456, 0.406])[:, None, None]
        # std = torch.as_tensor([0.229, 0.224, 0.225])[:, None, None]
        # im = (image * std) + mean
        # im = torchvision.transforms.ToPILImage()(im)
        # draw = ImageDraw.Draw(im)
        # for i, box in enumerate(targets['boxes']):
        #     w, h = im.size
        #     x0 = (box[0] - box[2] / 2) * w
        #     y0 = (box[1] - box[3] / 2) * h
        #     x1 = (box[0] + box[2] / 2) * w
        #     y1 = (box[1] + box[3] / 2) * h
        #     draw.rectangle((x0, y0, x1, y1))
        #     draw.text((x0, y0), f'class: {targets["labels"][i]}')
        # im = torchvision.transforms.Resize((self.targetHeight, self.targetWidth))(im)
        # im.show()

        return image, targets

    @staticmethod
    def loadAnnotations(path: str) -> np.ndarray:
        """
        :param path: annotation file path
                -> each line should be in the format of [class centerX centerY width height]

        :return: an array of objects of shape [centerX, centerY, width, height, class]
        """
        if not os.path.exists(path): return np.asarray([])

        ans = []
        with open(path, 'r') as f:
            for line in f.readlines():
                line = line.split(' ')
                c, (x, y, w, h) = int(line[0]), list(map(float, line[1:]))
                ans.append([x, y, w, h, c])
        return np.asarray(ans)


