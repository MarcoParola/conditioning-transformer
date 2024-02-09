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
import hydra


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



class COCODataset(Dataset):
    #def __init__(self, root: str, annotation: str, targetHeight: int, targetWidth: int, numClass: int):
    def __init__(self, root: str, annotation: str, numClass: int, scaling_thresholds: Dict[str, Tuple[float, float]] = None):
        self.root = root
        self.coco = COCO(annotation)
        self.ids = list(self.coco.imgs.keys())
        self.numClass = numClass
        self.scaling_thresholds = scaling_thresholds
        self.transforms = T.Compose([
            T.ToTensor()
        ])

        self.newIndex = {}
        classes = []
        for i, (k, v) in enumerate(self.coco.cats.items()):
            self.newIndex[k] = i
            classes.append(v['name'])

        '''
        with open('./checkpoint/classes.txt', 'w') as f:
            f.write(str(classes))
        '''

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Tuple[Tensor, dict]:
        imgID = self.ids[idx]
        imgInfo = self.coco.imgs[imgID]        
        imgPath = os.path.join(self.root, imgInfo['file_name'])
        image = Image.open(imgPath).convert('RGB')

        annotations = self.loadAnnotations(imgID, imgInfo['width'], imgInfo['height'])
        metadata = self.loadMetadata(imgID)

        if len(annotations) == 0:
            targets = {
                'boxes': torch.zeros(1, 4, dtype=torch.float32),
                'labels': torch.as_tensor([self.numClass], dtype=torch.int64),}
        else:
            targets = {
                'boxes': torch.as_tensor(annotations[..., :-1], dtype=torch.float32),
                'labels': torch.as_tensor(annotations[..., -1], dtype=torch.int64),}

        image, targets = self.transforms(image, targets)

        # TODO 
        #return image, metadata, targets
        return image, targets

    def loadAnnotations(self, imgID: int, imgWidth: int, imgHeight: int) -> np.ndarray:
        ans = []
        for annotation in self.coco.imgToAnns[imgID]:
            cat = self.newIndex[annotation['category_id']]
            bbox = annotation['bbox']

            # convert from [x1, y1, x2, y2] to [x, y, w, h]
            bbox[2] -= bbox[0]
            bbox[3] -= bbox[1]
            bbox = [val / imgHeight if i % 2 else val / imgWidth for i, val in enumerate(bbox)]
            ans.append(bbox + [cat])

        return np.asarray(ans)


    def loadMetadata(self, imgID: int) -> np.ndarray:
        imgInfo = self.coco.imgs[imgID]
        metadata = imgInfo['meta']
        timestamp = imgInfo['date_captured']
        metadata['Hour'] = int(timestamp.split('T')[1].split(':')[0])
        metadata['Month'] = int(timestamp.split('T')[0].split('-')[1])
        metadata = self.scale_harborfront_metadata(metadata)
        metadata = np.array(list(metadata.values()))
        return metadata

    def scale_harborfront_metadata(self, metadata: Dict[str, float]) -> Dict[str, float]:
        metadata = metadata.copy()
        for key, (min_val, max_val) in self.scaling_thresholds.items():
            metadata[key] = (metadata[key] - min_val) / (max_val - min_val)
        return metadata


def collateFunction(batch: List[Tuple[Tensor, dict]]) -> Tuple[Tensor, Tuple[Dict[str, Tensor]]]:
    batch = tuple(zip(*batch))
    return torch.stack(batch[0]), batch[1]



# TEST MAIN
@hydra.main(config_path='../config', config_name='config')
def main(args):
    data_folder = 'data'
    data_folder = os.path.join(args.currentDir, data_folder)
    data_file = 'data/small/Test.json'
    data_file = os.path.join(args.currentDir, data_file)
    num_classes = 4
    dataset = COCODataset(data_folder, data_file, num_classes, args.scaleMetadata)  
   
    print(dataset.__len__())
    # plot 10 images using matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    # plot 10 images using matplotlib and draw the corresponding annotations with plt.rectangle
    for i in range(10):
        image, target = dataset.__getitem__(i)
        
        fig, ax = plt.subplots()
        ax.imshow(image.permute(1, 2, 0))

        for i in range(len(target['boxes'])):
            img_w, img_h = image.size(2), image.size(1)
            x, y, w, h = target['boxes'][i]
            x = x * img_w
            y = y * img_h
            w = w * img_w
            h = h * img_h
            rect = patches.Rectangle((x,y), w,h, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        plt.show()
    
if __name__ == '__main__':
    main()