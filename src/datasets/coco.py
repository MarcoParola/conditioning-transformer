import os
from typing import List, Tuple, Dict
import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch import Tensor
from torch.utils.data.dataset import Dataset
import src.utils.transforms as T
import hydra
from collections import defaultdict



class COCODataset(Dataset):
    def __init__(self, 
            root: str, 
            annotation: str, 
            numClass: int = 4, 
            removeBackground: bool = True,):
        self.root = root
        self.coco = COCO(annotation)
        self.ids = list(self.coco.imgs.keys())
        self.numClass = numClass
        self.removeBackground = removeBackground
        self.transforms = T.Compose([
            T.ToTensor()
        ])

        self.newIndex = {}
        classes = []
        for i, (k, v) in enumerate(self.coco.cats.items()):
            self.newIndex[k] = i
            classes.append(v['name'])

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Tuple[Tensor, dict]:
        imgID = self.ids[idx]
        imgInfo = self.coco.imgs[imgID]        
        imgPath = os.path.join(self.root, imgInfo['file_name'])
        image = Image.open(imgPath).convert('L')

        # crop the pil image by removing the top part, remove the first 96 pixels
        if self.removeBackground:
            image = image.crop((0, 96, image.size[0], image.size[1]))

        annotations = self.loadAnnotations(imgID, imgWidth=image.size[0], imgHeight=image.size[1])

        if len(annotations) == 0:
            targets = {
                'boxes': torch.zeros(1, 4, dtype=torch.float32),
                'labels': torch.as_tensor([self.numClass], dtype=torch.int64)-1,}
        else:
            targets = {
                'boxes': torch.as_tensor(annotations[..., :-1], dtype=torch.float32),
                'labels': torch.as_tensor(annotations[..., -1], dtype=torch.int64)-1,}

        image, targets = self.transforms(image, targets)

        return image, targets


    def loadAnnotations(self, imgID: int, imgWidth: int, imgHeight: int) -> np.ndarray:
        ans = []
        for annotation in self.coco.imgToAnns[imgID]:
            cat = self.newIndex[annotation['category_id']]
            bbox = annotation['bbox']

            # adapt the annotations to the new image size (it has been removed the top part of the image of 96 pixels)
            if self.removeBackground:
                if bbox[1] + bbox[3] < 96:
                    continue
                  
                bbox[1] -= 96
                if bbox[1] < 0:
                    bbox[3] -= 96 - bbox[1]
                    bbox[1] = 0
                    if bbox[3] < 5:
                        continue

            bbox = [val / imgHeight if i % 2 else val / imgWidth for i, val in enumerate(bbox)]
            ans.append(bbox + [cat])

        return np.asarray(ans)



def collateFunction(batch: List[Tuple[Tensor, dict]]) -> Tuple[Tensor, Tuple[Dict[str, Tensor]]]:
    batch = tuple(zip(*batch))
    return torch.stack(batch[0]), batch[1]



class metaCOCODataset(Dataset):
    #def __init__(self, root: str, annotation: str, targetHeight: int, targetWidth: int, numClass: int):
    def __init__(self, root: str, annotation: str, numClass: int, scaling_thresholds: Dict[str, Tuple[float, float]] = None, removeBackground: bool = True):
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
        return image, metadata, targets


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
        metadata = torch.as_tensor(list(metadata.values()))
        return metadata

    def scale_harborfront_metadata(self, metadata: Dict[str, float]) -> Dict[str, float]:
        metadata = metadata.copy()
        for key, (min_val, max_val) in self.scaling_thresholds.items():
            metadata[key] = (metadata[key] - min_val) / (max_val - min_val)
            # shift the range to [-2, 2]
            #metadata[key] = (metadata[key] - 0.5) * 4
        return metadata



def meta_data_collateFunction(batch: List[Tuple[Tensor, Tensor, dict]]) -> Tuple[Tensor, Tensor, Tuple[Dict[str, Tensor]]]:
    batch = tuple(zip(*batch))
    return torch.stack(batch[0]), torch.stack(batch[1]), batch[2]



# TEST MAIN
@hydra.main(config_path='../../config', config_name='config', version_base='1.1')
def main(args):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from torch.utils.data import DataLoader

    num_classes = 4
    data_folder = 'data'
    data_file = 'data/new/Test.json'
    data_folder = os.path.join(args.currentDir, data_folder)
    data_file = os.path.join(args.currentDir, data_file)
    dataset = COCODataset(data_folder, data_file, num_classes, removeBackground=args.cropBackground)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collateFunction)
    print(dataset.__len__())
    img, trg = dataset.__getitem__(0)
    print(img.shape, trg['boxes'].shape, trg['labels'].shape)

    
    '''
    for i in range(50):
        img, target = dataset.__getitem__(i)
        print(img.shape, target['boxes'].shape, target['labels'].shape)
    
    # test for checking the size of the metadata sequence
    sizes = [0] * 100
    from tqdm import tqdm
    for j in tqdm(range(dataset.__len__())):
        image, meta, target = dataset.__getitem__(j)
        print(image.shape, meta.shape, target['boxes'].shape, target['labels'].shape)
    print(sizes)
    
    
    
    for images, metadata, targets in dataloader:
        print(type(images), type(metadata), type(targets))
        print(metadata)
        print(images.shape, metadata.shape, targets[0]['boxes'].shape, targets[0]['labels'].shape)
    '''
    
    

    testing_dir = 'outputs/test/'
    if not os.path.exists(testing_dir):
        os.makedirs(testing_dir)

    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

    for j in range(20):
        image, target = dataset.__getitem__(j)
        fig, ax = plt.subplots()
        ax.imshow(image.permute(1, 2, 0), cmap='gray')
        ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])

        # add title using the image name
        imgID = dataset.ids[j]
        imgInfo = dataset.coco.imgs[imgID]

        print(len(target['boxes']))

        for i in range(len(target['boxes'])):
            img_w, img_h = image.size(2), image.size(1)
            x, y, w, h = target['boxes'][i]
            x, y = x*img_w, y*img_h 
            w, h = w*img_w, h*img_h
            rect = patches.Rectangle((x,y), w,h, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        # zoom
        #x1,x2,y1,y2 = 70, 112, 72, 98 # img0
        #x1,x2,y1,y2 = 70, 127, 69, 105 # img1
        x1,x2,y1,y2 = 70, 140, 69, 114 # img12
        rect = patches.Rectangle((x1,y1), x2-x1, y2-y1, linewidth=3, edgecolor='blue', facecolor='none')
        ax.add_patch(rect)       

        axins = zoomed_inset_axes(ax, zoom=3.2, loc='upper right')
        axins.imshow(image.permute(1, 2, 0), cmap='gray')
        axins.set_xlim(x1, x2)
        axins.set_ylim(y2, y1)

        # draw bboxes in the zoomed area
        for i in range(len(target['boxes'])):
            img_w, img_h = image.size(2), image.size(1)
            x, y, w, h = target['boxes'][i]
            x, y = x*img_w, y*img_h 
            w, h = w*img_w, h*img_h
            rect = patches.Rectangle((x,y), w,h, linewidth=1, edgecolor='r', facecolor='none')
            axins.add_patch(rect)

        axins.set_xticks([])
        axins.set_yticks([])

        #mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="blue")

        plt.savefig(f'{testing_dir}img_{j}.png')
        plt.close(fig)
    


 

    
        
    
if __name__ == '__main__':
    main()