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


class COCODataset(Dataset):
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
            metadata[key] = (metadata[key] - 0.5) * 4 # shift the range to [-2, 2]
        return metadata

def collateFunction(batch: List[Tuple[Tensor, Tensor, dict]]) -> Tuple[Tensor, Tensor, Tuple[Dict[str, Tensor]]]:
    batch = tuple(zip(*batch))
    return torch.stack(batch[0]), torch.stack(batch[1]), batch[2]



# TEST MAIN
@hydra.main(config_path='../../config', config_name='config')
def main(args):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from torch.utils.data import DataLoader

    num_classes = 4
    data_folder = 'data'
    data_file = 'data/small/Test.json'
    data_folder = os.path.join(args.currentDir, data_folder)
    data_file = os.path.join(args.currentDir, data_file)
    dataset = COCODataset(data_folder, data_file, num_classes, args.scaleMetadata)  
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collateFunction)
    print(dataset.__len__())
    
    #for images, metadata, targets in dataloader:
    #    print(type(images), type(metadata), type(targets))
    #    print(images.shape, metadata.shape, targets[0]['boxes'].shape, targets[0]['labels'].shape)
        
    for j in range(dataset.__len__()):
        image, meta, target = dataset.__getitem__(j)
        fig, ax = plt.subplots()
        ax.imshow(image.permute(1, 2, 0))

        for i in range(len(target['boxes'])):
            img_w, img_h = image.size(2), image.size(1)
            x, y, w, h = target['boxes'][i]
            x, y = x*img_w, y*img_h 
            w, h = w*img_w, h*img_h
            rect = patches.Rectangle((x,y), w,h, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        plt.show()
        #plt.close(fig)
    
if __name__ == '__main__':
    main()