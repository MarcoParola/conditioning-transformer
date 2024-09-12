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



class VideoCOCODataset(Dataset):
    def __init__(self, 
            root: str, 
            annotation: str, 
            numClass: int = 4,
            video_dir: str = None,
            numFrames: int = 1,
            removeBackground: bool = True,
            dummy: bool = False):
        self.root = root
        self.coco = COCO(annotation)
        self.ids = list(self.coco.imgs.keys())
        self.numClass = numClass
        self.video_dir = video_dir
        self.numFrames = numFrames
        self.removeBackground = removeBackground
        self.dummy = dummy
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

        annotations = self.loadAnnotations(imgID, imgInfo['width'], imgInfo['height'])

        if len(annotations) == 0:
            targets = {
                'boxes': torch.zeros(1, 4, dtype=torch.float32),
                'labels': torch.as_tensor([self.numClass], dtype=torch.int64)-1,}
        else:
            targets = {
                'boxes': torch.as_tensor(annotations[..., :-1], dtype=torch.float32),
                'labels': torch.as_tensor(annotations[..., -1], dtype=torch.int64)-1,}

        if self.dummy:
            frames = [image] * self.numFrames
        else:
            frames = self.loadFrames(imgID)
        frames = [self.transforms(frame, {})[0] for frame in frames]

        image, targets = self.transforms(image, targets)
        image = torch.stack([image] + frames)
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

    def loadFrames(self, imgID: int) -> List[Image.Image]:
        img_info = self.coco.imgs[imgID]
        file_name = img_info['file_name']
        file_dir = file_name.split('/')[:-1]
        file_dir = os.path.join(self.video_dir, *file_dir)
        frames = []
        for i in range(1, self.numFrames+1):
            new_frame_name = file_name.split('/')[-1].split('.')[0] + f"_frame_{-i}.jpg"
            new_frame_path = os.path.join(file_dir, new_frame_name)
            frame = Image.open(new_frame_path).convert('L')

            if self.removeBackground:
                frame = frame.crop((0, 96, frame.size[0], frame.size[1]))

            frames.append(frame)
        return frames
        


def collateFunction(batch: List[Tuple[Tensor, dict]]) -> Tuple[Tensor, Tuple[Dict[str, Tensor]]]:
    batch = tuple(zip(*batch))
    return torch.stack(batch[0]), batch[1]



# TEST MAIN
@hydra.main(config_path='../../config', config_name='config', version_base='1.1')
def main(args):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from torch.utils.data import DataLoader

    dataset = VideoCOCODataset(args.dataDir, args.trainAnnFile, args.numClass, args.trainVideoFrames, args.numFrames)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collateFunction)
    print(dataset.__len__())
    
    img, trg = dataset.__getitem__(0)
    print(img.shape, trg['boxes'].shape, trg['labels'].shape)

    for i in range(10):
        img, trg = dataset.__getitem__(i)
        fig, ax = plt.subplots(1, 6, figsize=(20, 10))
        
        for j in range(6):
            ax[j].imshow(img[j].squeeze().cpu().numpy(), cmap='gray')
            for box, label in zip(trg['boxes'], trg['labels']):
                box = box.cpu().numpy()
                x, y, w, h = box
                x = x * img[j].shape[2]
                y = y * img[j].shape[1]
                w = w * img[j].shape[2]
                h = h * img[j].shape[1]
                rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
                ax[j].add_patch(rect)

        plt.show()


        

            

    
if __name__ == '__main__':
    main()