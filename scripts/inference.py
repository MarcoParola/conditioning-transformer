import os
import numpy as np
import torch
from torch.cuda import amp
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import hydra
from tqdm import tqdm

from src.models import SetCriterion
from src.datasets import collateFunction, COCODataset
from src.utils import load_model
from src.utils.misc import baseParser, cast2Float
from src.utils.utils import load_weights
from src.utils.boxOps import boxCxcywh2Xyxy, gIoU, boxIoU
from src.models.matcher import HungarianMatcher


@hydra.main(config_path="../config", config_name="config")
def main(args):
    
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.outputDir, exist_ok=True)

    matcher = HungarianMatcher(args.classCost, args.bboxCost, args.giouCost)

    # load data and model
    test_dataset = COCODataset(args.dataDir, args.testAnnFile, args.numClass, args.scaleMetadata)
    model = load_model(args).to(device)  
    
    # multi-GPU training
    if args.multi:
        model = torch.nn.DataParallel(model)

    model.eval()
    with torch.no_grad():
        for i in range(test_dataset.__len__()):

            img, metadata, target = test_dataset.__getitem__(i)           
            
            imgID = test_dataset.ids[i]
            imgInfo = test_dataset.coco.imgs[imgID]  
            print('\nfile', imgInfo['file_name'])

            img = img.unsqueeze(0).to(device)
            metadata = metadata.unsqueeze(0).to(device)
            target = [{k: v.to(device) for k, v in target.items()}]


            out = model(img, metadata)
            logits = out['class']
 
            ids = matcher(out, target)
            idx = SetCriterion.getPermutationIdx(ids)

            targetClassO = torch.cat([t['labels'] for t, (_, J) in zip(target, ids)])
            targetClass = torch.full(logits.shape[:2], args.numClass, dtype=torch.int64, device=logits.device)
            targetClass[idx] = targetClassO

            # ignore boxes that has no object
            mask = targetClassO != args.numClass
            boxes = out['bbox'][idx][mask]
            targetBoxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(target, ids)], 0)[mask]

            # plot the image
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            fig, ax = plt.subplots()
            ax.imshow(img.squeeze(0).cpu().permute(1, 2, 0))

            for j in range(len(target[0]['boxes'])):
                x, y, w, h = target[0]['boxes'][j].tolist()
                x, y, w, h = x*img.size(3), y*img.size(2), w*img.size(3), h*img.size(2)
                lbl = target[0]['labels'][j].item()
                print(lbl, [x, y, w, h])
                rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
            

            # MARK: - compute statistics
            with torch.no_grad():
                predClass = torch.nn.functional.softmax(logits[idx], -1).max(-1)[1]
                classMask = (predClass == targetClassO)[mask]
                ious = torch.diag(boxIoU(boxCxcywh2Xyxy(boxes), boxCxcywh2Xyxy(targetBoxes))[0])
                for  lbl, bbox in zip(predClass.tolist(), boxes.tolist()):
                    print(lbl, bbox)
                    x, y, w, h = bbox
                    x, y, w, h = x*img.size(3), y*img.size(2), w*img.size(3), h*img.size(2)
                    rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='g', facecolor='none')
                    ax.add_patch(rect)
            plt.show()
            



   
if __name__ == '__main__':
    main()