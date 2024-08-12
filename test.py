import os
import numpy as np
import torch
from torch.cuda import amp
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import wandb
import hydra
from tqdm import tqdm

from src.models import SetCriterion
from src.datasets import collateFunction, COCODataset
from src.utils import load_model, load_datasets
from src.utils.misc import cast2Float
from src.utils.utils import load_weights


@hydra.main(config_path="config", config_name="config")
def main(args):
    
    args.wandbProject = args.wandbProject + '_eval'
    wandb.init(entity=args.wandbEntity , project=args.wandbProject, config=dict(args))
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.outputDir, exist_ok=True)

    # load data
    train_dataset, val_dataset, test_dataset = load_datasets(args)
    test_dataloader = DataLoader(test_dataset, 
        batch_size=1, 
        shuffle=False, 
        collate_fn=collateFunction,
        num_workers=args.numWorkers)
    
    # load model
    criterion = SetCriterion(args).to(device)
    model = load_model(args).to(device)  
    
    # multi-GPU training
    if args.multi:
        model = torch.nn.DataParallel(model)

    model.eval()
    criterion.eval()
    with torch.no_grad():
        testMetrics = []

        for batch, (imgs, targets) in enumerate(tqdm(test_dataloader)):
            imgs = imgs.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            out = model(imgs)
            metrics = criterion(out, targets)
            testMetrics.append(metrics)

        testMetrics = {k: torch.stack([m[k] for m in testMetrics]).mean() for k in testMetrics[0]}
        for k,v in testMetrics.items():
            wandb.log({f"test/{k}": v.item()}, step=0)
    
    wandb.finish()

   
if __name__ == '__main__':
    main()
