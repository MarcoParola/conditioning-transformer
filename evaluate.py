import os
import numpy as np
import torch
from torch.cuda import amp
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import wandb
import hydra

from models import DETR, SetCriterion
from utils.dataset import collateFunction, COCODataset
from utils.misc import baseParser, cast2Float
from tqdm import tqdm

@hydra.main(config_path="config", config_name="config")
def main(args):
    print(args)
    #wandb.init(entity=args.wandbEntity , project=args.wandbProject, config=args)

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.outputDir, exist_ok=True)

    # load data
    test_dataset = COCODataset(args.dataDir, args.testAnnFile, args.numClass)
    test_dataloader = DataLoader(test_dataset, 
        batch_size=args.batchSize, 
        shuffle=False, 
        collate_fn=collateFunction,
        #pin_memory=True, 
        num_workers=args.numWorkers)
    
    # load model
    model = DETR(args).to(device)
    criterion = SetCriterion(args).to(device)
    if args.weight:
        print(f'loading pre-trained weights from {args.weight}')
        model_path = os.path.join(args.currentDir, args.weight)
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print('no pre-trained weights found, training from scratch...')
        return
    
    # multi-GPU training
    if args.multi:
        model = torch.nn.DataParallel(model)

    
    model.eval()
    criterion.eval()
    with torch.no_grad():
        testMetrics = []
        i=0 

        # loop through test dataset using tqdm

        for data in tqdm(test_dataloader):
            x,y = data
            i+=1
            x = x.to(device)
            y = [{k: v.to(device) for k, v in t.items()} for t in y]
            out = model(x)
            metrics = criterion(out, y)
            testMetrics.append(metrics)

        testMetrics = {k: torch.stack([m[k] for m in testMetrics]).mean() for k in testMetrics[0]}
        for k,v in testMetrics.items():
            print(f"test/{k}: {v.item()}")
    
        '''
        wandb.log({"val/loss": avgLoss}, step=epoch * batches)
        for k,v in valMetrics.items():
            wandb.log({f"val/{k}": v.item()}, step=batch + epoch * batches)
        '''



        

if __name__ == '__main__':
    main()
