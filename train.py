import os
from argparse import ArgumentParser

import numpy as np
import torch
from torch.cuda import amp
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import wandb

from models import DETR, SetCriterion
from utils.dataset import MangoDataset, collateFunction, COCODataset
from utils.misc import baseParser, MetricsLogger, saveArguments, logMetrics, cast2Float


def main(args):
    print(args)
    saveArguments(args, args.taskName)

    
    wandb.init(entity=args.wandbEntity , project=args.wandbProject, config=args)

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    os.makedirs(args.outputDir, exist_ok=True)

    # load data
    train_dataset = COCODataset(args.dataDir, args.trainAnnFile, args.numClass)
    val_dataset = COCODataset(args.dataDir, args.valAnnFile, args.numClass)
    test_dataset = COCODataset(args.dataDir, args.testAnnFile, args.numClass)

    train_dataloader = DataLoader(train_dataset, 
        batch_size=args.batchSize, 
        shuffle=True, 
        collate_fn=collateFunction, 
        pin_memory=True, 
        num_workers=args.numWorkers)

    val_dataloader = DataLoader(val_dataset, 
        batch_size=args.batchSize, 
        shuffle=False, 
        collate_fn=collateFunction,
        pin_memory=True, 
        num_workers=args.numWorkers)

    test_dataloader = DataLoader(test_dataset, 
        batch_size=args.batchSize, 
        shuffle=False, 
        collate_fn=collateFunction,
        pin_memory=True, 
        num_workers=args.numWorkers)

    # load model
    model = DETR(args).to(device)
    criterion = SetCriterion(args).to(device)

    # resume training
    '''
    if args.weight and os.path.exists(args.weight):
        print(f'loading pre-trained weights from {args.weight}')
        model.load_state_dict(torch.load(args.weight, map_location=device))
    '''

    # multi-GPU training
    if args.multi:
        model = torch.nn.DataParallel(model)

    # separate learning rate
    paramDicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lrBackbone,
        },
    ]

    optimizer = AdamW(paramDicts, args.lr, weight_decay=args.weightDecay)
    lrScheduler = StepLR(optimizer, args.lrDrop)
    prevBestLoss = np.inf
    batches = len(train_dataloader)
    logger = MetricsLogger()

    model.train()
    criterion.train()

    scaler = amp.GradScaler()

    for epoch in range(args.epochs):
        losses = []
        trainMetrics = []
        for batch, (x, y) in enumerate(train_dataloader):
            x = x.to(device)
            y = [{k: v.to(device) for k, v in t.items()} for t in y]

            if args.amp:
                with amp.autocast():
                    out = model(x)
                # cast output to float to overcome amp training issue
                out = cast2Float(out)
            else:
                out = model(x)

            metrics = criterion(out, y)
            trainMetrics.append(metrics)

            loss = sum(v for k, v in metrics.items() if 'loss' in k)
            losses.append(loss.cpu().item())

            # MARK: - print & save training details
            print(f'Epoch {epoch} | {batch + 1} / {batches}')
            logMetrics({k: v for k, v in metrics.items() if 'aux' not in k})
            logger.step(metrics, epoch, batch)
            #for k,v in metrics.items():
            #    wandb.log(f"train/{k}", v.item())

            # MARK: - backpropagation
            optimizer.zero_grad()
            if args.amp:
                scaler.scale(loss).backward()
                if args.clipMaxNorm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clipMaxNorm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.clipMaxNorm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clipMaxNorm)
                optimizer.step()

        lrScheduler.step()
        logger.epochEnd(epoch)
        avgLoss = np.mean(losses)
        wandb.log({"train/loss": avgLoss}, step=epoch * batches)
        print(f'Epoch {epoch}, loss: {avgLoss:.8f}')

        trainMetrics = {k: torch.stack([m[k] for m in trainMetrics]).mean() for k in trainMetrics[0]}
        for k,v in trainMetrics.items():
            wandb.log({f"train/{k}": v.item()}, step=epoch * batches)
        

        # MARK: - validation
        # check if it's the end of the epoch
        if batch == batches - 1:
            model.eval()
            criterion.eval()
            with torch.no_grad():
                valMetrics = []
                for x, y in val_dataloader:
                    x = x.to(device)
                    y = [{k: v.to(device) for k, v in t.items()} for t in y]

                    out = model(x)
                    metrics = criterion(out, y)
                    valMetrics.append(metrics)

                valMetrics = {k: torch.stack([m[k] for m in valMetrics]).mean() for k in valMetrics[0]}
                logMetrics(valMetrics)
                for k,v in valMetrics.items():
                    wandb.log({f"val/{k}": v.item()}, step=batch + epoch * batches)
                wandb.log(valMetrics, step=batch + epoch * batches)
            model.train()
            criterion.train()

        if avgLoss < prevBestLoss:
            print('[+] Loss improved from {:.8f} to {:.8f}, saving model...'.format(prevBestLoss, avgLoss))
            if not os.path.exists(args.outputDir):
                os.mkdir(args.outputDir)

            try:
                stateDict = model.module.state_dict()
            except AttributeError:
                stateDict = model.state_dict()
            torch.save(stateDict, f'{args.outputDir}/{args.taskName}.pt')
            prevBestLoss = avgLoss
            logger.addScalar('Model', avgLoss, epoch)
        logger.flush()
    logger.close()


if __name__ == '__main__':
    parser = ArgumentParser('python3 train.py', parents=[baseParser()])

    # MARK: - training config
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--lrBackbone', default=1e-5, type=float)
    parser.add_argument('--batchSize', default=8, type=int)
    parser.add_argument('--weightDecay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--lrDrop', default=1000, type=int)
    parser.add_argument('--clipMaxNorm', default=.1, type=float)

    # MARK: - loss
    parser.add_argument('--classCost', default=1., type=float)
    parser.add_argument('--bboxCost', default=5., type=float)
    parser.add_argument('--giouCost', default=2., type=float)
    parser.add_argument('--eosCost', default=.1, type=float)

    # MARK: - dataset
    parser.add_argument('--dataDir', default='./data', type=str)
    parser.add_argument('--trainAnnFile', default='./data/Train.json', type=str)
    parser.add_argument('--valAnnFile', default='./data/Valid.json', type=str)
    parser.add_argument('--testAnnFile', default='./data/Test.json', type=str)

    # MARK: - miscellaneous
    parser.add_argument('--outputDir', default='./checkpoint', type=str)
    parser.add_argument('--taskName', default='mango', type=str)
    parser.add_argument('--numWorkers', default=8, type=int)
    parser.add_argument('--multi', default=False, type=bool)
    parser.add_argument('--amp', default=False, type=bool)

    # MARK: - wandb
    parser.add_argument('--wandbEntity', default='marcoparola', type=str)
    parser.add_argument('--wandbProject', default='conditioning-transformer', type=str)

    args = parser.parse_args()
    main(args)
