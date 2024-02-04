import os
from argparse import ArgumentParser
import configparser

import numpy as np
import torch
from torch.cuda import amp
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import wandb

from models import DETR, SetCriterion
from utils.dataset import MangoDataset, collateFunction, COCODataset
from utils.misc import baseParser, cast2Float


def main(args):
    print(args)
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
        #pin_memory=True, 
        num_workers=args.numWorkers)

    val_dataloader = DataLoader(val_dataset, 
        batch_size=args.batchSize, 
        shuffle=False, 
        collate_fn=collateFunction,
        #pin_memory=True, 
        num_workers=args.numWorkers)

    test_dataloader = DataLoader(test_dataset, 
        batch_size=args.batchSize, 
        shuffle=False, 
        collate_fn=collateFunction,
        #pin_memory=True, 
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
    
    scaler = amp.GradScaler()

    
    print(f'Number of batches: {batches}')
    val_dataloader1 = DataLoader(val_dataset, 
        batch_size=16, 
        shuffle=False, 
        collate_fn=collateFunction,
        #pin_memory=True, 
        num_workers=args.numWorkers)
    batches1 = len(val_dataloader1)
    print(f'Number of batches: {batches1}')

    model.train()
    criterion.train()
    '''
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
        avgLoss = np.mean(losses)
        wandb.log({"train/loss": avgLoss}, step=epoch * batches)
        print(f'Epoch {epoch}, loss: {avgLoss:.8f}')

        trainMetrics = {k: torch.stack([m[k] for m in trainMetrics]).mean() for k in trainMetrics[0]}
        for k,v in trainMetrics.items():
            wandb.log({f"train/{k}": v.item()}, step=epoch * batches)
        '''

    for epoch in range(args.epochs):
        total_loss = 0.0
        total_metrics = None  # Initialize total_metrics

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
            print({k: v for k, v in metrics.items() if 'aux' not in k})

            # Initialize total_metrics on the first batch
            if total_metrics is None:
                total_metrics = {k: 0.0 for k in metrics}

            # Calculate mean values progressively
            for k, v in metrics.items():
                total_metrics[k] += v.item()

            loss = sum(v for k, v in metrics.items() if 'loss' in k)
            total_loss += loss.item()

            # MARK: - print & save training details
            print(f'Epoch {epoch} | {batch + 1} / {batches}')

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

        # Calculate average loss and metrics
        avg_loss = total_loss / len(train_dataloader)
        avg_metrics = {k: v / len(train_dataloader) for k, v in total_metrics.items()}

        lrScheduler.step()
        wandb.log({"train/loss": avg_loss}, step=epoch * batches)
        print(f'Epoch {epoch}, loss: {avg_loss:.8f}')

        for k, v in avg_metrics.items():
            wandb.log({f"train/{k}": v}, step=epoch * batches)


        # MARK: - validation
        if batch == batches - 1:
            model.eval()
            criterion.eval()
            with torch.no_grad():
                valMetrics = []
                losses = []
                for x, y in val_dataloader:
                    x = x.to(device)
                    y = [{k: v.to(device) for k, v in t.items()} for t in y]
                    out = model(x)

                    metrics = criterion(out, y)
                    valMetrics.append(metrics)
                    loss = sum(v for k, v in metrics.items() if 'loss' in k)
                    losses.append(loss.cpu().item())

                valMetrics = {k: torch.stack([m[k] for m in valMetrics]).mean() for k in valMetrics[0]}
                avgLoss = np.mean(losses)
                wandb.log({"val/loss": avgLoss}, step=epoch * batches)
                for k,v in valMetrics.items():
                    wandb.log({f"val/{k}": v.item()}, step=batch + epoch * batches)

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


if __name__ == '__main__':
    parser = ArgumentParser('python3 train.py', parents=[baseParser()])
    parser.add_argument("-c", "--config_file", type=str, help='Config file')

    

    args = parser.parse_args()

    if args.config_file:
        config = configparser.ConfigParser()
        config.read(args.config_file)
        defaults = {}
        defaults.update(dict(config.items("Defaults")))
        parser.set_defaults(**defaults)
        args = parser.parse_args() 
    main(args)
