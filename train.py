import os
import numpy as np
import torch
from torch.cuda import amp
from torch.optim import AdamW
from torch.utils.data import DataLoader
import wandb
import hydra
import gc
from tqdm import tqdm

from src.models import SetCriterion
from src.datasets import collateFunction, COCODataset
from src.utils import load_model, load_datasets
from src.utils import cast2Float
from src.utils import EarlyStopping

import torch.distributed as dist

@hydra.main(config_path="config", config_name="config")
def main(args):
    print("Starting training...")   

    wandb.init(entity=args.wandbEntity, project=args.wandbProject, config=dict(args))
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.outputDir, exist_ok=True)

    # load data
    train_dataset, val_dataset, test_dataset = load_datasets(args)
    
    train_dataloader = DataLoader(train_dataset, 
        batch_size=args.batchSize, 
        shuffle=True, 
        collate_fn=collateFunction, 
        num_workers=args.numWorkers)
    
    val_dataloader = DataLoader(val_dataset, 
        batch_size=args.batchSize, 
        shuffle=False, 
        collate_fn=collateFunction,
        num_workers=args.numWorkers)

    test_dataloader = DataLoader(test_dataset, 
        batch_size=1, 
        shuffle=False, 
        collate_fn=collateFunction,
        num_workers=args.numWorkers)
    
    # set model and criterion, load weights if available
    model = load_model(args).to(device)
    criterion = SetCriterion(args, model).to(device)    

    # separate learning rate
    paramDicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lrBackbone,},
    ]

    early_stopping = EarlyStopping(patience=args.patience)
    optimizer = AdamW(paramDicts, args.lr, weight_decay=args.weightDecay)
    prevBestLoss = np.inf
    batches = len(train_dataloader)
    scaler = amp.GradScaler()
    model.train()
    criterion.train()

    for epoch in range(args.epochs):
        wandb.log({"epoch": epoch}, step=epoch * batches)
        total_loss = 0.0
        total_metrics = None  # Initialize total_metrics
        
        # MARK: - training
        for batch, (imgs, targets) in enumerate(tqdm(train_dataloader)):
            imgs = imgs.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            # gc every 50 batches
            if batch % 700 == 0:
                torch.cuda.empty_cache()
                gc.collect()

            if args.amp:
                with amp.autocast():
                    out = model(imgs)
                out = cast2Float(out) # cast output to float to overcome amp training issue
            else:
                out = model(imgs)

            metrics = criterion(out, targets)
            
            # Initialize total_metrics on the first batch
            if total_metrics is None:
                total_metrics = {k: 0.0 for k in metrics}

            # Calculate mean values progressively
            for k, v in metrics.items():
                total_metrics[k] += v.item()

            loss = sum(v for k, v in metrics.items() if 'loss' in k)
            total_loss += loss.item()

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

        wandb.log({"train/loss": avg_loss}, step=epoch * batches)
        print(f'Epoch {epoch}, loss: {avg_loss:.8f}')

        for k, v in avg_metrics.items():
            wandb.log({f"train/{k}": v}, step=epoch * batches)
        

        
        # MARK: - validation
        model.eval()
        criterion.eval()
        with torch.no_grad():
            valMetrics = []
            losses = []
            for imgs, targets in tqdm(val_dataloader):
                imgs = imgs.to(device)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                out = model(imgs)

                metrics = criterion(out, targets)
                valMetrics.append(metrics)
                loss = sum(v for k, v in metrics.items() if 'loss' in k)
                losses.append(loss.cpu().item())

            valMetrics = {k: torch.stack([m[k] for m in valMetrics]).mean() for k in valMetrics[0]}
            avgLoss = np.mean(losses)
            wandb.log({"val/loss": avgLoss}, step=epoch * batches)
            for k,v in valMetrics.items():
                wandb.log({f"val/{k}": v.item()}, step= epoch * batches)
        

        # check if the model is estrnn-yolos, if so, predict the first 10 images of the val set
        if args.model == 'estrnn-yolos':
            for _i in range(20):
                img, target = val_dataset.__getitem__(_i)
                print(img.shape)

                pred = model.estrnn_enhancer(img.unsqueeze(0))

                print(img.shape, pred.shape)

                # get first image among the frames and sum it to the prediction
                img = img[0].squeeze().cpu().numpy()
                pred = pred.squeeze().detach().cpu().numpy()
                enhanced_img = img + pred

                # save both original and predicted images 
                from skimage.io import imsave

                # scale the image to 0-1
                enhanced_img = (enhanced_img - enhanced_img.min()) / (enhanced_img.max() - enhanced_img.min())
                img = (img - img.min()) / (img.max() - img.min())
                # convert to 0-255 
                enhanced_img = (enhanced_img * 255).astype(np.uint8)
                img = (img * 255).astype(np.uint8)
                imsave(f"{wandb.run.dir}/val_epoch{epoch}_img_{_i}.png", enhanced_img)
                imsave(f"{wandb.run.dir}/val_img_{_i}_original.png", img)
                

        model.train()
        criterion.train()

        # MARK: - save model
        if avgLoss < prevBestLoss:
            print('[+] Loss improved from {:.8f} to {:.8f}, saving model...'.format(prevBestLoss, avgLoss))
            torch.save(model.state_dict(), f'{wandb.run.dir}/best.pt')
            wandb.save(f'{wandb.run.dir}/best.pt')
            prevBestLoss = avgLoss

        # MARK: - early stopping
        if early_stopping(avgLoss):
            print('[+] Early stopping at epoch {}'.format(epoch))
            break


    

    model.eval()
    criterion.eval()
    with torch.no_grad():
        valMetrics = []
        losses = []
        for imgs, targets in tqdm(test_dataloader):
            imgs = imgs.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            out = model(imgs)

            metrics = criterion(out, targets)
            valMetrics.append(metrics)
            loss = sum(v for k, v in metrics.items() if 'loss' in k)
            losses.append(loss.cpu().item())

        valMetrics = {k: torch.stack([m[k] for m in valMetrics]).mean() for k in valMetrics[0]}
        avgLoss = np.mean(losses)
        wandb.log({"test/loss": avgLoss}, step=epoch * batches)
        for k,v in valMetrics.items():
            wandb.log({f"test/{k}": v.item()}, step=batch + epoch * batches)

    wandb.finish()

        

if __name__ == '__main__':
    main()
