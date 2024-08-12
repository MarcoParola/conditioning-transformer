import torch 
import os

def load_weights(model, weight_path, device):

    if weight_path:
        print(f'loading pre-trained weights from {weight_path}')
        model.load_state_dict(torch.load(weight_path, map_location=device))
    else:
        print('no pre-trained weights found, training from scratch...')
    return model
        
def load_model(args):
    if args.model == 'early-sum-detr':
        from src.models.detr.earlySumDetr import EarlySummationDETR
        model = EarlySummationDETR(args)
    elif args.model == 'early-concat-detr':
        from src.models.detr.earlyConcatDetr import EarlyConcatenationDETR
        model = EarlyConcatenationDETR(args)
    elif args.model == 'early-mul-detr':
        from src.models.detr.earlyMulDetr import EarlyMultiplicationDETR
        model = EarlyMultiplicationDETR(args)
    elif args.model == 'early-affine-detr':
        from src.models.detr.earlyAffineDetr import EarlyAffineDETR
        model = EarlyAffineDETR(args)
    elif args.model == 'early-shift-detr':
        from src.models.detr.earlyShiftDetr import EarlyShiftDETR
        model = EarlyShiftDETR(args)
    elif args.model == 'detr':
        from src.models.detr.detr import DETR
        model = DETR(args)
    elif args.model == 'yolos':
        from src.models.yolos.yolos import Yolos
        model = Yolos(args)
    elif args.model == 'early-concat-yolos':
        from src.models.yolos.earlyConcatYolos import EarlyConcatenationYOLOS
        model = EarlyConcatenationYOLOS(args)
    elif args.model == 'enhanced-yolos':
        from src.models.yolos.enhancedYolos import EnhancedYolos
        model = EnhancedYolos(args)
    elif args.model == 'vsr-yolos':
        from src.models.yolos.vsr_yolos import VSRYolos
        model = VSRYolos(args)
    elif args.model == 'estrnn-yolos':
        from src.models.yolos.estrnn_yolos import ESTRNNYolos
        model = ESTRNNYolos(args)
    else:
        raise ValueError(f'unknown model: {args.model}')

    if args.weight != '':
        device = torch.device(args.device)
        model_path = os.path.join(args.currentDir, args.weight)
        model = load_weights(model, model_path, device)

    # multi-GPU training
    if args.multi:
        model = torch.nn.DataParallel(model)

    return model 


def load_datasets(args):
    if args.model == 'vsr-yolos' or args.model == 'estrnn-yolos':
        from src.datasets.coco_video import VideoCOCODataset
        train_dataset = VideoCOCODataset(args.dataDir, args.trainAnnFile, args.numClass, args.trainVideoFrames, args.numFrames, dummy=args.dummy)
        val_dataset = VideoCOCODataset(args.dataDir, args.valAnnFile, args.numClass, args.valVideoFrames, args.numFrames, dummy=args.dummy)
        test_dataset = VideoCOCODataset(args.dataDir, args.testAnnFile, args.numClass, args.testVideoFrames, args.numFrames, dummy=args.dummy)
        
    else:
        from src.datasets.coco import COCODataset
        train_dataset = COCODataset(args.dataDir, args.trainAnnFile, args.numClass)
        val_dataset = COCODataset(args.dataDir, args.valAnnFile, args.numClass)
        test_dataset = COCODataset(args.dataDir, args.testAnnFile, args.numClass)

    return train_dataset, val_dataset, test_dataset