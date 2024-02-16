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
        from src.models.earlySumDetr import EarlySummationDETR
        model = EarlySummationDETR(args)
    elif args.model == 'early-concat-detr':
        from src.models.earlyConcatDetr import EarlyConcatenationDETR
        model = EarlyConcatenationDETR(args)
    elif args.model == 'detr':
        from src.models.detr import DETR
        model = DETR(args)
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