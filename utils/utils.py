import torch 

def load_weights(model, weight_path, device):

    if weight_path:
        print(f'loading pre-trained weights from {weight_path}')
        model.load_state_dict(torch.load(weight_path, map_location=device))
    else:
        print('no pre-trained weights found, training from scratch...')
    return model
        