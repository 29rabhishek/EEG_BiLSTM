import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

def regional_info_extraction(data, laterization_dict):
    """
    This fucntion extract region level information for more
    info EEG-based image classification via a region-level stacked bi-directional deep learning framework

    Parameter 
    data: tensor, input_data
    left_hms: list, index of left hemisphere EEG channels
    right_hms: list, index of right hemisphere EEG channels
    middle_hms: list, index of middle hemisphere EEG channels

    Return
    X: tensor,
    """
    left_hms = laterization_dict['left_hms']
    right_hms = laterization_dict['right_hms']
    middle_hms = laterization_dict['middle_hms']

    comb_idx = list(zip(left_hms, right_hms))
    hms_diff = []
    data = data.reshape((data.shape[0], data.shape[1], data.shape[2], 1))

    for (i, j) in comb_idx:
        hms_diff.append(data[:, i, :, :] - data[:, j, :, :])

    D = torch.cat(hms_diff, dim = 2)
    S = data[:, middle_hms, : ,:].permute((0,2,1,3)).reshape((data.shape[0], data.shape[2], -1))
    X = torch.cat((D,S), dim = 2)
    
    # print(f"no. of iter: {len(comb_idx)}")
    # print(f"D shape: {D.shape} & D dimension {D.ndim}")
    # print(f"S shape: {S.shape} & S dimension {S.ndim}")
    # print(f"X shape: {X.shape} & X dimension {X.ndim}")
    return X


def multi_class_svm_loss(scores, labels, margin=1.0):
    """
    Multi-class SVM loss function.

    Parameters:
    - scores: Raw scores from the model (before softmax)
    - labels: True labels
    - margin: Margin for the SVM loss

    Returns:
    - loss: Scalar value of the loss
    """
    # Get the number of classes
    num_classes = scores.size(1)

    # Compute the scores for the correct classes
    correct_scores = scores[torch.arange(len(scores)), labels]

    # Compute the margins
    margins = scores - correct_scores.view(-1, 1) + margin

    # Zero out the margins for the correct classes
    margins[torch.arange(len(margins)), labels] = 0

    # Compute the loss and take the mean
    loss = torch.sum(torch.relu(margins)) / len(scores)

    return loss

def checkpoint(
    model_dict : dict,
    optimizer:torch.optim.Optimizer,
    epoch: int,
    loss:float,
    path
    ):
    '''
    Save the states model, optimizer, loss, epoch
    '''
    model_checkpoint  = {
        'encoding_model': model_dict['encoding_model'].state_dict(),
        'cls_model': model_dict['cls_model'].state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch+1,
        'loss': loss
    }
    new_path = f'model_checkpoint_{epoch+1}.pth'
    path = path/new_path
    torch.save(model_checkpoint, path)
    print(f'Model Checkpoint {epoch+1}')

def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().detach().cpu().numpy().mean())
            max_grads.append(p.grad.abs().detach().cpu().numpy().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

def grad_flow(named_parameters):
    layers = []
    avg_grads = []
    max_grads= []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            avg_grads.append(p.grad.abs().detach().cpu().numpy().mean())
            max_grads.append(p.grad.abs().detach().cpu().numpy().max())
    return (layers, avg_grads, max_grads)
