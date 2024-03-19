import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from pathlib import Path
from datetime import date

class regional_info_extraction:
    def __init__(
            self,
            left_hms ,
            right_hms ,
            middle_hms,
            ):
        self.left_hms = left_hms
        self.right_hms = right_hms
        self.middle_hms = middle_hms
        

    def fit_transform(self, data):
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
        comb_idx = list(zip(self.left_hms, self.right_hms))
        hms_diff = []
        data = data.reshape((data.shape[0], data.shape[1], data.shape[2], 1))

        for (i, j) in comb_idx:
            hms_diff.append(data[:, i, :, :] - data[:, j, :, :])

        D = np.concatenate(hms_diff, axis = 2)
        S = data[:, self.middle_hms, : ,:].transpose((0,2,1,3)).reshape((data.shape[0], data.shape[2], -1))
        X = np.concatenate((D,S), axis = 2)
        
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

def model_checkpoint(
    model_type: str,
    model : torch.nn.Module,
    optimizer:torch.optim.Optimizer,
    scheduler : torch.optim.lr_scheduler,
    hist_dict: dict,
    path : str
    ):
    '''
    Save the states model, optimizer, loss, epoch
    '''
    checkpoint_dict  = {
        model_type: model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'hist_dict': hist_dict
    }
    new_path = f"{model_type}_checkpoint_{date.today().strftime('%d_%m')}_{hist_dict['epochs'][-1]}.pth"
    path = Path(path)/new_path
    torch.save(checkpoint_dict, path)
    print("")
    print(f"{model_type} Model Checkpoint {hist_dict['epochs'][-1]}")
    print("")


def latest_weight_file_path(checkpoint_folder_path, model_type):
    model_filename = f"{model_type}_checkpoint_*"
    weight_files = list(Path(checkpoint_folder_path).glob(model_filename))
    if len(weight_files) == 0:
        return None
    weight_files.sort()
    return str(weight_files[0])

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
