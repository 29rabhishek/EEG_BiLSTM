import torch
from utils import regional_info_extraction
import torch.utils.functional as F

def train_step(
    train_dataloader,
    feature_encoding_model,
    cls_model,
    optimizer,
    lr_scheduler,
    loss_fn,
    device,
    right_hms
    left_hms
    middle_hms
    ):
    """
    This function train the model and return training loss

    Parameter
    train_dataloader: type DataLoader, DataLoader of train dataset
    feature_encoding_model: torch.nn.Module
    cls_model: torch.nn.Module
    optimizer: torch.optim
    lr_scheduler: torch.optim.lr_scheduler, learning rate scheduler
    loss_fn: torch.nn
    device:
    left_hms: list
    right_hms: list
    middle_hms: list
    
    Return 
    model_loss: list[tensor]
    model_acc: list[tensor]
    """
    feature_encoding_model.train()
    cls_model.train()
    model_loss = []
    model_acc = []
    for X, y in train_dataloader:
        X = regional_info_extraction(X, left_hms, right_hms, middle_hms)
        X = feature_encoding_model(X)
        yhat = cls_model(X)
        loss = loss_fn(yhat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model_loss += loss
        acc = (F.softmax(yhat, dim = 1).argmax(dim = 1) == y).sum().item()/len(y)
        mode_acc += acc
    model_loss /= len(train_dataloader)
    model_acc /= len(train_dataloader)
    return model_loss, mode_acc

