import torch
from utils import regional_info_extraction
import torch.utils.functional as F
# from typing import Dict, List, Tuple

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
    This function train the model and return training loss and validation accuracy

    Parameter
    train_dataloader: type DataLoader, DataLoader of train dataset
    feature_encoding_model: torch.nn.Module
    cls_model: torch.nn.Module
    optimizer: torch.optim
    scheduler: torch.optim.lr_scheduler, learning rate scheduler
    loss_fn: torch.nn.Module
    device: torch.device
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
        X.to(device)
        y.to(device)
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


def test_step(
    test_dataloader,
    feature_encoding_model,
    cls_model,
    loss_fn,
    device,
    right_hms
    left_hms
    middle_hms
    ):
    """
    This function test the model and return validation loss and validation accuarcy

    Parameter
    test_dataloader: type DataLoader, DataLoader of test dataset
    feature_encoding_model: torch.nn.Module
    cls_model: torch.nn.Module
    loss_fn: torch.nn
    device: torch.device
    left_hms: list
    right_hms: list
    middle_hms: list

    Return 
    test_loss: list[tensor]
    test_acc: list[tensor]
    """
    feature_encoding_model.eval()
    cls_model.eval()
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for batch, (X,y) in enumerate(test_dataloader):
            X = regional_info_extraction(X, left_hms, right_hms, middle_hms)
            X.to(device)
            y.to(device)
            X = feature_encoding_model(X)
            yhat = cls_model(X)
            loss = loss_fn(yhat, y)

            test_loss += loss
            yhat_class = torch.argmax(torch.softmax(yhat, dim =1), dim =1)
            test_acc += (yhat_class == y).sum().item()/len(yhat)

    test_loss = test_loss/len(test_dataloader)
    test_acc = test_acc/len(test_dataloader)
    return test_loss, test_acc


def train_engin(
    epochs : int,
    train_dataloader : torch.utils.data.DataLoader,
    test_dataloader : torch.utils.data.DataLoader,
    feature_encoding_model : torch.nn.Module,
    cls_model: torch.nn.Module,
    optimizer : torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    loss_fn : torch.nn.Module,
    device : torch.device,
    left_hms: list,
    right_hms: list,
    middle_hms: list
    ):
    """
    This Function Train and do validation testing, return result dictionary
    Parameter:
        epochs: type int, number of iteration
    return:
        res_dict: type Dict, keys train_loss, train_acc, test_loss, test_acc
    """
    res_dict = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
        }
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(
            train_dataloader = train_dataloader,
            feature_encoding_model=feature_encoding_model,
            cls_model=cls_model,
            optimizer = optimizer,
            scheduler = scheduler,
            loss_fn=loss_fn,
            device=device,
            left_hms = left_hms,
            right_hms = right_hms,
            middle_hms = middle_hms
            )
        
        test_loss, test_acc = test_step(
            test_dataloader = test_dataloader,
            feature_encoding_model=feature_encoding_model,
            cls_model=cls_model,
            loss_fn=loss_fn,
            device=device,
            left_hms = left_hms,
            right_hms = right_hms,
            middle_hms = middle_hms
            )
        print(
        f"Epoch: {epoch+1} | "
        f"train_loss: {train_loss:.4f} | "
        f"train_acc: {train_acc:.4f} | "
        f"test_loss: {test_loss:.4f} | "
        f"test_acc: {test_acc:.4f}"
        )
        res_dict['train_loss'].append(train_loss)
        res_dict['train_acc'].append(train_acc)
        res_dict['test_loss'].append(test_loss)
        res_dict['test_acc'].append(test_acc)

    return res_dict
