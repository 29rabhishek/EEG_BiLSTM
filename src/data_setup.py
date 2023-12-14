from torch.utils.data import Dataset, DataLoader
import numpy as np
import random 

def train_test_dataloader(X, y, test_frac, batch_size):
    """
    this function split X into train and test data using test_frac for test size
    then return train_dataloader and test_dataloader type(DataLoader)

    Parameter
    X: tensor
    y: tensor
    test_frac:  float in range (0,1)
    batch_size: int 

    Return
    train_dataloader: torch.utils.data.DataLoader
    test_dataloader: torch.utils.data.DataLoader
    """
    test_size = int(X.shape[0]*(test_frac))
    train_size = X.shape[0] - test_size

    idx = range(0, X.shape[0])
    train_idx = random.sample(idx, train_size)
    test_idx = list(set(idx) - set(train_idx))

    train_dataset = eeg_dataset(X[train_idx], y[train_idx])
    test_dataset = eeg_dataset(X[test_idx], y[test_idx])

    train_dataloader = DataLoader(train_dataset, batch_size = batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size)
    print(f"Train Size: {train_size} || Test Size: {test_size} ")
    return train_dataloader, test_dataloader
    

class eeg_dataset(Dataset):
    def __init__(self, X, y):
        self.data = X
        self.class_id = y
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.class_id[idx]