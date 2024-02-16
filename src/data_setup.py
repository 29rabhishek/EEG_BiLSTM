from torch.utils.data import Dataset, DataLoader
import torch

def train_test_dataloader(train : dict, test : dict,  batch_size : int):
    """
    this function return train_dataloader and test_dataloader type(DataLoader)

    Parameter
    train: dictionary keys X, y
    test: dictionary keys X, y
    batch_size: int 

    Return
    train_dataloader: torch.utils.data.DataLoader
    test_dataloader: torch.utils.data.DataLoader
    """
    train_dataset = eeg_dataset(torch.tensor(train["X"]), torch.tensor(train["y"]))
    test_dataset = eeg_dataset(torch.tensor(test["X"]), torch.tensor(test["y"]))

    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle= True)
    print(f"Train Size: {train['X'].shape[0]} || Test Size: {test['X'].shape[0]} ")
    return train_dataloader, test_dataloader
    

class eeg_dataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.class_id = label
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):

        return self.data[idx], self.class_id[idx]