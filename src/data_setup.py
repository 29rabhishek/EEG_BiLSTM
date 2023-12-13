from torch.utils.data import Dataset, DataLoader

def train_test_dataloader(X, y, test_size):
    train_idx = int(X.shape[0]*(1-test_size))
    test_idx = X.shape[0] - train_idx
    return 
    

class eeg_dataset(Dataset):
    def __init__(self, X, y):
        self.data = X
        self.class_id = y
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.class_id[idx]