# NxCxt
import torch
def create_test_data(n_samples = 1, n_channels = 12, n_timeStep = 24, n_classes = 4):
        
        return torch.randn((n_samples, n_channels, n_timeStep)), torch.randn(n_samples, n_classes).argmax(axis = 1)
