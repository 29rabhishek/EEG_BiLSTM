from sample_data_gen import create_test_data
from data_setup import train_test_dataloader
from train import train_engin
sample_x, sample_y = create_test_data()
train_dataloader, test_dataloader = train_test_dataloader(sample_x, sample_y, 0.2, 32)
EPOCHS = 2

hist_dict = train_engin(
    epochs=EPOCHS,
    
)


