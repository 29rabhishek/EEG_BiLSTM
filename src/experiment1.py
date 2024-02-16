import torch
import time
import pickle
import os
from pathlib import Path
from torch.optim import Adam, lr_scheduler
from torch.nn import CrossEntropyLoss
from model import build_model, Feature_Encoding
from trainer import Trainer
from data_setup import train_test_dataloader
from config import get_config
import logging

logging.basicConfig(filename='training.log', level=logging.INFO, format="%(asctime)s :: %(levelname)s :: %(message)s")

start_time = time.time()
logging.info("Experiment has Started !!")
train_test_path = Path(os.getcwd()).parents[0]/"data/train_test_data.pkl"
with open(train_test_path, "rb") as f:
    train_test_data = pickle.load(f)


# data_path = Path(os.getcwd()).parents[0]/"EEG_BiLSTM/data/data(5-95).npy"
# label_path = Path(os.getcwd()).parents[0]/"EEG_BiLSTM/data/label.npy"
# lat_dict_path = Path(os.getcwd()).parents[0]/"EEG_BiLSTM/data/lat_dict.pkl"


## Parameters of Training
device = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 2500
BATCH_SIZE = 440
LEARNING_RATE = 1e-3
CHECKPOINT_FREQ = 50
MODEL_TYPE = "encoding"
model_folder_path = f"{Path(os.getcwd()).parents[0]}/data/unstacked"


# checkpoint_to_load_path = f"{Path(os.getcwd()).parents[0]}/EEG_BiLSTM/data/unstacked/encoding_checkpoint_30.pth"
model_config = get_config()["Encoding_parameters"]
model = build_model(Feature_Encoding, model_config)
model.to(device)

optimizer = Adam(model.parameters(), lr = LEARNING_RATE)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
    factor=0.1, patience=10, threshold=0.0001, threshold_mode='abs')
loss = CrossEntropyLoss()

trainer_kwarg = {
    "model": model,
    "optimizer":  optimizer,
    "scheduler": scheduler,
    "loss_fn": CrossEntropyLoss(),
    "device": device,
    "model_type": MODEL_TYPE,
}

model_trainer = Trainer(**trainer_kwarg)

# train, test dataloader
train_dataloader, valid_dataloader = train_test_dataloader(
    train = train_test_data["train"],
    test = train_test_data["test"],
    batch_size = BATCH_SIZE
    )

hist_dict = model_trainer.train_engin(
        epochs  = EPOCHS,
        train_dataloader = train_dataloader,
        test_dataloader  = valid_dataloader,
        checkpoint_freq = CHECKPOINT_FREQ,
        checkpoint_to_save_path = model_folder_path, 
        )
end_time = time.time()

execution_time = end_time - start_time

logging.info(f"Experiment has Completed execution time: {execution_time}")
print(f"time to run code in sec: {execution_time}")
