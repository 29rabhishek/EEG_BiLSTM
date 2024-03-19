# Author: Abhishek Rathore
# Date Created: 18.03.2024
"""
This script exclude regional info calculation, directly run on data.npy and label data
"""

import torch
import time
from datetime import date
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
import numpy as np
import wandb

start_time = time.time()
wandb.login()



logging.basicConfig(
    filename=f"training logs/exp5_train_{date.today().strftime('%d_%m')}.log",
    level=logging.INFO,
    format="%(asctime)s :: %(levelname)s :: %(message)s"
    )
logging.info("Experiment has Started !!")

# loading data
# data: (n_samples, 128, 440) --> (n_samples, 440, 128)
train_data_path = Path(os.getcwd()).parents[0]/"data/Data/Train/"
val_data_path = Path(os.getcwd()).parents[0]/"data/Data/Val/"
test_data_path = Path(os.getcwd()).parents[0]/"data/Data/Test/"

data = np.load(Path(os.getcwd()).parents[0]/"data/data(5-95).npy").transpose(0,2,1)
label = np.load(Path(os.getcwd()).parents[0]/"data/label.npy")

"""
split used in this Experiment_5
train/val/test split 80/10/10 %
"""



X_train, y_train = np.load(train_data_path/"eeg.npy"), np.load(train_data_path/"label.npy")
X_valid, y_valid = np.load(val_data_path/"eeg.npy"), np.load(val_data_path/"label.npy")
X_test, y_test = np.load(test_data_path/"eeg.npy"), np.load(test_data_path/"label.npy")

logging.info(f"Train Size X: {X_train.shape[0]}, y: {y_train.shape[0]}")
logging.info(f"Valid Size X: {X_valid.shape[0]}, y: {y_valid.shape[0]}")
logging.info(f"Test Size X: {X_test.shape[0]}, y: {y_test.shape[0]}")

train_test_data = {
    "train": {
        "X": X_train,
        "y": y_train
    },
    "test" : {
        "X" : X_valid,
        "y" : y_valid

    }
}


## Parameters of Training
device = 'cuda' if torch.cuda.is_available() else 'cpu'

EPOCHS = 2500
BATCH_SIZE = 440
LEARNING_RATE = 1e-3
CHECKPOINT_FREQ = 50
MODEL_TYPE = "encoding"
Experiment = "experiment_5"
model_folder_path = f"{Path(os.getcwd()).parents[0]}/data/{Experiment}"



# loading model config
model_config = get_config(use_regional_info=False)
model = build_model(Feature_Encoding, model_config)
model.to(device)

# initalizing wandb run
run = wandb.init(
    project="EEG_BiLSTM",
    notes = "experiment5 with new downloaded data",
    config={
        "learning_rate": LEARNING_RATE,
        "batch_size":"BATCH_SIZE",
        "epochs": "EPOCHS",
        "model_parameter": model_config
    }
)

optimizer = Adam(model.parameters(), lr = LEARNING_RATE)
# scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
#     factor=0.1, patience=10, threshold=0.0001, threshold_mode='abs')
scheduler = lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)
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

hist_dict, model = model_trainer.train_engin(
        epochs  = EPOCHS,
        train_dataloader = train_dataloader,
        test_dataloader  = valid_dataloader,
        checkpoint_freq = CHECKPOINT_FREQ,
        checkpoint_to_save_path = model_folder_path,
        wandb_run = run
        )


# evaluating the model on test data
model.eval()
yhat_test = model(torch.tensor(X_test).to(device))
test_acc = (torch.nn.softmax(yhat_test, dim = 1).argmax(dim = 1) == torch.tensor(y_test).to(device)).sum().item() / yhat_test.shape[0]

logging.info(f"Test Accuracy: {test_acc::4f}")

#dumping hist_dict
path_to_hist_dict = f"{Path(os.getcwd()).parents[0]}/data/exp_5_hist_dict_{date.today().strftime('%d_%m')}.pkl"
with open( path_to_hist_dict, "wb") as F:
    pickle.dump(hist_dict, F)

end_time = time.time()
execution_time = end_time - start_time
logging.info(f"Experiment has Completed execution time: {execution_time}")
print(f"time to run code in sec: {execution_time}")
