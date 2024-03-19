# Author: Abhishek Rathore
# Date Created: 4th March 2024
"""
This script exclude regional info calculation, directly run on data(5-95).npy and label data
"""

import torch
import time
from datetime import date
import pickle
import os
from pathlib import Path
from torch.optim import Adam, lr_scheduler
from torch.nn import CrossEntropyLoss
from model import build_model, Feature_Extraction
from trainer import Trainer
from data_setup import train_test_dataloader
from config import get_config
import logging
import numpy as np
import wandb

start_time = time.time()
wandb.login()



logging.basicConfig(
    filename=f"training logs/exp4_train_{date.today().strftime('%d_%m')}.log",
    level=logging.INFO,
    format="%(asctime)s :: %(levelname)s :: %(message)s"
    )
logging.info("Experiment has Started !!")

# loading data
# data: (n_samples, 128, 440) --> (n_samples, 440, 128)
data = np.load(Path(os.getcwd()).parents[0]/"data/data(5-95).npy").transpose(0,2,1)
label = np.load(Path(os.getcwd()).parents[0]/"data/label.npy")

"""
split used in this Experiment _4
train\test split 80\20 %
    Train split : 72%
    Valid split : 8%
    Test split  : 20%
"""
# splitting data in train, valid, test split

train_valid_size  =  round((data.shape[0])*.8)
train_size = round(train_valid_size*.85)

val_size = train_valid_size - train_size
test_size = data.shape[0] - train_valid_size



X_train, y_train = data[: train_size], label[: train_size]
X_valid, y_valid = data[train_size : train_valid_size], label[train_size : train_valid_size]
X_test, y_test = data[train_valid_size : ], label[train_valid_size : ]

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
Experiment = "experiment_4"
model_folder_path = f"{Path(os.getcwd()).parents[0]}/data/{Experiment}"



# loading model config
model_config = None
model = build_model(Feature_Extraction, config = model_config)
model.to(device)

model_hyperparameter = {
    "BiLSTM_50_unit": 50,
    "LSTM_128_unit":128,
    "LSTM_50_unit":50,
    "fcn": 128,
    "output": 40
}

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

# initalizing wandb run
run = wandb.init(
    project="EEG_BiLSTM",
    notes = "experiment4",
    config={
        "learning_rate": LEARNING_RATE,
        "batch_size":BATCH_SIZE,
        "epochs": EPOCHS,
        "model_parameter": model_hyperparameter,
        "optimizer":"Adam",
        "Loss Function": "CrossEntropyLoss",
        "Scheduler": "ExponentialLR"
    }
)

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
run.log({"test_acc": test_acc})
logging.info(f"Test Accuracy: {test_acc::4f}")

#dumping hist_dict
path_to_hist_dict = f"{Path(os.getcwd()).parents[0]}/data/exp_4_hist_dict_{date.today().strftime('%d_%m')}.pkl"
with open( path_to_hist_dict, "wb") as F:
    pickle.dump(hist_dict, F)

end_time = time.time()
execution_time = end_time - start_time
logging.info(f"Experiment has Completed execution time: {execution_time}")
print(f"time to run code in sec: {execution_time}")
run.finish()