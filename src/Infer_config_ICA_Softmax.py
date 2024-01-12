import torch
from torch.optim import Adam, lr_scheduler
from torch.nn import BCELoss
from sample_data_gen import create_test_data
from data_setup import train_test_dataloader
from model import Feature_Encoding, Classification_Model_Softmax
from train import train_engin

device = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_EPOCHS = 1
BATCH_SIZE = 32
LEARNING_RATE = 0.001
left_hms = [1,3,6] #left_hemisphere
right_hms = [0,2,5] #right_hemisphere
middle_hms = [8, 9] #middle_hemisphere



if len(left_hms) != len(right_hms):
    raise Exception(f"left_hms len: {len(left_hms)} != right_hms len: {len(right_hms)}")

# Model parameter of Feature_Encoding Module

# input size for BiLSTM/LSTM in Feature Encoding Module
lstm_input_size =  len(left_hms)+len(right_hms)
fcn_out_features = 5,
lstm_hidden_size = 5 ,
lstm_num_layer = 2,
lstm_config = "BiLSTM"




sample_x, sample_y = create_test_data()
train_dataloader, test_dataloader = train_test_dataloader(sample_x, sample_y, 0.2, BATCH_SIZE)

feature_encoding_model = Feature_Encoding(
    lstm_input_size = lstm_input_size,
    fcn_out_features = fcn_out_features,
    lstm_hidden_size = lstm_hidden_size ,
    lstm_num_layer = lstm_num_layer,
    lstm_config = lstm_config
    ).to(device)

cls_model = Classification_Model_Softmax(input_size = fcn_out_features).to(device)


loss_fn = BCELoss()
optimizer = Adam(params=[Feature_Encoding.parameters(), Classification_Model_Softmax.parameters()], lr = LEARNING_RATE)
scheduler = lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)
hist_dict = train_engin(
    epochs=NUM_EPOCHS,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    feature_encoding_model=Feature_Encoding,
    cls_model=Classification_Model_Softmax,
    optimizer=optimizer,
    scheduler = scheduler,
    loss_fn=loss_fn,
    device=device,
    left_hms=left_hms,
    right_hms=right_hms,
    middle_hms=middle_hms,
    ica_model = None
)





