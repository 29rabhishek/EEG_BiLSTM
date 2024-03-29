from torch import nn
import torch.nn.functional as F


class Feature_Encoding(nn.Module):
    def __init__(
        self,
        lstm_input_size, # depend on regional info extraction module
        fcn_out_features,
        lstm_hidden_size ,
        lstm_num_layer,
        lstm_config,
        num_classes,
        dropout,
    ):
        super().__init__()
        self.BiLstm_layer = nn.LSTM(
            input_size = lstm_input_size,
            hidden_size = lstm_hidden_size,
            num_layers = lstm_num_layer,
            # bias = True,
            batch_first = True,
            dropout= dropout,
            bidirectional = True if lstm_config == "BiLSTM" else False
            )
        self.D = 2 if lstm_config == "BiLSTM" else 1
        self.fcn_layer = nn.Linear(in_features = self.D*lstm_hidden_size, out_features = fcn_out_features, bias=True)
        self.relu = nn.ReLU()
        self.fcn_layer2 = nn.Linear(in_features = fcn_out_features, out_features = num_classes)

    def forward(self, x):
        # self.BiLstm_layer.flatten_parameters()
        x , _ = self.BiLstm_layer(x)
        x = self.fcn_layer(x[:, -1, :])
        x = self.relu(x)
        x = self.fcn_layer2(x)
        return x
    
class Feature_Extraction(nn.Module):
    """
    Raw EEG Data with LSTM based Feature Extraction
    this model taken from 
    EEG Based Feature Extraction for Visual classification for Deep Learning paper
    """
    def __init__(self):
        super().__init__()
        self.bilstm_50_unit = nn.LSTM(
            input_size=128,
            hidden_size=50,
            batch_first=True,
            bidirectional= True
            )
        self.lstm_128_unit = nn.LSTM(
            input_size=100,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            )
        self.lstm_50_unit = nn.LSTM(
            input_size=128,
            hidden_size=50,
            num_layers=1,
            batch_first=True
        )
        self.fcn = nn.Linear(
            in_features=50,
            out_features=128
            )
        self.output_layer = nn.Linear(
            in_features=128,
            out_features=40
        )


    def forward(self, x):
        self.bilstm_50_unit.flatten_parameters()
        self.lstm_128_unit.flatten_parameters()
        self.lstm_50_unit.flatten_parameters()
        x, _ = self.bilstm_50_unit(x)
        x, _ = self.lstm_128_unit(x)
        x, _ = self.lstm_50_unit(x)
        x = self.fcn(x[:, -1, :])
        x = self.output_layer(x)
        return x
    
class Classification_Model_Softmax(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fcn = nn.Linear(in_features = input_size, out_features = output_size)
    def forward(self, x):
        x = self.fcn(x)
        return F.softmax(x, dim = 1)
    
class Multiclass_SVM(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(in_features = input_size, out_features = output_size, bias = True)
    def forward(self, x):
        return self.linear(x)
        


def build_model(model, config, is_xavier_init = False):
    if config is not None:
        model = model(**config)
    else:
        model = model()
    if is_xavier_init:
        for name, param in model.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)

    return model
