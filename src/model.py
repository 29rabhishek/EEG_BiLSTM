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
        self.Stack_BiLstm_layer = nn.LSTM(input_size = lstm_input_size,
                                          hidden_size = lstm_hidden_size,
                                          num_layers = lstm_num_layer,
                                          bias = True,
                                          batch_first = True,
                                          dropout= dropout,
                                          bidirectional = True if lstm_config == "BiLSTM" else False)
        self.D = 2 if lstm_config == "BiLSTM" else 1
        self.fcn_layer = nn.Linear(in_features = self.D*lstm_hidden_size, out_features = fcn_out_features, bias=True)
        self.activation_layer = nn.ReLU()
        self.fcn_layer2 = nn.Linear(in_features = fcn_out_features, out_features = num_classes)

    def forward(self, x):
        self.Stack_BiLstm_layer.flatten_parameters()
        x , _ = self.Stack_BiLstm_layer(x)
        x = self.fcn_layer(x[:, -1, :])
        x = self.activation_layer(x)
        x = self.fcn_layer2(x)
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
        


def build_model(model, config):
    return model(**config)
