from torch import nn
import torch.nn.functional as F


class Feature_Encoding(nn.Module):
    def __init__(
        self,
        input_size = 25,
        out_features = 5,
        num_layer = 2,
        hidden_size = 5,
        lstm_config = "BiLSTM"
    ):
        super().__init__(self)
        self.Stack_BiLstm_layer = nn.LSTM(input_size = input_size,
                                          hidden_size = hidden_size,
                                          num_layers = num_layer,
                                          bias = True,
                                          batch_first = True,
                                          dropout=0.0,
                                          bidirectional = True if lstm_config == "BiLSTM" else False)
        self.fcn_layer = nn.Linear(in_features = hidden_size, out_features = out_features, bias=True)
        self.activation_layer = nn.ReLU()

    def forward(self, x):
        x = self.stack_BiLSTM_layer(x)
        x = self.fcn_layer(x[:, -1, :])
        x = self.activation_layer(x)
        return x
    

class Classification_model(nn.Module):
    def __init__(self, input_size):
        super().__init__(self)
        self.fcn = nn.Linear(in_features = input_size, out_features = 40)
    def forward(self, x):
        return F.softmax(x, dim = 1)
        