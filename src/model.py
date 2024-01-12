from torch import nn
import torch.nn.functional as F


class Feature_Encoding(nn.Module):
    def __init__(
        self,
        input_size, # depend on regional info extraction module
        out_features = 5,
        num_layer = 2,
        hidden_size = 5,
        lstm_config = "BiLSTM"
    ):
        super().__init__()
        self.Stack_BiLstm_layer = nn.LSTM(input_size = input_size,
                                          hidden_size = hidden_size,
                                          num_layers = num_layer,
                                          bias = True,
                                          batch_first = True,
                                          dropout=0.0,
                                          bidirectional = True if lstm_config == "BiLSTM" else False)
        self.fcn_layer = nn.Linear(in_features = 2*hidden_size, out_features = out_features, bias=True)
        self.activation_layer = nn.ReLU()

    def forward(self, x):
        x, _ = self.stack_BiLSTM_layer(x)
        x = self.fcn_layer(x[:, -1, :])
        x = self.activation_layer(x)
        return x
    

class Classification_Model_Softmax(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fcn = nn.Linear(in_features = input_size, out_features = 40)
    def forward(self, x):
        return F.softmax(x, dim = 1)
    

class multiclass_svm(nn.module):
    def __inti__(self, input_size, num_classes):
        super().__init__(self)
        self.linear = nn.Linear(in_features = input_size, out_features = num_classes, bias = True)
    def foreard(self, X):
        return self.linear(X)
        