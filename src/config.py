from pathlib import Path

def get_config():
    config_dict = {
        "Encoding_parameters" : {
        "lstm_input_size": 85,
        "fcn_out_features": 60,
        "lstm_hidden_size": 68,
        "lstm_num_layer": 1,
        "lstm_config" : "BiLSTM",
        "num_classes": 40,
        "dropout": 0
        }
    }
    return config_dict

