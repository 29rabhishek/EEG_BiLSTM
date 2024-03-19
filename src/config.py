from pathlib import Path

def get_config(use_regional_info = True):
    config_dict1 = {
        "lstm_input_size": 85,
        "lstm_hidden_size": 68,
        "lstm_num_layer": 1,
        "lstm_config" : "BiLSTM",
        "fcn_out_features": 60,
        "num_classes": 40,
        "dropout": 0
        }
 
    config_dict2 = {
        "lstm_input_size": 128,
        "lstm_hidden_size": 68,
        "lstm_num_layer": 2,
        "lstm_config" : "BiLSTM",
        "fcn_out_features":60,
        "num_classes": 39,
        "dropout": 0
        }
    if use_regional_info:
        return config_dict1
    else:
        return config_dict2

