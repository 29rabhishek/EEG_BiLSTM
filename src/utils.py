import torch
def regional_info_extraction(data, left_hms, right_hms, middle_hms):
    """
    This fucntion extract region level information for more
    info EEG-based image classification via a region-level stacked bi-directional deep learning framework

    Parameter 
    data: tensor, input_data
    left_hms: list, index of left hemisphere EEG channels
    right_hms: list, index of right hemisphere EEG channels
    middle_hms: list, index of middle hemisphere EEG channels

    Return
    X: tensor,
    """
    comb_idx = list(zip(left_hms, right_hms))
    hms_diff = []
    print(data.shape)
    data = data.reshape((data.shape[0], data.shape[1], data.shape[2], 1))

    for (i, j) in comb_idx:
        hms_diff.append(data[:, i, :, :] - data[:, j, :, :])

    D = torch.cat(hms_diff, dim = 2)
    S = data[:, middle_hms, : ,:].permute((0,2,1,3)).reshape((data.shape[0], data.shape[2], -1))
    X = torch.cat((D,S), dim = 2)
    
    # print(f"no. of iter: {len(comb_idx)}")
    # print(f"D shape: {D.shape} & D dimension {D.ndim}")
    # print(f"S shape: {S.shape} & S dimension {S.ndim}")
    # print(f"X shape: {X.shape} & X dimension {X.ndim}")
    return X