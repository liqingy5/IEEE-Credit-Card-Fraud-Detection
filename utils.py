import os
import random

_random_seed_ = 5
random.seed(_random_seed_)

with open('.env') as f:
    _data_pth_ = f.readline().strip()
    _data_pth_ = os.path.expanduser(_data_pth_)
    _model_pth_ = f'{_data_pth_}/models'
    if not os.path.exists(_model_pth_):
        os.mkdir(_model_pth_)
        
'''
data_type: standard, missing, redundant, undersample
'''
def load_standard_data(data_type="standard"):
    from pandas import read_csv
    from sklearn.model_selection import train_test_split
    if data_type == "standard":
        type_pth = "joined"
    elif data_type == "missing":
        type_pth = "joined_missing_value"
    elif data_type == "redundant":
        type_pth = "joined_redundant_value"
    elif data_type == "undersampled":
        type_pth = "joined_undersampled_value"
        
    pth = f"{_data_pth_}/processed/train_{type_pth}.csv"
   

    data = read_csv(pth, index_col=0)
    y, X = data['isFraud'], data.drop(columns=['isFraud'])
    return train_test_split(X, y, test_size=0.33, random_state=_random_seed_)