import os
import random

random.seed(5)

with open('.env') as f:
    _data_pth_ = f.readline().strip()
    _data_pth_ = os.path.expanduser(_data_pth_)
    _model_pth_ = f'{_data_pth_}/models'
    if not os.path.exists(_model_pth_):
        os.mkdir(_model_pth_)