import json
import numpy as np
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import pickle as pkl


def read_mv_mnist(data_dir):

    train_data_dir = os.path.join(data_dir, 'client_4_train.pkl')
    with open(train_data_dir, 'rb') as inf:
        train_data = pkl.load(inf)
    return train_data

def read_user_mv_mnist(index, data):
    train_data = data[index]
    x_train, y_train = train_data[0], train_data[1]
    x_train = torch.Tensor(x_train).type(torch.float32)
    y_train = torch.Tensor(y_train).type(torch.int64)

    train_data = [(x, y) for x, y in zip(x_train, y_train)]

    return index, train_data








