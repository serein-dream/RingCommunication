# some tools for wide deep model
# author: WenYi
# create: 2019-09-23
import torch.nn as nn
import torch
import torch.nn.functional as F


def linear(inp, out, dropout):
    """
    linear model module by nn.sequential
    :param inp: int, linear model input dimensio
    :param out: int, linear model output dimension
    :param dropout: float dropout probability for linear layer
    :return: tensor
    """
    return nn.Sequential(
        nn.Linear(inp, out),
        nn.LeakyReLU(),
        nn.Dropout(dropout)
    )


def set_method(method):
    if method == 'regression':
        return None, F.mse_loss
    if method == 'binary':
        return torch.sigmoid, F.binary_cross_entropy
    if method == 'multiclass':
        return F.softmax, F.cross_entropy

