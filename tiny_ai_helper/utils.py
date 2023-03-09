# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT
##


def get_default_device():
    """
    Returns default device
    """
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    return device


def get_acc_class(batch_predict, batch_y):
    """
    Returns class accuracy
    """
    
    batch_y = torch.argmax(batch_y, dim=1)
    batch_predict = torch.argmax(batch_predict, dim=1)
    acc = torch.sum( torch.eq(batch_y, batch_predict) ).item()
    
    return acc


def get_acc_binary(batch_predict, batch_y):
    """
    Returns binary accuracy
    """
    
    from torcheval.metrics import BinaryAccuracy
    
    batch_predict = batch_predict.reshape(batch_predict.shape[0])
    batch_y = batch_y.reshape(batch_y.shape[0])
    
    acc = BinaryAccuracy() \
        .to(batch_predict.device) \
        .update(batch_predict, batch_y) \
        .compute().item()
    
    return round(acc * len(batch_y))