# -*- coding: utf-8 -*-

##
# Tiny ai helper
# Copyright (с) Ildar Bikmamatov 2022 - 2023 <support@bayrell.org>
# License: MIT
##

import torch
import numpy as np
from typing import overload
from PIL import Image, ImageDraw
from .utils import resize_image


class InsertFirstAxis(torch.nn.Module):
    
    """
    Insert first Axis for convolution layer
    """
    
    def __call__(self, t):
        t = t[:,None,:]
        return t


class MoveRGBToEnd(torch.nn.Module):
        
    def __call__(self, t):
        l = len(t.shape)
        t = torch.moveaxis(t, l-3, l-1)
        return t


class MoveRGBToBegin(torch.nn.Module):
        
    def __call__(self, t):
        l = len(t.shape)
        t = torch.moveaxis(t, l-1, l-3)
        return t


class ToIntImage(torch.nn.Module):
    
    def __call__(self, t):
        
        t = t * 255
        t = t.to(torch.uint8)
        
        return t


class ToFloatImage(torch.nn.Module):
    
    def __call__(self, t):
        
        t = t.to(torch.float)
        t = t / 255.0
        
        return t


class ReadImage(torch.nn.Module):
    
    def __init__(self, mode=None):
        
        self.mode=mode
    
    def __call__(self, t):
        
        t = Image.open(t)
        
        if self.mode is not None and self.mode != t.mode:
            t = t.convert(self.mode)
        
        t = torch.from_numpy( np.array(t) )
        
        return t


class ResizeImage(torch.nn.Module):
    
    def __init__(self, size, contain=True, color=None):
        
        torch.nn.Module.__init__(self)
        
        self.size = size
        self.contain = contain
        self.color = color
    
    def __call__(self, t):
        
        t = resize_image(t, self.size, contain=self.contain, color=self.color)
        
        return t
    
    def extra_repr(self) -> str:
        return 'size={}, contain={}, color={}'.format(
            self.size, self.contain, self.color
        )


class NormalizeImage(torch.nn.Module):
    
    def __init__(self, mean, std, inplace=False):
        
        import torchvision
        
        torch.nn.Module.__init__(self)
        
        self.mean = mean
        self.std = std
        self.inplace = inplace
        self.normalize = torchvision.transforms.Normalize(mean=mean, std=std, inplace=inplace)
    
    def __call__(self, t):
        
        t = self.normalize(t)
        
        return t
    
    def extra_repr(self) -> str:
        return 'mean={}, std={}, inplace={}'.format(
            self.mean, self.std, self.inplace
        )


class PreparedModule(torch.nn.Module):
    
    def __init__(self, module, weight_path, *args, **kwargs):
        
        torch.nn.Module.__init__(self)
        
        self.module = module
        self.weight_path = weight_path
        
        for param in self.module.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        x = self.module(x)
        return x
    
    def load(self, *args, **kwargs):
        
        """
        Load model from file
        """
        
        state_dict = torch.load( self.weight_path )
        self.module.load_state_dict( state_dict )
    
    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        pass
    
    def load_state_dict(self, state_dict, strict = True):
        pass


class TensorList:
    
    def __init__(self, *tensors):
        self.tensors = tensors
    
    def to(self, device):
        for m in self.modules:
            self.tensors[m] = self.tensors[m].to(device)
    
    def __repr__(self):
        return "TensorList(" + str(self.tensors) + ")"
    

class Stacking(torch.nn.Module):
    
    def __init__(self, *args):
        torch.nn.Module.__init__(self)
        for i, module in enumerate(args):
            self.add_module(str(i), module)
    
    def forward(self, tensor_list):
        
        device = tensor_list[0].device
        res = torch.tensor([]).to(device)
        
        keys = list(self._modules.keys())
        for index, m in enumerate(keys):
            if self._modules[m] is not None:
                module = self._modules[m]
                x = module(tensor_list[index])
            else:
                x = tensor_list[index]
            
            res = torch.cat( (res, x), dim = 1 )
            
        return res
    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        keys = self._modules.keys()
        for m in keys:
            module = self._modules[m]
            if module is not None:
                module.state_dict(
                    destination=destination,
                    prefix=prefix + m + '.',
                    keep_vars=keep_vars
                )


class Pipe():
    def __init__(self, *args):
        self.pipe = args
    
    def __call__(self, value):
        for fn in self.pipe:
            value = fn(value)
        return value