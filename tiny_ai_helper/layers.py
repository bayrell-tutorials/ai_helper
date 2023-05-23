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


class Lambda(torch.nn.Module):
    
    """
    Lambda layer
    """
    
    def __init__(self, f):
        torch.nn.Module.__init__(self)
        self.f=f
    
    def forward(self, x):
        return self.f(x)


class InsertFirstAxis(torch.nn.Module):
    
    """
    Insert first Axis for convolution layer
    """
    
    def forward(self, t):
        t = t[:,None,:]
        return t


class InsertLastAxis(torch.nn.Module):
    
    """
    Insert last Axis for convolution layer
    """
    
    def forward(self, t):
        t = t[:,None]
        return t


class MoveRGBToEnd(torch.nn.Module):
        
    def forward(self, t):
        l = len(t.shape)
        t = torch.moveaxis(t, l-3, l-1)
        return t


class MoveRGBToBegin(torch.nn.Module):
        
    def forward(self, t):
        l = len(t.shape)
        t = torch.moveaxis(t, l-1, l-3)
        return t


class ToIntImage(torch.nn.Module):
    
    def forward(self, t):
        
        t = t * 255
        t = t.to(torch.uint8)
        
        return t


class ToFloatImage(torch.nn.Module):
    
    def forward(self, t):
        
        t = t.to(torch.float)
        t = t / 255.0
        
        return t


class ToFloat(torch.nn.Module):
    
    def forward(self, t):
        
        t = t.to(torch.float)
        
        return t


class ReadImage(torch.nn.Module):
    
    def __init__(self, mode=None):
        torch.nn.Module.__init__(self)
        self.mode=mode
    
    def forward(self, batch):
        
        res = []
        for t in batch:
        
            t = Image.open(t)
            
            if self.mode is not None and self.mode != t.mode:
                t = t.convert(self.mode)
            
            res.append(t)
        
        return res


class ImageToTensor(torch.nn.Module):
    
    def forward(self, batch):
        
        res = torch.tensor([])
        for t in batch:
            
            t = torch.from_numpy( np.array(t) )
            t = t[None, :]
            res = torch.cat( (res, t) )
        
        return res


class ResizeImage(torch.nn.Module):
    
    def __init__(self, size, contain=True, color=None):
        
        torch.nn.Module.__init__(self)
        
        self.size = size
        self.contain = contain
        self.color = color
    
    def forward(self, batch):
        
        res = []
        for t in batch:
            t = resize_image(t, self.size, contain=self.contain, color=self.color)
            res.append(t)
        
        return res


class NormalizeImage(torch.nn.Module):
    
    def __init__(self, mean, std, inplace=False):
        
        import torchvision
        
        torch.nn.Module.__init__(self)
        
        self.mean = mean
        self.std = std
        self.inplace = inplace
        self.normalize = torchvision.transforms.Normalize(mean=mean, std=std, inplace=inplace)
    
    def forward(self, t):
        
        t = self.normalize(t)
        
        return t
    
    def extra_repr(self) -> str:
        return 'mean={}, std={}, inplace={}'.format(
            self.mean, self.std, self.inplace
        )


class PreparedModule(torch.nn.Module):
    
    def __init__(self, module, weight_path=None, forward=None, *args, **kwargs):
        
        torch.nn.Module.__init__(self)
        
        self.module = module
        self.weight_path = weight_path
        self._forward = forward
        
        for param in self.module.parameters():
            param.requires_grad = False
        
        self.load_weight()
    
    def forward(self, x):
        
        if self._forward:
            x = self._forward(self, x)
        else:
            x = self.module(x)
            
        return x
    
    def load_weight(self):
        """
        Load weight
        """
        if self.weight_path:
            state_dict = torch.load( self.weight_path )
            self.module.load_state_dict( state_dict )
    
    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        pass
    
    
class Stacking(torch.nn.Module):
    
    def __init__(self, *args, is_tensor_list=True):
        
        torch.nn.Module.__init__(self)
        
        for i, module in enumerate(args):
            self.add_module(str(i), module)
        
        self.is_tensor_list = is_tensor_list
    
    def forward(self, tensor_list):
        
        device = tensor_list[0].device
        res = torch.tensor([]).to(device)
        
        keys = list(self._modules.keys())
        for index, m in enumerate(keys):
            
            if self.is_tensor_list:
                x = tensor_list[index]
            else:
                x = tensor_list
            
            if self._modules[m] is not None:
                module = self._modules[m]
                x = module(x)
            
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
        torch.nn.Module.__init__(self)
        self.pipe = args
    
    def forward(self, value):
        for fn in self.pipe:
            value = fn(value)
        return value


class ModuleRemoveLastClassifier(PreparedModule):
    
    def __init__(self, module, weight_path=None, *args, **kwargs):
        
        PreparedModule.__init__(self, module, weight_path, *args, **kwargs)
        
        # Remove last layer
        classifier = list(self.module.classifier.children())
        classifier = classifier[:-1]
        self.module.classifier = torch.nn.Sequential(*classifier)


class ModuleRemoveAllClassifier(PreparedModule):
    
    def __init__(self, module, weight_path=None, *args, **kwargs):
        PreparedModule.__init__(self, module, weight_path, *args, **kwargs)
    
    def forward(self, x):
        x = self.module.features(x)
        x = self.module.avgpool(x)
        return x