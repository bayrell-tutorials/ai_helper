# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT
##

import numpy as np
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
        
        if isinstance(t, Image.Image):
            t = torch.from_numpy( np.array(t) )
            t = t.to(torch.uint8)
            t = torch.moveaxis(t, 2, 0)
            return t
        
        t = t * 255
        t = t.to(torch.uint8)
        
        return t


class ToFloatImage(torch.nn.Module):
    
    def __call__(self, t):
        
        if isinstance(t, Image.Image):
            t = torch.from_numpy( np.array(t) )
            t = t.to(torch.uint8)
            t = torch.moveaxis(t, 2, 0)
        
        t = t.to(torch.float)
        t = t / 255.0
        
        return t


class ReadImage:
    
    def __init__(self, mode=None):
        
        self.mode=mode
    
    def __call__(self, t):
        
        t = Image.open(t)
        
        if self.mode is not None:
            t = t.convert(self.mode)
        
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