# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT
##

import torch, os
from PIL import Image, ImageDraw


def append_tensor(res, t):
	
	"""
	Append tensor
	"""
	
	t = t[None, :]
	res = torch.cat( (res, t) )
	return res


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


def resize_image(image, size, contain=True, color=None):
    """
    Resize image canvas
    """
    
    if contain:
        image_new = image.copy()
        image_new.thumbnail(size, Image.LANCZOS)
        image_new = resize_image_canvas(image_new, size, color=color)
        return image_new
    
    width, height = image.size
    rect = (width, width)
    if width > height:
        rect = (height, height)
    
    image_new = resize_image_canvas(image, rect, color=color)
    image_new.thumbnail(size, Image.Resampling.LANCZOS)
    
    return image_new
    

def resize_image_canvas(image, size, color=None):
    """
    Resize image canvas
    """
    
    width, height = size
    
    if color == None:
        pixels = image.load()
        color = pixels[0, 0]
        del pixels
        
    image_new = Image.new(image.mode, (width, height), color = color)
    draw = ImageDraw.Draw(image_new)
    
    position = (
        math.ceil((width - image.size[0]) / 2),
        math.ceil((height - image.size[1]) / 2),
    )
    
    image_new.paste(image, position)
    
    del draw, image
    
    return image_new


def list_files(path="", recursive=True):
	
	"""
		Returns files in folder
	"""
	
	def read_dir(path, recursive=True):
		res = []
		items = os.listdir(path)
		for item in items:
			
			item_path = os.path.join(path, item)
			
			if item_path == "." or item_path == "..":
				continue
			
			if os.path.isdir(item_path):
				if recursive:
					res = res + read_dir(item_path, recursive)
			else:
				res.append(item_path)
			
		return res
	
	try:
		items = read_dir( path, recursive )
			
		def f(item):
			return item[len(path + "/"):]
		
		items = list( map(f, items) )
	
	except Exception:
		items = []
	
	return items


def list_dirs(path=""):
	
	"""
		Returns dirs in folder
	"""
	
	try:
		items = os.listdir(path)
	except Exception:
		items = []
    
	return items