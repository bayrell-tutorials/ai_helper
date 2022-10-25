# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT
##

import torch
from .Utils import *


class AbstractLayerFactory:
	
	
	def __init__(self, *args, **kwargs):
		
		"""
		Constructor
		"""
		
		self.module = None
		self.parent:ExtendModule = None
		self.layer_name = ""
		self.input_shape = None
		self.output_shape = None
		self.args = args
		self.kwargs = kwargs
	
	
	def get_name(self):
		
		"""
		Returns name
		"""
		
		return ""
	
	
	def create_layer(self, vector_x):
		
		"""
		Create new layer
		"""
		
		if self.module:
			vector_x = self.module(vector_x)
		
		return self.module, vector_x

	
	def forward(self, vector_x):
		
		if self.module:
			return self.module(vector_x)
			
		return vector_x
	
	
class Conv3d(AbstractLayerFactory):
	
	
	def __init__(self, out_channels, *args, **kwargs):
		
		AbstractLayerFactory.__init__(self, *args, **kwargs)
		self.out_channels = out_channels
	
	
	def get_name(self):
		return "Conv3d"
	
	
	def create_layer(self, vector_x):
		
		"""
		Returns Conv3d
		"""
		
		in_channels = vector_x.shape[1]
		out_channels = self.out_channels
		kwargs = self.kwargs
		
		self.module = torch.nn.Conv3d(
			in_channels=in_channels,
			out_channels=out_channels,
			**kwargs
		)
		
		vector_x = self.module(vector_x)
		
		return self.module, vector_x


class Conv2d(AbstractLayerFactory):
	
	
	def __init__(self, out_channels, *args, **kwargs):
		
		AbstractLayerFactory.__init__(self, *args, **kwargs)
		self.out_channels = out_channels
	
	
	def get_name(self):
		return "Conv2d"
	
	
	def create_layer(self, vector_x):
	
		"""
		Returns Conv2d
		"""
		
		in_channels = vector_x.shape[1]
		out_channels = self.out_channels
		kwargs = self.kwargs
		
		self.module = torch.nn.Conv2d(
			in_channels=in_channels,
			out_channels=out_channels,
			**kwargs
		)
		
		vector_x = self.module(vector_x)
		
		return self.module, vector_x


class Dropout(AbstractLayerFactory):
	
	
	def __init__(self, p, *args, **kwargs):
		
		AbstractLayerFactory.__init__(self, *args, **kwargs)
		self.p = p
	
	
	def get_name(self):
		return "Dropout"
	
	
	def create_layer(self, vector_x):
	
		"""
		Returns Dropout
		"""
		
		kwargs = self.kwargs
		layer_out = torch.nn.Dropout(p=self.p, **kwargs)
		
		return layer_out, vector_x


class MaxPool2d(AbstractLayerFactory):
	
	
	def get_name(self):
		return "MaxPool2d"
	
	
	def create_layer(self, vector_x):
	
		"""
		Returns MaxPool2d
		"""
		
		kwargs = self.kwargs
		self.module = torch.nn.MaxPool2d(**kwargs)
		
		vector_x = self.module(vector_x)
		
		return self.module, vector_x


class Linear(AbstractLayerFactory):
	
	def __init__(self, out_features, *args, **kwargs):
		
		AbstractLayerFactory.__init__(self, *args, **kwargs)
		self.out_features = out_features
	
	
	def get_name(self):
		return "Linear"
	
	
	def create_layer(self, vector_x):
		
		in_features = vector_x.shape[1]
		out_features = self.out_features
		
		self.module = torch.nn.Linear(
			in_features=in_features,
			out_features=out_features
		)
		
		vector_x = self.module(vector_x)
		
		return self.module, vector_x


class Relu(AbstractLayerFactory):
	
	def get_name(self):
		return "Relu"
	
	def forward(self, vector_x):
		vector_x = torch.nn.functional.relu(vector_x)
		return vector_x
	
	def create_layer(self, vector_x):
		return None, vector_x
	

class Softmax(AbstractLayerFactory):
	
	def get_name(self):
		return "Softmax"
	
	def create_layer(self, vector_x):
		
		dim = self.kwargs["dim"] if "dim" in self.kwargs else -1
		self.module = torch.nn.Softmax(dim)
		
		return self.module, vector_x


class Model_Save(AbstractLayerFactory):
	
	def get_name(self):
		return "Save"
	
	def forward(self, vector_x):
		
		save_name = self.args[1] if len(self.args) >= 2 else ""
		
		if save_name:
			self.parent._saves[save_name] = vector_x
		
		return vector_x
	
	def create_layer(self, vector_x):
		return None, vector_x
	
	
class Model_Concat(AbstractLayerFactory):
	
	def get_name(self):
		return "Concat"
	
	def forward(self, vector_x):
		
		save_name = self.args[1] if len(self.args) >= 2 else ""
		dim = self.kwargs["dim"] if "dim" in self.kwargs else 1
		
		if save_name and save_name in self.parent._saves:
			save_x = self.parent._saves[save_name]
			vector_x = torch.cat([vector_x, save_x], dim=dim)
		
		return vector_x
	
	def create_layer(self, vector_x):
		return None, vector_x


class Layer(AbstractLayerFactory):
	
	def __init__(self, name, module, *args, **kwargs):
		
		AbstractLayerFactory.__init__(self, *args, **kwargs)
		self.name = name
		self.module = module
	
	def get_name(self):
		return self.name



class Transform_Flat(torch.nn.Module):
	
	def __init__(self, pos=1):
		
		torch.nn.Module.__init__(self)
		
		self.pos = pos
	
	def __call__(self, t):
		
		pos = self.pos
		
		if pos < 0:
			pos = pos - 1
		
		shape = list(t.shape)
		shape = shape[:pos]
		shape.append(-1)
		
		t = t.reshape( shape )
		
		return t


def Flat():
	return Layer("Flat", Transform_Flat())


class Transform_InsertFirstAxis(torch.nn.Module):
	
	"""
	Insert first Axis for convolution layer
	"""
	
	def __call__(self, t):
		t = t[:,None,:]
		return t


def InsertFirstAxis():
	
	"""
	Insert first Axis for convolution layer
	"""
	
	return Layer("InsertFirstAxis", Transform_InsertFirstAxis())


class Transform_MoveRGBToEnd(torch.nn.Module):
		
	def __call__(self, t):
		l = len(t.shape)
		t = torch.moveaxis(t, l-3, l-1)
		return t


def ToIntImage():
	return Layer("MoveRGBToEnd", Transform_MoveRGBToEnd())


class Transform_MoveRGBToBegin(torch.nn.Module):
		
	def __call__(self, t):
		l = len(t.shape)
		t = torch.moveaxis(t, l-1, l-3)
		return t


def ToIntImage():
	return Layer("MoveRGBToBegin", Transform_MoveRGBToBegin())
		

class Transform_ToIntImage(torch.nn.Module):
	
	def __call__(self, t):
		
		if isinstance(t, Image.Image):
			t = torch.from_numpy( np.array(t) )
			t = t.to(torch.uint8)
			t = torch.moveaxis(t, 2, 0)
			return t
		
		t = t * 255
		t = t.to(torch.uint8)
		
		return t


def ToIntImage():
	return Layer("ToIntImage", Transform_ToIntImage())


class Transform_ToFloatImage(torch.nn.Module):
	
	def __call__(self, t):
		
		if isinstance(t, Image.Image):
			t = torch.from_numpy( np.array(t) )
			t = t.to(torch.uint8)
			t = torch.moveaxis(t, 2, 0)
		
		t = t.to(torch.float)
		t = t / 255.0
		
		return t


def ToFloatImage():
	return Layer("ToFloatImage", Transform_ToFloatImage())


class Transform_ReadImage:
	
	def __call__(self, t):
		
		t = Image.open(t)
		return t
		

class Transform_Resize(torch.nn.Module):
	
	def __init__(self, size, contain=True, color=None):
		
		torch.nn.Module.__init__(self)
		
		self.size = size
		self.contain = contain
		self.color = color
	
	def __call__(self, t):
		
		t = resize_image(t, self.size, contain=self.contain, color=self.color)
		
		return t

def Resize(size, contain=True, color=None):
	return Layer("Resize", Transform_Resize(size, contain=contain, color=color))



def ImageNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
	return Layer("ImageNormalize",
		transforms.Normalize(mean=mean, std=std)
	)
