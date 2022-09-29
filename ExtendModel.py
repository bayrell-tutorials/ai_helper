# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT
##

import os, torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary

from .AbstractModel import AbstractModel
from .Utils import *


LayerFactory_items = {}


def register_layer_factory(key, value):
	
	"""
	Regisyer layer factory
	"""
	
	LayerFactory_items[key] = value


class LayerFactory:
	
	def __init__(self, *args, **kwargs):
		
		"""
		Init
		"""
		
		self.args = args
		self.kwargs = kwargs
	
	
	def get_name(self):
		
		"""
		Returns name
		"""
		
		return self.args[0]
	
	
	def create_layer(self, work_tensor, module):
		
		"""
		Create layer
		"""
		
		layer_out = None
		name = self.get_name()
		
		if name in LayerFactory_items:
			factory_callback = LayerFactory_items[name]
			layer_out, work_tensor = factory_callback.create_layer(self, work_tensor, module)
			
		return layer_out, work_tensor


class Factory_Conv3d:

	def create_layer(self, factory:LayerFactory, work_tensor, module):
		
		"""
		Returns Conv3d
		"""
		
		in_channels = work_tensor.shape[1]
		out_channels = factory.args[1]
		kwargs = factory.kwargs
		
		layer_out = torch.nn.Conv3d(
			in_channels=in_channels,
			out_channels=out_channels,
			**kwargs
		)
		
		work_tensor = layer_out(work_tensor)
		
		return layer_out, work_tensor


class Factory_Conv2d:
	
	def create_layer(self, factory:LayerFactory, work_tensor, module):
	
		"""
		Returns Conv2d
		"""
		
		in_channels = work_tensor.shape[1]
		out_channels = factory.args[1]
		kwargs = factory.kwargs
		
		layer_out = torch.nn.Conv2d(
			in_channels=in_channels,
			out_channels=out_channels,
			**kwargs
		)
		
		work_tensor = layer_out(work_tensor)
		
		return layer_out, work_tensor


class Factory_Dropout:
	
	def create_layer(self, factory:LayerFactory, work_tensor, module):
	
		"""
		Returns Dropout
		"""
		
		p = factory.args[1]
		kwargs = factory.kwargs
		
		layer_out = torch.nn.Dropout(p=p, **kwargs)
		
		return layer_out, work_tensor


class Factory_MaxPool2d:
	
	def create_layer(self, factory:LayerFactory, work_tensor, module):
	
		"""
		Returns MaxPool2d
		"""
		
		kwargs = factory.kwargs
		layer_out = torch.nn.MaxPool2d(**kwargs)
		
		work_tensor = layer_out(work_tensor)
		
		return layer_out, work_tensor


class Factory_Flat:
	
	def create_layer(self, factory:LayerFactory, work_tensor, module):
		
		args = factory.args
		pos = args[1] if len(args) >= 2 else 1
		
		if pos < 0:
			pos = pos - 1
		
		shape = list(work_tensor.shape)
		shape = shape[:pos]
		shape.append(-1)
		
		work_tensor = work_tensor.reshape( shape )
		
		return None, work_tensor


class Factory_InsertFirstAxis:
	
	def create_layer(self, factory:LayerFactory, work_tensor, module):
		
		work_tensor = work_tensor[:,None,:]
		
		return None, work_tensor


class Factory_Linear:
	
	def create_layer(self, factory:LayerFactory, work_tensor, module):
		
		in_features = work_tensor.shape[1]
		out_features = factory.args[1]
		
		layer_out = torch.nn.Linear(
			in_features=in_features,
			out_features=out_features
		)
		
		work_tensor = layer_out(work_tensor)
		
		return layer_out, work_tensor
	


"""
Register factories
"""

register_layer_factory("Conv3d", Factory_Conv3d())
register_layer_factory("Conv2d", Factory_Conv2d())
register_layer_factory("Dropout", Factory_Dropout())
register_layer_factory("MaxPool2d", Factory_MaxPool2d())
register_layer_factory("Flat", Factory_Flat())
register_layer_factory("InsertFirstAxis", Factory_InsertFirstAxis())
register_layer_factory("Linear", Factory_Linear())



def layer(*args, **kwargs):
	
	"""
	Define layer
	"""
	
	return LayerFactory(*args, **kwargs)



class ExtendModule(torch.nn.Module):
	
	def __init__(self, model):
		
		"""
		Constructor
		"""
		
		super(ExtendModule, self).__init__()
		
		self._model = model
		self._layer_shapes = []
		
	
	def forward(self, x):
		
		"""
		Forward model
		"""
		
		return x
	
	
	def init_layers(self, layers, debug=False):
		
		"""
		Init layers
		"""
		
		input_shape = self._model.input_shape
		output_shape = self._model.output_shape
		
		arr = list(input_shape)
		arr.insert(0, 1)
		
		work_tensor = torch.zeros( tuple(arr) )
		self._layer_shapes.append( ("Input", work_tensor.shape) )
		
		if debug:
			print ("Input:" + " " + str( tuple(work_tensor.shape) ))
		
		for index, obj in enumerate(layers):
			
			if isinstance(obj, LayerFactory):
				
				layer_factory: LayerFactory = obj
				name = layer_factory.get_name()
				layer_name = str(index) + "_" + name
				
				layer, work_tensor = layer_factory.create_layer(work_tensor, self)
				
				self._layer_shapes.append( (layer_name, work_tensor.shape) )
				
				if debug:
					print (layer_name + " => " + str(tuple(work_tensor.shape)))
				
				if layer:
					self.add_module(layer_name, layer)
					
	


class ExtendModel(AbstractModel):
	
	
	def __init__(self):
		
		AbstractModel.__init__(self)
		
		self.model_name = ""
	
	
	def get_name(self):
		
		"""
		Returns model name
		"""
		
		return os.path.join("data", "model", self.model_name)
	
	
	def create_model_ex(self, model_name="", layers=[], debug=False):
		
		"""
		Extended create model
		"""
		
		self.model_name = model_name
		self.model = ExtendModule(self)
		self.model.init_layers(layers, debug=debug)
		
		pass
	
	
	def summary(self):
		
		"""
		Show model summary
		"""
		
		summary(self.model)
		
		for arr in self.model._layer_shapes:
			print ( arr[0] + " => " + str(tuple(arr[1])) )
	
	