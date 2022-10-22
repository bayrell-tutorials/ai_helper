# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT
##

import os, torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchsummary import summary
from .Utils import *
from .ModelDatabase import *


class AbstractModel:
	
	
	def __init__(self):
		
		from .TrainStatus import TrainStatus
		self.train_status = TrainStatus()
		self.train_status.model = self
		self.train_loader = None
		self.test_loader = None
		self.train_dataset = None
		self.test_dataset = None
		self.batch_size = 64
		self.module = None
		self.optimizer = None
		self.loss = None
		self.verbose = True
		self.num_workers = os.cpu_count()
		self.max_epochs = 50
		self.min_epochs = 10
		self.max_acc = 0.95
		self.max_acc_rel = 1.5
		self.min_loss_test = 0.001
		self.input_shape = (1)
		self.output_shape = (1)
		self.onnx_path = "web"
		self.onnx_opset_version = 9
		self.model_name = ""
		self.model_database = ModelDatabase()
		self.model_database.set_path( os.path.join(os.getcwd(), "data", "model") )
		
		self._is_debug = False
		self._is_trained = False
		
		
	def debug(self, value):
		
		"""
		Set debug level
		"""
		
		self._is_debug = value
		
		
	def print_debug(self, *args):
		
		"""
		Print if debug level is True
		"""
		
		if self._is_debug:
			print(*args)
		
		
	def get_tensor_device(self):
		
		"""
		Returns tensor device name
		"""
		
		return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
		
		
	def get_model_name(self):
		
		"""
		Returns model name
		"""
		
		return self.model_name
	
	
	def get_model_path(self):
		
		"""
		Returns model path
		"""
		
		return os.path.join(os.getcwd(), "data", "model", self.model_name + ".zip")
	
	
	def get_onnx_path(self):
		
		"""
		Returns model onnx path
		"""
		
		return os.path.join(os.getcwd(), self.onnx_path, self.model_name + ".onnx")
	
	
	def is_loaded(self):
		
		"""
		Returns true if model is loaded
		"""
		
		return self.module is not None
	
	
	def is_trained(self):
		
		"""
		Returns true if model is loaded
		"""
		
		return self.is_loaded() and self._is_trained
	
	
	def load_dataset(self, **kwargs):
		
		"""
		Load dataset
		"""
		
		type = None
		count = None
		
		if "type" in kwargs:
			type = kwargs["type"]
		
		if "count" in kwargs:
			count = kwargs["count"]
		
		if type == "train":
			
			self.train_dataset, self.test_dataset = self.get_train_dataset()
	
	
	def get_train_dataset(self, **kwargs):
		
		"""
		Returns normalized train and test datasets
		"""
		
		train_dataset = TensorDataset( torch.tensor(), torch.tensor() )
		test_dataset = TensorDataset( torch.tensor(), torch.tensor() )
		
		return train_dataset, test_dataset
	
	
	def get_train_data_count(self):
		
		"""
		Returns train data count
		"""
		
		if (self.train_dataset is not None and
			isinstance(self.train_dataset, Dataset)):
				return len(self.train_dataset)
		
		return 1
	
	
	def get_test_data_count(self):
		
		"""
		Returns test data count
		"""
		
		if (self.test_dataset is not None and
			isinstance(self.test_dataset, TensorDataset)):
				return self.test_dataset.tensors[0].shape[0]
		return 1
	
	
	def convert_batch(self, x=None, y=None):
		
		"""
		Convert batch
		"""
		
		if x is not None:
			x = x.to(torch.float)
		
		if y is not None:
			y = y.to(torch.float)
		
		return x, y
	
	
	def create_model(self):
		
		"""
		Create model
		"""
		
		self.module = None
		self._is_trained = False
	
	
	def create_model_ex(self, model_name="", layers=[], debug=False):
		
		"""
		Create extended model
		"""
		
		self.model_name = model_name
		self.module = ExtendModule(self)
		self.module.init_layers(layers, debug=debug)
		self._is_trained = False
	
	
	def summary(self):
		
		"""
		Show model summary
		"""
		
		tensor_device = self.get_tensor_device()
		module = self.module.to(tensor_device)
		summary(self.module, self.input_shape, device=str(tensor_device))
		
		if (isinstance(self.module, ExtendModule)):
			for arr in self.module._shapes:
				print ( arr[0] + " => " + str(tuple(arr[1])) )
	
	
	def save(self, file_name=None):
		
		"""
		Save model to file
		"""
		
		self.model_database.save(self.model_name, self.module, self.train_status)
	
	
	def save_onnx(self, tensor_device=None):
		
		"""
		Save model to onnx file
		"""
		
		import torch, torch.onnx
		
		if tensor_device is None:
			tensor_device = self.get_tensor_device()
		
		onnx_model_path = self.get_onnx_path()
		
		# Prepare data input
		data_input = torch.zeros(self.input_shape).to(torch.float32)
		data_input = data_input[None,:]
		
		# Move to tensor device
		model = self.module.to(tensor_device)
		data_input = data_input.to(tensor_device)
		
		torch.onnx.export(
			model,
			data_input,
			onnx_model_path,
			opset_version = self.onnx_opset_version,
			input_names = ['input'],
			output_names = ['output'],
			verbose=False
		)
	
	
	def load(self):
		
		"""
		Load model from file
		"""
		
		is_loaded = self.model_database.load(self.model_name, self.module, self.train_status)
		if is_loaded:
			self._is_trained = self.check_is_trained()
		
		
		
	def check_answer(self, **kwargs):
		
		"""
		Check answer
		"""
		
		tensor_y = kwargs["tensor_y"]
		tensor_predict = kwargs["tensor_predict"]
		
		y = get_answer_from_vector(tensor_y)
		predict = get_answer_from_vector(tensor_predict)
		
		return predict == y
		
		
	def check_answer_batch(self, **kwargs):
		
		"""
		Check batch. Returns count right answers
		"""
		
		res = 0
		
		type = kwargs["type"]
		batch_x = kwargs["batch_x"]
		batch_y = kwargs["batch_y"]
		batch_predict = kwargs["batch_predict"]
		
		for i in range(batch_x.shape[0]):
			
			tensor_x = batch_x[i]
			tensor_y = batch_y[i]
			tensor_predict = batch_predict[i]
			
			flag = self.check_answer(
				tensor_x=tensor_x,
				tensor_y=tensor_y,
				tensor_predict=tensor_predict,
				type=type,
			)
			
			if flag:
				res = res + 1
		
		return res
	
	
	def check_is_trained(self):
		
		"""
		Returns True if model is trained
		"""
		
		epoch_number = self.train_status.epoch_number
		acc_train = self.train_status.get_acc_train()
		acc_test = self.train_status.get_acc_test()
		acc_rel = self.train_status.get_acc_rel()
		loss_test = self.train_status.get_loss_test()
		
		if epoch_number >= 50:
			return True
		
		if acc_train > 0.95 and epoch_number >= 10:
			return True
		
		if acc_test > 0.95  and epoch_number >= 10:
			return True
		
		if acc_rel > 1.5 and acc_train > 0.8:
			return True
		
		if loss_test < 0.001 and epoch_number >= 10:
			return True
		
		return False
	
		
	def on_end_epoch(self, **kwargs):
		
		"""
		On epoch end
		"""
		
		self._is_trained = self.check_is_trained()
		
		if self._is_trained:
			self.stop_training()
		
		self.save()
		
		
	def stop_training(self):
		
		"""
		Stop training
		"""
		
		self.train_status.stop_train()
		
	
	def train(self, tensor_device=None):
		
		"""
		Train model
		"""
		
		# Adam optimizer
		if self.optimizer is None:
			self.optimizer = torch.optim.Adam(self.module.parameters(), lr=3e-4)
		
		# Mean squared error
		if self.loss is None:
			self.loss = torch.nn.MSELoss()
		
		if tensor_device is None:
			tensor_device = self.get_tensor_device()
		
		if self.train_loader is None and self.train_dataset is not None:
			self.train_loader = DataLoader(
				self.train_dataset,
				num_workers=self.num_workers,
				batch_size=self.batch_size,
				drop_last=False,
				shuffle=True
			)
		
		if self.test_loader is None and self.test_dataset is not None:
			self.test_loader = DataLoader(
				self.test_dataset,
				num_workers=self.num_workers,
				batch_size=self.batch_size,
				drop_last=False,
				shuffle=False
			)
		
		module = self.module.to(tensor_device)
		
		# Do train
		train_status = self.train_status
		train_status.epoch_number = 1
		train_status.do_training = True
		train_status.train_data_count = self.get_train_data_count()
		train_status.on_start_train()
		
		try:
		
			while True:
				
				train_status.clear_iter()
				train_status.on_start_epoch()
				
				# Train batch
				for batch_x, batch_y in self.train_loader:
					
					batch_x = batch_x.to(tensor_device)
					batch_y = batch_y.to(tensor_device)
					batch_x, batch_y = self.convert_batch(x=batch_x, y=batch_y)
					
					train_status.on_start_batch_train(batch_x, batch_y)
					
					# Predict model
					batch_predict = module(batch_x)

					# Get loss value
					loss_value = self.loss(batch_predict, batch_y)
					train_status.loss_train = train_status.loss_train + loss_value.item()
					
					# Gradient
					self.optimizer.zero_grad()
					loss_value.backward()
					self.optimizer.step()
					
					# Calc accuracy
					accuracy = self.check_answer_batch(
						train_status = train_status,
						batch_x = batch_x,
						batch_y = batch_y,
						batch_predict = batch_predict,
						type = "train"
					)
					train_status.acc_train = train_status.acc_train + accuracy
					train_status.batch_train_iter = train_status.batch_train_iter + 1
					train_status.train_count = train_status.train_count + batch_x.shape[0]
					
					train_status.on_end_batch_train(batch_x, batch_y)
					
					del batch_x, batch_y
					
					# Clear CUDA
					if torch.cuda.is_available():
						torch.cuda.empty_cache()
				
					
				# Test batch
				for batch_x, batch_y in self.test_loader:
					
					batch_x = batch_x.to(tensor_device)
					batch_y = batch_y.to(tensor_device)
					batch_x, batch_y = self.convert_batch(x=batch_x, y=batch_y)
					
					train_status.on_start_batch_test(batch_x, batch_y)
					
					# Predict model
					batch_predict = module(batch_x)
					
					# Get loss value
					loss_value = self.loss(batch_predict, batch_y)
					train_status.loss_test = train_status.loss_test + loss_value.item()
					
					# Calc accuracy
					accuracy = self.check_answer_batch(
						train_status = train_status,
						batch_x = batch_x,
						batch_y = batch_y,
						batch_predict = batch_predict,
						type = "test"
					)
					train_status.acc_test = train_status.acc_test + accuracy
					train_status.batch_test_iter = train_status.batch_test_iter + 1
					train_status.test_count = train_status.test_count + batch_x.shape[0]
					
					train_status.on_end_batch_test(batch_x, batch_y)
					
					del batch_x, batch_y
					
					# Clear CUDA
					if torch.cuda.is_available():
						torch.cuda.empty_cache()
				
				# Epoch callback
				train_status.on_end_epoch()
				
				if not train_status.do_training:
					break
				
				train_status.epoch_number = train_status.epoch_number + 1
			
			self._is_trained = True
			
		except KeyboardInterrupt:
			
			print ("")
			print ("Stopped manually")
			print ("")
			
			pass
		
		train_status.on_end_train()
		
		
	def show_train_history(self):
		
		"""
		Show train history
		"""
		
		file_name, _ = os.path.splitext( self.model_database.get_model_path(self.model_name) )
		history_image = file_name + ".png"
		
		make_parent_dir(history_image)
		
		fig, axs = plt.subplots(2)
		axs[0].plot( np.multiply(self.train_status.history['loss_train'], 100), label='train loss')
		axs[0].plot( np.multiply(self.train_status.history['loss_test'], 100), label='test loss')
		axs[0].legend()
		axs[1].plot( np.multiply(self.train_status.history['acc_train'], 100), label='train acc')
		axs[1].plot( np.multiply(self.train_status.history['acc_test'], 100), label='test acc')
		axs[1].legend()
		fig.supylabel('Percent')
		plt.xlabel('Epoch')
		plt.savefig(history_image)
		plt.show()
		
		
		
	def predict(self, vector_x, tensor_device=None):
		
		"""
		Predict model
		"""
		
		if tensor_device is None:
			tensor_device = self.get_tensor_device()
		
		vector_x = vector_x.to(tensor_device)
		module = self.module.to(tensor_device)
		vector_x, _ = self.convert_batch(x=vector_x)
		
		vector_y = module(vector_x)
		
		return vector_y
	


def do_train(model:AbstractModel, summary=False):
	
	"""
	Start training
	"""
	
	if summary:
		model.summary()
			
	# Load dataset
	model.load_dataset(type="train")
	
	# Train the model
	if model.is_trained():
		model.train()
		model.show_train_history()
		


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
		
		return None, vector_x

	
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


class Flat(AbstractLayerFactory):
	
	def __init__(self, pos=1, *args, **kwargs):
		
		AbstractLayerFactory.__init__(self, *args, **kwargs)
		self.pos = pos
	
	
	def get_name(self):
		return "Flat"
	
	
	def forward(self, vector_x):
		
		args = self.args
		pos = self.pos
		
		if pos < 0:
			pos = pos - 1
		
		shape = list(vector_x.shape)
		shape = shape[:pos]
		shape.append(-1)
		
		vector_x = vector_x.reshape( shape )
		
		return vector_x
	
	
	def create_layer(self, vector_x):
		
		vector_x = self.forward(vector_x)
		
		return None, vector_x


class InsertFirstAxis(AbstractLayerFactory):
	
	"""
	Insert first Axis for convolution layer
	"""
	
	def get_name(self):
		return "InsertFirstAxis"
	
	
	def forward(self, vector_x):
		
		vector_x = vector_x[:,None,:]
		
		return vector_x
	
	
	def create_layer(self, vector_x):
		
		vector_x = self.forward(vector_x)
		
		return None, vector_x


class MoveRGBToEnd(AbstractLayerFactory):
	
	"""
	Move RGB channel to end
	"""
	
	def get_name(self):
		return "MoveRGBToEnd"
	
	
	def forward(self, vector_x):
		
		vector_x = torch.moveaxis(vector_x, 1, 3)
		
		return vector_x
	
	
	def create_layer(self, vector_x):
		
		vector_x = self.forward(vector_x)
		
		return None, vector_x


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
	
	def __init__(self, name, layer, *args, **kwargs):
		
		AbstractLayerFactory.__init__(self, *args, **kwargs)
		self.name = name
		self.layer = layer
	
	def get_name(self):
		return self.name
	
	def create_layer(self, vector_x):
		return self.layer, vector_x
	

class ExtendModule(torch.nn.Module):
	
	def __init__(self, model):
		
		"""
		Constructor
		"""
		
		super(ExtendModule, self).__init__()
		
		self._model = model
		self._layers = []
		self._shapes = []
		self._saves = {}
		
	
	def forward(self, x):
		
		"""
		Forward model
		"""
		
		for index, obj in enumerate(self._layers):
			
			if isinstance(obj, AbstractLayerFactory):
				
				layer_factory: AbstractLayerFactory = obj
				x = layer_factory.forward(x)
				
				
		return x
	
	
	def init_layers(self, layers=[], debug=False):
		
		"""
		Init layers
		"""
		
		self._layers = layers
		
		input_shape = self._model.input_shape
		output_shape = self._model.output_shape
		
		arr = list(input_shape)
		arr.insert(0, 1)
		
		vector_x = torch.zeros( tuple(arr) )
		self._shapes.append( ("Input", vector_x.shape) )
		
		if debug:
			print ("Input:" + " " + str( tuple(vector_x.shape) ))
		
		index = 1
		
		for obj in self._layers:
			
			if isinstance(obj, AbstractLayerFactory):
				
				layer_factory: AbstractLayerFactory = obj
				layer_factory.parent = self
				
				name = layer_factory.get_name()
				layer_name = str( index ) + "_" + name
				layer_factory.layer_name = layer_name
				layer_factory.input_shape = vector_x.shape
				
				layer, vector_x = layer_factory.create_layer(vector_x)
				layer_factory.output_shape = vector_x.shape
				
				self._shapes.append( (layer_name, vector_x.shape) )
				
				if debug:
					print (layer_name + " => " + str(tuple(vector_x.shape)))
				
				if layer:
					self.add_module(layer_name, layer)
					
				index = index + 1



class TransformMoveRGBToEnd:
		
	def __call__(self, t):
		t = torch.moveaxis(t, 0, 2)
		return t
		

class TransformToIntImage:
	
	def __call__(self, t):
		t = t * 255
		t = t.to(torch.uint8)
		
		return t
