# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT
##

import os, time, torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchsummary import summary
from .Utils import *
from .Layers import *
from .ModelDatabase import *


class AbstractModel:
	
	
	def __init__(self, *args, **kwargs):
		
		from .TrainStatus import TrainStatus
		self.train_status = TrainStatus()
		self.train_status.model = self
		self.train_status.epoch_number = 0
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
		#self.num_workers = 0
		
		self.max_epochs = kwargs["max_epochs"] if "max_epochs" in kwargs else 50
		self.min_epochs = kwargs["min_epochs"] if "min_epochs" in kwargs else 10
		self.max_acc = kwargs["max_acc"] if "max_acc" in kwargs else 0.95
		self.max_acc_rel = kwargs["max_acc_rel"] if "max_acc_rel" in kwargs else 1.5
		self.min_loss_test = kwargs["min_loss_test"] if "min_loss_test" in kwargs else 0.001
		self.batch_size = kwargs["batch_size"] if "batch_size" in kwargs else 64
		self.input_shape = kwargs["input_shape"] if "input_shape" in kwargs else (1)
		self.output_shape = kwargs["output_shape"] if "output_shape" in kwargs else (1)
		
		self.onnx_path = kwargs["onnx_path"] \
			if "onnx_path" in kwargs else os.path.join(os.getcwd(), "web")
		self.onnx_opset_version = kwargs["onnx_opset_version"] \
			if "onnx_opset_version" in kwargs else 9
		self.model_name = kwargs["model_name"] if "model_name" in kwargs else ""
		
		self.model_path = kwargs["model_path"] \
			if "model_path" in kwargs else os.path.join(os.getcwd(), "data", "model")
		
		self.model_database = ModelDatabase()
		self.model_database.set_path( self.model_path )
		
		self._is_trained = False
		self._is_debug = kwargs["debug"] if "debug" in kwargs else False
		self._read_tensor = kwargs["read_tensor"] if "read_tensor" in kwargs else None
		self._convert_batch = kwargs["convert_batch"] if "convert_batch" in kwargs else None
		self._check_is_trained = kwargs["check_is_trained"] \
			if "check_is_trained" in kwargs else None
			
		self.layers = kwargs["layers"] if "layers" in kwargs else None
		self.train_dataset = kwargs["train_dataset"] if "train_dataset" in kwargs else False
		self.test_dataset = kwargs["test_dataset"] if "test_dataset" in kwargs else False
		
		if self.layers is not None:
			self.create_model_ex(
				model_name = self.model_name,
				layers = self.layers,
				debug = self._is_debug,
			)
		
	def debug(self, value):
		
		"""
		Set debug level
		"""
		
		self._is_debug = value
		
		
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
	
	
	def get_onnx_path(self):
		
		"""
		Returns model onnx path
		"""
		
		return os.path.join(self.onnx_path, self.model_name + ".onnx")
	
	
	def is_trained(self):
		
		"""
		Returns true if model is loaded
		"""
		
		return (self.module is not None) and self._is_trained
	
	
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
		
		if self._convert_batch is not None:
			return self._convert_batch(self, x=x, y=y)
		
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
		
		if debug:
			self.summary()
	
	
	def summary(self):
		
		"""
		Show model summary
		"""
		
		tensor_device = self.get_tensor_device()
		module = self.module.to(tensor_device)
		summary(self.module, tuple(self.input_shape), device=str(tensor_device))
		
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
	
	
	def load(self, epoch_number=None):
		
		"""
		Load model from file
		"""
		
		is_loaded = self.model_database.load(self.model_name,
			self.module, self.train_status,
			epoch_number=epoch_number
		)
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
		
		if self._check_is_trained is not None:
			return self._check_is_trained(self)
		
		epoch_number = self.train_status.epoch_number
		acc_train = self.train_status.get_acc_train()
		acc_test = self.train_status.get_acc_test()
		acc_rel = self.train_status.get_acc_rel()
		loss_test = self.train_status.get_loss_test()
		
		if epoch_number >= self.max_epochs:
			return True
		
		if acc_train > self.max_acc and epoch_number >= self.min_epochs:
			return True
		
		if acc_test > self.max_acc  and epoch_number >= self.min_epochs:
			return True
		
		if acc_rel > self.max_acc_rel and acc_train > 0.8:
			return True
		
		if loss_test < self.min_loss_test and epoch_number >= self.min_epochs:
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
		
		if self._is_trained:
			return
		
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
		train_status.do_training = True
		train_status.train_data_count = self.get_train_data_count()
		train_status.on_start_train()
		
		try:
		
			while True:
				
				train_status.clear_iter()
				train_status.epoch_number = train_status.epoch_number + 1
				train_status.time_start = time.time()
				train_status.on_start_epoch()
				
				module.train()
				
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
					train_status.loss_train_iter = train_status.loss_train_iter + loss_value.item()
					
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
					train_status.acc_train_iter = train_status.acc_train_iter + accuracy
					train_status.batch_train_iter = train_status.batch_train_iter + 1
					train_status.train_count_iter = train_status.train_count_iter + batch_x.shape[0]
					
					train_status.time_end = time.time()
					train_status.on_end_batch_train(batch_x, batch_y)
					
					del batch_x, batch_y
					
					# Clear CUDA
					if torch.cuda.is_available():
						torch.cuda.empty_cache()
				
				module.eval()
				
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
					train_status.loss_test_iter = train_status.loss_test_iter + loss_value.item()
					
					# Calc accuracy
					accuracy = self.check_answer_batch(
						train_status = train_status,
						batch_x = batch_x,
						batch_y = batch_y,
						batch_predict = batch_predict,
						type = "test"
					)
					train_status.acc_test_iter = train_status.acc_test_iter + accuracy
					train_status.batch_test_iter = train_status.batch_test_iter + 1
					train_status.test_count_iter = train_status.test_count_iter + batch_x.shape[0]
					
					train_status.time_end = time.time()
					train_status.on_end_batch_test(batch_x, batch_y)
					
					del batch_x, batch_y
					
					# Clear CUDA
					if torch.cuda.is_available():
						torch.cuda.empty_cache()
				
				# Epoch callback
				train_status.time_end = time.time()
				train_status.on_end_epoch()
				self.on_end_epoch()
				
				if not train_status.do_training:
					break
			
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
		if fig.supylabel:
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
	
	# Train the model
	if not model.is_trained():
		print ("Train model " + str(model.model_name))
		model.train()
		


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
			print ("Debug model " + str(self._model.model_name))
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

