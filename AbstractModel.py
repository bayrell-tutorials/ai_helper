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


class AbstractModel:
	
	
	def __init__(self):
		
		from .TrainStatus import TrainStatus
		self.train_status = TrainStatus()
		self.train_status.model = self
		self.train_loader = None
		self.test_loader = None
		self.control_loader = None
		self.train_dataset = None
		self.test_dataset = None
		self.control_dataset = None
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
		self.onnx_opset_version = 9
		self.model_name = ""
		
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
		
		
	def get_name(self):
		
		"""
		Returns model name
		"""
		
		return os.path.join("data", "model", self.model_name)
	
	
	def get_path(self):
		
		"""
		Returns model path
		"""
		
		return self.get_name() + ".zip"
	
	
	def get_path_onnx(self):
		
		"""
		Returns model onnx path
		"""
		
		return self.get_name() + ".onnx"
	
	
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
		
		if type == "control":
			
			self.control_dataset = self.get_control_dataset(count=count)
	
	
	def get_train_dataset(self, **kwargs):
		
		"""
		Returns normalized train and test datasets
		"""
		
		train_dataset = TensorDataset( torch.tensor(), torch.tensor() )
		test_dataset = TensorDataset( torch.tensor(), torch.tensor() )
		
		return train_dataset, test_dataset
	
	
	def get_control_dataset(self, **kwargs):
		
		"""
		Returns normalized control dataset
		"""
		
		dataset = TensorDataset( torch.tensor(), torch.tensor() )
		return dataset
		
	
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
	
	
	def get_control_data_count(self):
		
		"""
		Returns control data count
		"""
		
		if (self.control_dataset is not None and
			isinstance(self.control_dataset, TensorDataset)):
				return self.control_dataset.tensors[0].shape[0]
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
		
		pass
	
	
	def summary(self):
		
		"""
		Show model summary
		"""
		
		summary(self.module)
		
		if (isinstance(self.module, ExtendModule)):
			for arr in self.module._layer_shapes:
				print ( arr[0] + " => " + str(tuple(arr[1])) )
	
	
	def save(self, file_name=None):
		
		"""
		Save model to file
		"""
		
		if file_name is None:
			file_name = self.get_path()
		
		if self.module:
			
			dir_name = os.path.dirname(file_name)
			if not os.path.isdir(dir_name):
				os.makedirs(dir_name)
			
			torch.save(self.module.state_dict(), file_name)
	
	
	def save_onnx(self, tensor_device=None):
		
		"""
		Save model to onnx file
		"""
		
		import torch, torch.onnx
		
		if tensor_device is None:
			tensor_device = self.get_tensor_device()
		
		onnx_model_path = self.get_path_onnx()
		
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
	
	
	def load(self, file_name=None):
		
		"""
		Load model from file
		"""
		
		if file_name is None:
			file_name = self.get_path()
		
		self._is_trained = False
		
		try:
			if os.path.isfile(file_name) and self.module:
				self.module.load_state_dict(torch.load(file_name))
				self._is_trained = True
		
		except:
			pass
		
		
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
		
		
	def on_end_epoch(self, **kwargs):
		
		"""
		On epoch end
		"""
		
		epoch_number = self.train_status.epoch_number
		acc_train = self.train_status.get_acc_train()
		acc_test = self.train_status.get_acc_test()
		acc_rel = self.train_status.get_acc_rel()
		loss_test = self.train_status.get_loss_test()
		
		if epoch_number >= self.max_epochs:
			self.stop_training()
		
		if acc_train > self.max_acc and epoch_number >= self.min_epochs:
			self.stop_training()
		
		if acc_test > self.max_acc and epoch_number >= self.min_epochs:
			self.stop_training()
		
		if acc_rel > self.max_acc_rel and acc_train > 0.8:
			self.stop_training()
		
		if loss_test < self.min_loss_test and epoch_number >= self.min_epochs:
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
			self.optimizer = torch.optim.Adam(self.module.parameters(), lr=1e-3)
		
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
		train_status.do_training = 1
		train_status.train_data_count = self.get_train_data_count()
		train_status.on_start_train()
		
		try:
		
			while True:
				
				train_status.clear_iter()
				train_status.on_start_epoch()
				
				# Train batch
				for batch_x, batch_y in self.train_loader:
					
					batch_x, batch_y = self.convert_batch(x=batch_x, y=batch_y)
					batch_x = batch_x.to(tensor_device)
					batch_y = batch_y.to(tensor_device)
					
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
					
					batch_x, batch_y = self.convert_batch(x=batch_x, y=batch_y)
					batch_x = batch_x.to(tensor_device)
					batch_y = batch_y.to(tensor_device)
					
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
		
		history_image = self.get_name() + ".png"
		
		dir_name = os.path.dirname(history_image)
		if not os.path.isdir(dir_name):
			os.makedirs(dir_name)
		
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
		
		vector_x, _ = self.convert_batch(x=vector_x)
		
		vector_x = vector_x.to(tensor_device)
		module = self.module.to(tensor_device)
		
		vector_y = module(vector_x)
		
		return vector_y
	
		
	
	def control(self,
		control_dataset=None, control_loader=None,
		batch_size=32, check_answer_batch=None,
		tensor_device=None,
		verbose=True
	):
		
		"""
		Control model
		"""
		
		if tensor_device is None:
			tensor_device = self.get_tensor_device()
			
		model = self.module.to(tensor_device)
		
		if control_dataset is None:
			control_dataset = self.control_dataset
		
		if control_loader is None and control_dataset is not None:
			control_loader = DataLoader(
				control_dataset,
				batch_size = batch_size,
				drop_last = False,
				shuffle = False
			)
		
		if check_answer_batch is None:
			check_answer_batch = self.__class__.check_answer_batch
		
		# Output answers
		correct_answers = 0
		total_questions = 0
		control_data_count = self.get_control_data_count()
		
		# Run control dataset
		for batch_x, batch_y in control_loader:
			
			batch_x, batch_y = self.convert_batch(x=batch_x, y=batch_y)
			batch_x = batch_x.to(tensor_device)
			batch_y = batch_y.to(tensor_device)
			
			# Вычислим результат модели
			batch_predict = model(batch_x)
			
			if check_answer_batch != None:
				correct = check_answer_batch(
					self,
					batch_x = batch_x,
					batch_y = batch_y,
					batch_predict = batch_predict,
					type="control"
				)
				correct_answers = correct_answers + correct
			
			total_questions = total_questions + batch_x.shape[0]
			
			"""
			correct = correct_answers / total_questions
			
			iter_value = total_questions / control_data_count
			msg = ("\rstep={iter_value}%, correct={rate}%").format(
				iter_value = round(iter_value * 100),
				correct = round(correct * 100),
			)
			
			print (msg, end='')
			"""
			
		if verbose and total_questions > 0:
			print ("Control rate: " +
				str(correct_answers) + " of " + str(total_questions) + " " +
				"(" + str(round( correct_answers / total_questions * 100)) + "%)"
			)
		
		return correct_answers, total_questions


layers_factories = {}


def register_layer_factory(key, value):
	
	"""
	Regisyer layer factory
	"""
	
	layers_factories[key] = value


class AbstractLayerFactory:
	
	
	def __init__(self):
		
		"""
		Constructor
		"""
		
		self.module = None
	
	
	def init_factory(self, *args, **kwargs):
		
		"""
		Init factory
		"""
		
		self.args = args
		self.kwargs = kwargs
	
	
	def get_name(self):
		
		"""
		Returns name
		"""
		
		name = self.args[0]
		return name
	
	
	def create_layer(self, work_tensor, module):
		
		"""
		Create new layer
		"""
		
		return None, work_tensor

	
	def forward(self, x):
		
		if self.module:
			return self.module(x)
			
		return x
	
	
class Factory_Conv3d(AbstractLayerFactory):

	def create_layer(self, work_tensor, module):
		
		"""
		Returns Conv3d
		"""
		
		in_channels = work_tensor.shape[1]
		out_channels = self.args[1]
		kwargs = self.kwargs
		
		self.module = torch.nn.Conv3d(
			in_channels=in_channels,
			out_channels=out_channels,
			**kwargs
		)
		
		work_tensor = self.module(work_tensor)
		
		return self.module, work_tensor


class Factory_Conv2d(AbstractLayerFactory):
	
	def create_layer(self, work_tensor, module):
	
		"""
		Returns Conv2d
		"""
		
		in_channels = work_tensor.shape[1]
		out_channels = self.args[1]
		kwargs = self.kwargs
		
		self.module = torch.nn.Conv2d(
			in_channels=in_channels,
			out_channels=out_channels,
			**kwargs
		)
		
		work_tensor = self.module(work_tensor)
		
		return self.module, work_tensor


class Factory_Dropout(AbstractLayerFactory):
	
	def create_layer(self, work_tensor, module):
	
		"""
		Returns Dropout
		"""
		
		p = self.args[1]
		kwargs = self.kwargs
		
		layer_out = torch.nn.Dropout(p=p, **kwargs)
		
		return layer_out, work_tensor


class Factory_MaxPool2d(AbstractLayerFactory):
	
	def create_layer(self, work_tensor, module):
	
		"""
		Returns MaxPool2d
		"""
		
		kwargs = self.kwargs
		self.module = torch.nn.MaxPool2d(**kwargs)
		
		work_tensor = self.module(work_tensor)
		
		return self.module, work_tensor


class Factory_Flat(AbstractLayerFactory):
	
	
	def forward(self, x):
		
		args = self.args
		pos = args[1] if len(args) >= 2 else 1
		
		if pos < 0:
			pos = pos - 1
		
		shape = list(x.shape)
		shape = shape[:pos]
		shape.append(-1)
		
		x = x.reshape( shape )
		
		return x
	
	
	def create_layer(self, work_tensor, module):
		
		work_tensor = self.forward(work_tensor)
		
		return None, work_tensor


class Factory_InsertFirstAxis(AbstractLayerFactory):
	
	
	def forward(self, x):
		
		x = x[:,None,:]
		
		return x
	
	
	def create_layer(self, work_tensor, module):
		
		work_tensor = self.forward(work_tensor)
		
		return None, work_tensor


class Factory_Linear(AbstractLayerFactory):
	
	def create_layer(self, work_tensor, module):
		
		in_features = work_tensor.shape[1]
		out_features = self.args[1]
		
		self.module = torch.nn.Linear(
			in_features=in_features,
			out_features=out_features
		)
		
		work_tensor = self.module(work_tensor)
		
		return self.module, work_tensor
	

class Factory_Softmax(AbstractLayerFactory):
	
	def create_layer(self, work_tensor, module):
		
		dim = self.kwargs["dim"] if "dim" in self.kwargs else -1
		self.module = torch.nn.Softmax(dim)
		
		return self.module, work_tensor


"""
Register layers factories
"""

register_layer_factory("Conv3d", Factory_Conv3d)
register_layer_factory("Conv2d", Factory_Conv2d)
register_layer_factory("Dropout", Factory_Dropout)
register_layer_factory("MaxPool2d", Factory_MaxPool2d)
register_layer_factory("Flat", Factory_Flat)
register_layer_factory("InsertFirstAxis", Factory_InsertFirstAxis)
register_layer_factory("Linear", Factory_Linear)
register_layer_factory("Softmax", Factory_Softmax)



def layer(*args, **kwargs):
	
	"""
	Define layer
	"""
	
	factory = None
	name = args[0]
	if name in layers_factories:
		factory_class = layers_factories[name]
		factory = factory_class()
		factory.init_factory(*args, **kwargs)
		
	return factory



class ExtendModule(torch.nn.Module):
	
	def __init__(self, model):
		
		"""
		Constructor
		"""
		
		super(ExtendModule, self).__init__()
		
		self._model = model
		self._layers = []
		self._layer_shapes = []
		
	
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
		
		work_tensor = torch.zeros( tuple(arr) )
		self._layer_shapes.append( ("Input", work_tensor.shape) )
		
		if debug:
			print ("Input:" + " " + str( tuple(work_tensor.shape) ))
		
		for index, obj in enumerate(self._layers):
			
			if isinstance(obj, AbstractLayerFactory):
				
				layer_factory: AbstractLayerFactory = obj
				name = layer_factory.get_name()
				layer_name = str(index) + "_" + name
				
				layer, work_tensor = layer_factory.create_layer(work_tensor, self)
				
				self._layer_shapes.append( (layer_name, work_tensor.shape) )
				
				if debug:
					print (layer_name + " => " + str(tuple(work_tensor.shape)))
				
				if layer:
					self.add_module(layer_name, layer)
