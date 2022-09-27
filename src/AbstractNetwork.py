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

from ai_helper import *


class AbstractNetwork:
	
	
	def __init__(self):
		#AbstractNetwork.__init__(self)
		
		from .TrainStatus import TrainStatus
		self.train_status = TrainStatus()
		self.train_status.net = self
		self.train_loader = None
		self.test_loader = None
		self.control_loader = None
		self.train_dataset = None
		self.test_dataset = None
		self.control_dataset = None
		self.batch_size = 64
		self.test_size = 0.1
		self.model = None
		self.optimizer = None
		self.loss = None
		self.verbose = True
		self.max_epochs = 50
		self.min_epochs = 10
		self.max_acc = 0.95
		self.max_acc_rel = 1.5
		self.min_loss_test = 0.001
		self.input_shape = (1)
		self.output_shape = (1)
		self.onnx_opset_version = 9
		
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
		return os.path.join("data", "model")
	
	
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
		return self.model is not None
	
	
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
			
			self.train_dataset, self.test_dataset = self.get_train_dataset(
				test_size=self.test_size
			)
		
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
			isinstance(self.train_dataset, TensorDataset)):
				return self.train_dataset.tensors[0].shape[0]
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
	
	
	def create_model(self):
		"""
		Create model
		"""
		self.model = None
		self._is_trained = False
		
	
	def summary(self):
		"""
		Show model summary
		"""
		summary(self.model)
	
	
	def save(self, file_name=None):
		
		"""
		Save model to file
		"""
		
		if file_name is None:
			file_name = self.get_path()
		
		if self.model:
			
			dir_name = os.path.dirname(file_name)
			if not os.path.isdir(dir_name):
				os.makedirs(dir_name)
			
			torch.save(self.model.state_dict(), file_name)
	
	
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
		model = self.model.to(tensor_device)
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
			if os.path.isfile(file_name) and self.model:
				self.model.load_state_dict(torch.load(file_name))
				self._is_trained = True
		
		except:
			pass
		
		
	def check_answer(self, **kwargs):
		"""
		Check answer
		"""
		
		type = kwargs["type"]
		tensor_x = kwargs["tensor_x"]
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
			
			tensor_x = batch_x[i] * 256
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
		acc_train = self.train_status.acc_train
		acc_test = self.train_status.acc_test
		
		epoch_number = self.train_status.epoch_number
		acc_train = self.train_status.get_acc_train()
		acc_test = self.train_status.get_acc_test()
		acc_rel = self.train_status.get_acc_rel()
		loss_test = self.train_status.loss_test
		
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
		
		pass
		
		
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
			self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
		
		# Mean squared error
		if self.loss is None:
			self.loss = torch.nn.MSELoss()
		
		if tensor_device is None:
			tensor_device = self.get_tensor_device()
		
		if self.train_loader is None and self.train_dataset is not None:
			self.train_loader = DataLoader(
				self.train_dataset,
				batch_size=self.batch_size,
				drop_last=False,
				shuffle=True
			)
		
		if self.test_loader is None and self.test_dataset is not None:
			self.test_loader = DataLoader(
				self.test_dataset,
				batch_size=self.batch_size,
				drop_last=False,
				shuffle=False
			)
		
		model = self.model.to(tensor_device)
		
		# Do train
		train_status = self.train_status
		train_status.epoch_number = 1
		train_status.do_training = 1
		train_status.train_data_count = self.get_train_data_count()
		train_status.on_start_train()
		
		try:
		
			while True:
				
				train_status.on_start_epoch()
				
				# Train batch
				for batch_x, batch_y in self.train_loader:
					
					batch_x = batch_x.to(tensor_device)
					batch_y = batch_y.to(tensor_device)
					
					train_status.on_start_batch_train(batch_x, batch_y)
					
					# Predict model
					batch_predict = model(batch_x)

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
						train_kind = "train",
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
					
					train_status.on_start_batch_test(batch_x, batch_y)
					
					# Predict model
					batch_predict = model(batch_x)
					
					# Get loss value
					loss_value = self.loss(batch_predict, batch_y)
					train_status.loss_test = train_status.loss_test + loss_value.item()
					
					# Calc accuracy
					accuracy = self.check_answer_batch(
						train_status = train_status,
						batch_x = batch_x,
						batch_y = batch_y,
						batch_predict = batch_predict,
						train_kind = "test",
						type = "train"
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
		
		vector_x = vector_x.to(tensor_device)
		model = self.model.to(tensor_device)
		
		vector_y = model(vector_x)
		
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
			
		model = self.model.to(tensor_device)
		
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

