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


class AbstractNetwork:
	
	
	def __init__(self):
		#AbstractNetwork.__init__(self)
		
		self.input_shape = None
		self.output_shape = None
		self.train_loader = None
		self.test_loader = None
		self.control_loader = None
		self.train_dataset = None
		self.test_dataset = None
		self.control_dataset = None
		self.epochs = 0
		self.batch_size = 64
		self.model = None
		self.history = None
		self.optimizer = None
		self.loss = None
		
		self._is_trained = False
		self._do_training = True
		
		
	def get_tensor_device(self):
		r"""
		Returns tensor device name
		"""
		return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
		
		
	def get_name(self):
		r"""
		Returns model name
		"""
		return os.path.join("data", "model")
	
	
	def get_path(self):
		r"""
		Returns model path
		"""
		return self.get_name() + ".zip"
	
	
	def is_loaded(self):
		r"""
		Returns true if model is loaded
		"""
		return self.model is not None
	
	
	def is_trained(self):
		r"""
		Returns true if model is loaded
		"""
		return self.is_loaded() and self._is_trained
	
	
	def load_dataset(self, type):
		"""
		Load dataset
		"""
		pass
		
	
	def get_train_data_count(self):
		if (self.train_dataset is not None and
			isinstance(self.train_dataset, TensorDataset)):
				return self.train_dataset.tensors[0].shape[0]
		return 1
	
	
	def create_model(self):
		r"""
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
		
		r"""
		Save model to file
		"""
		
		if file_name is None:
			file_name = self.get_path()
		
		if self.model:
			
			dir_name = os.path.dirname(file_name)
			if not os.path.isdir(dir_name):
				os.makedirs(dir_name)
			
			torch.save(self.model.state_dict(), file_name)
	
	
	def load(self, file_name=None):
		
		r"""
		Load model from file
		"""
		
		if file_name is None:
			file_name = self.get_path()
		
		self._is_trained = False
		
		if os.path.isfile(file_name) and self.model:
			self.model.load_state_dict(torch.load(file_name))
			self._is_trained = True
		
		
	def train_epoch_callback(self, **kwargs):
		r"""
		Train epoch callback
		"""
		pass
		
		
	def stop_training(self):
		r"""
		Stop training
		"""
		self._do_training = False
		
	
	def train(self,
		tensor_device=None,
		verbose=True,
		train_epoch_callback=None,
		train_data_count=None
	):
		
		r"""
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
		
		if train_epoch_callback is None:
			train_epoch_callback = self.__class__.train_epoch_callback
		
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
		
		self.history = {
			"loss_train": [],
			"loss_test": [],
		}
		
		if train_data_count is None:
			train_data_count = self.get_train_data_count()
		
		# Do train
		self._do_training = True
		
		for step_index in range(self.epochs):
		  
			loss_train = 0
			loss_test = 0

			batch_iter = 0

			# Train batch
			for batch_x, batch_y in self.train_loader:

				batch_x = batch_x.to(tensor_device)
				batch_y = batch_y.to(tensor_device)

				# Predict model
				model_res = model(batch_x)

				# Get loss value
				loss_value = self.loss(model_res, batch_y)
				loss_train = loss_value.item()
				
				# Gradient
				self.optimizer.zero_grad()
				loss_value.backward()
				self.optimizer.step()

				# Clear CUDA
				if torch.cuda.is_available():
					torch.cuda.empty_cache()

				del batch_x, batch_y

				batch_iter = batch_iter + self.batch_size
				batch_iter_value = round(batch_iter / train_data_count * 100)
				
				if verbose:
					print (f"\rStep {step_index+1}, {batch_iter_value}%", end='')
			
			
			# Test batch
			for batch_x, batch_y in self.test_loader:

				batch_x = batch_x.to(tensor_device)
				batch_y = batch_y.to(tensor_device)

				# Predict model
				model_res = model(batch_x)

				# Get loss value
				loss_value = self.loss(model_res, batch_y)
				loss_test = loss_value.item()
				
				# Clear CUDA
				if torch.cuda.is_available():
					torch.cuda.empty_cache()
					
			
			# Output train step info
			if verbose:
				print ("\r", end='')
				print (f"Step {step_index+1}, loss: {loss_train},\tloss_test: {loss_test}")
			
			# Add history
			self.history["loss_train"].append(loss_train)
			self.history["loss_test"].append(loss_test)

			# Epoch callback
			if train_epoch_callback is not None:
				train_epoch_callback(
					self,
					loss_train=loss_train,
					loss_test=loss_test,
					step_index=step_index,
					train_data_count=train_data_count,
				)
			
			if not self._do_training:
				break
			
		
		self._is_trained = True
		
		
	def train_show_history(self):
		
		r"""
		Show train history
		"""
		
		history_image = self.get_name() + ".png"
		
		dir_name = os.path.dirname(history_image)
		if not os.path.isdir(dir_name):
			os.makedirs(dir_name)
		
		plt.plot( np.multiply(self.history['loss_train'], 100), label='train loss')
		plt.plot( np.multiply(self.history['loss_test'], 100), label='test loss')
		plt.ylabel('Percent')
		plt.xlabel('Epoch')
		plt.legend()
		plt.savefig(history_image)
		plt.show()
		
		
	def predict(self, vector_x, tensor_device=None):
		
		r"""
		Predict model
		"""
		
		if tensor_device is None:
			tensor_device = self.get_tensor_device()
		
		vector_x = vector_x.to(tensor_device)
		model = self.model.to(tensor_device)
		
		vector_y = model(vector_x)
		
		return vector_y
	
	
	def check_answer(self, batch_x, batch_y, batch_predict):
		"""
		Check batch anser
		"""
		
		return 0
		
	
	def control(self,
		control_dataset=None, control_loader=None,
		batch_size=32, check_answer=None,
		tensor_device=None,
		verbose=True
	):
		
		r"""
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
				batch_size=batch_size,
				drop_last=False,
				shuffle=False
			)
		
		if check_answer is None:
			check_answer = self.__class__.check_answer
		
		# Output answers
		correct_answers = 0
		total_questions = 0
		
		# Run control dataset
		for batch_x, batch_y in control_loader:

			batch_x = batch_x.to(tensor_device)
			batch_y = batch_y.to(tensor_device)
			
			# Вычислим результат модели
			batch_predict = model(batch_x)
			
			if check_answer != None:
				correct = check_answer(
					self,
					batch_x = batch_x,
					batch_y = batch_y,
					batch_predict = batch_predict
				)
				correct_answers = correct_answers + correct
			
			total_questions = total_questions + 1
		
		if verbose:
			print ("Control rate: " +
				str(correct_answers) + " of " + str(total_questions) + " " +
				"(" + str(round( correct_answers / total_questions * 100)) + "%)"
			)
		
		return correct_answers, total_questions

