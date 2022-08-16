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
		
		self.train_loader = None
		self.test_loader = None
		self.control_loader = None
		self.train_dataset = None
		self.test_dataset = None
		self.control_dataset = None
		self.batch_size = 64
		self.test_size = 0.1
		self.model = None
		self.history = None
		self.optimizer = None
		self.loss = None
		
		self._is_debug = False
		self._is_trained = False
		self._do_training = True
		
		
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
	
	
	def load_dataset(self, type):
		
		"""
		Load dataset
		"""
		
		if type == "train":
			
			self.train_dataset, self.test_dataset = self.get_train_dataset(
				test_size=self.test_size
			)
		
		if type == "control":
			
			self.control_dataset = self.get_control_dataset()
	
	
	def get_train_dataset(cls):
		
		"""
		Returns normalized train and test datasets
		"""
		
		train_dataset = TensorDataset( torch.tensor(), torch.tensor() )
		test_dataset = TensorDataset( torch.tensor(), torch.tensor() )
		
		return train_dataset, test_dataset
	
	
	def get_control_dataset(cls):
		
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
	
	
	def load(self, file_name=None):
		
		"""
		Load model from file
		"""
		
		if file_name is None:
			file_name = self.get_path()
		
		self._is_trained = False
		
		if os.path.isfile(file_name) and self.model:
			self.model.load_state_dict(torch.load(file_name))
			self._is_trained = True
		
		
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
		
		
	def train_epoch_callback(self, **kwargs):
		"""
		Train epoch callback
		"""
		
		epoch_number = kwargs["epoch_number"]
		accuracy_train = kwargs["accuracy_train"]
		accuracy_test = kwargs["accuracy_test"]
		
		if epoch_number >= 20:
			self.stop_training()
			
		if accuracy_train > 0.95:
			self.stop_training()
		
		if accuracy_test > 0.95:
			self.stop_training()
		
		pass
		
		
	def stop_training(self):
		"""
		Stop training
		"""
		self._do_training = False
		
	
	def train(self,
		tensor_device=None,
		verbose=True,
		train_epoch_callback=None,
		train_data_count=None,
		check_answer_batch=None
	):
		
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
		
		if check_answer_batch is None:
			check_answer_batch = self.__class__.check_answer_batch
		
		self.history = {
			"loss_train": [],
			"loss_test": [],
			"acc_train": [],
			"acc_test": [],
		}
		
		if train_data_count is None:
			train_data_count = self.get_train_data_count()
		
		# Do train
		self._do_training = True
		
		epoch_number = 1
		
		try:
		
			while True:
				
				batch_train_iter = 0
				batch_test_iter = 0
				train_count = 0
				test_count = 0
				loss_train = 0
				loss_test = 0
				accuracy_train = 0
				accuracy_test = 0

				# Train batch
				for batch_x, batch_y in self.train_loader:
					
					train_count = train_count + batch_x.shape[0]
					batch_x = batch_x.to(tensor_device)
					batch_y = batch_y.to(tensor_device)

					# Predict model
					batch_predict = model(batch_x)

					# Get loss value
					loss_value = self.loss(batch_predict, batch_y)
					loss_train = loss_train + loss_value.item()
					
					# Gradient
					self.optimizer.zero_grad()
					loss_value.backward()
					self.optimizer.step()
					
					# Calc accuracy
					accuracy = self.check_answer_batch(
						epoch_number=epoch_number,
						batch_iter=batch_train_iter,
						train_count=train_count,
						batch_x=batch_x,
						batch_y=batch_y,
						batch_predict=batch_predict,
						train_kind="train",
						type="train"
					)
					accuracy_train = accuracy_train + accuracy
					batch_train_iter = batch_train_iter + 1
					
					if verbose:
						
						accuracy_train_value = accuracy_train / train_count
						loss_train_value = loss_train / batch_train_iter
						
						msg = ("\rStep {epoch_number}, {iter_value}%" +
							", acc: .{acc}, loss: .{loss}"
						).format(
							epoch_number=epoch_number,
							iter_value=round(train_count / train_data_count * 100),
							loss=str(round(loss_train_value * 10000)).zfill(4),
							acc=str(round(accuracy_train_value * 100)).zfill(2),
						)
						
						print (msg, end='')
					
					del batch_x, batch_y
					
					# Clear CUDA
					if torch.cuda.is_available():
						torch.cuda.empty_cache()
				
					
				# Test batch
				for batch_x, batch_y in self.test_loader:
					
					test_count = test_count + batch_x.shape[0]
					batch_x = batch_x.to(tensor_device)
					batch_y = batch_y.to(tensor_device)

					# Predict model
					batch_predict = model(batch_x)
					
					# Get loss value
					loss_value = self.loss(batch_predict, batch_y)
					loss_test = loss_test + loss_value.item()
					batch_test_iter = batch_test_iter + 1
					
					# Calc accuracy
					accuracy = self.check_answer_batch(
						epoch_number=epoch_number,
						train_count=test_count,
						batch_iter=batch_test_iter,
						batch_x=batch_x,
						batch_y=batch_y,
						batch_predict=batch_predict,
						train_kind="test",
						type="train"
					)
					accuracy_test = accuracy_test + accuracy
					
					# Clear CUDA
					if torch.cuda.is_available():
						torch.cuda.empty_cache()
						
				# Add history
				loss_train = loss_train / batch_train_iter
				loss_test = loss_test / batch_test_iter
				accuracy_train = accuracy_train / train_count
				accuracy_test =  accuracy_test / test_count
				self.history["loss_train"].append(loss_train)
				self.history["loss_test"].append(loss_test)
				self.history["acc_train"].append(accuracy_train)
				self.history["acc_test"].append(accuracy_test)
				
				# Output train step info
				if verbose:
					print ("\r", end='')
					
					msg = ("Step {epoch_number}, " +
						"acc: .{accuracy_train}, " +
						"acc_test: .{accuracy_test}, " +
						"loss: .{loss_train}, " +
						"loss_test: .{loss_test}"
					).format(
						epoch_number = epoch_number,
						loss_train = str(round(loss_train * 10000)).zfill(4),
						loss_test = str(round(loss_test * 10000)).zfill(4),
						accuracy_train = str(round(accuracy_train * 100)).zfill(2),
						accuracy_test = str(round(accuracy_test * 100)).zfill(2),
					)
					
					print (msg)

				# Epoch callback
				if train_epoch_callback is not None:
					train_epoch_callback(
						self,
						loss_train=loss_train,
						loss_test=loss_test,
						accuracy_train=accuracy_train,
						accuracy_test=accuracy_test,
						epoch_number=epoch_number,
						train_data_count=train_data_count,
					)
				
				if not self._do_training:
					break
				
				epoch_number = epoch_number + 1
				
			self._is_trained = True
			
		except KeyboardInterrupt:
			
			print ("")
			print ("Stopped manually")
			print ("")
			
			pass

		
		
	def train_show_history(self):
		
		"""
		Show train history
		"""
		
		history_image = self.get_name() + ".png"
		
		dir_name = os.path.dirname(history_image)
		if not os.path.isdir(dir_name):
			os.makedirs(dir_name)
		
		fig, axs = plt.subplots(2)
		axs[0].plot( np.multiply(self.history['loss_train'], 100), label='train loss')
		axs[0].plot( np.multiply(self.history['loss_test'], 100), label='test loss')
		axs[0].legend()
		axs[1].plot( np.multiply(self.history['acc_train'], 100), label='train acc')
		axs[1].plot( np.multiply(self.history['acc_test'], 100), label='test acc')
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
				batch_size=batch_size,
				drop_last=False,
				shuffle=False
			)
		
		if check_answer_batch is None:
			check_answer_batch = self.__class__.check_answer_batch
		
		# Output answers
		correct_answers = 0
		total_questions = 0
		
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
		
		if verbose:
			print ("Control rate: " +
				str(correct_answers) + " of " + str(total_questions) + " " +
				"(" + str(round( correct_answers / total_questions * 100)) + "%)"
			)
		
		return correct_answers, total_questions

