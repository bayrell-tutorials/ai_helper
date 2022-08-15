# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT
##

import os, torch


class AbstractNetwork:
	
	
	def __init__(self):
		#AbstractNetwork.__init__(self)
		
		self.input_shape = None
		self.output_shape = None
		self.tensor_device = None
		self.train_loader = None
		self.test_loader = None
		self.epochs = 0
		self.batch_size = 64
		self.model = None
		self.history = None
		self.optimizer = None
		self.loss = None
		
		self._is_trained = False
		
		
	def get_name(self):
		r"""
		Returns model name
		"""
		return os.path.join("data", "model")
	
	
	def get_path(self):
		r"""
		Returns model path
		"""
		return self.get_name()
	
	
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
	
	
	def create_model(self):
		r"""
		Create model
		"""
		self.model = None
		self._is_trained = False
	
	
	def save(self, file_name=None):
		
		r"""
		Save model to file
		"""
		
		if file_name is None:
			file_name = self.get_path()
			
		torch.save(self.model.state_dict(), file_name)
	
	
	def load(self, file_name=None):
		
		r"""
		Load model from file
		"""
		
		if file_name is None:
			file_name = self.get_path()
		
		self._is_trained = False
		
		if os.path.isfile(file_name):
			self.model.load_state_dict(torch.load(file_name))
			self._is_trained = True
		
	
	def train(self, verbose=True, stop_train_callback=None):
		
		r"""
		Train model
		"""
		
		model = self.model.to(self.tensor_device)
		
		self.history = {
		  "loss_train": [],
		  "loss_test": [],
		}
		
		train_count = 1
		
		# Do train
		for step_index in range(self.epochs):
		  
			loss_train = 0
			loss_test = 0

			batch_iter = 0

			# Train batch
			for batch_x, batch_y in self.train_loader:

				batch_x = batch_x.to(self.tensor_device)
				batch_y = batch_y.to(self.tensor_device)

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
				batch_iter_value = round(batch_iter / train_count * 100)
				
				if verbose:
					print (f"\rStep {step_index+1}, {batch_iter_value}%", end='')
			
			
			# Test batch
			for batch_x, batch_y in self.test_loader:

				batch_x = batch_x.to(self.tensor_device)
				batch_y = batch_y.to(self.tensor_device)

				# Predict model
				model_res = model(batch_x)

				# Get loss value
				loss_value = self.loss(model_res, batch_y)
				loss_test = loss_value.item()
			
			
			# Output train step info
			if verbose:
				print ("\r", end='')
				print (f"Step {step_index+1}, loss: {loss_train},\tloss_test: {loss_test}")
			
			
			# Is stop train ?
			is_stop = False
			if stop_train_callback is not None:
				is_stop = stop_train_callback(
					loss_train=loss_train,
					loss_test=loss_test,
					step_index=step_index,
				)
			else:
				is_stop = loss_test < 0.015 and step_index > 5
			
			# Stop train
			if is_stop:
				break
			
			
			# Add history
			self.history["loss_train"].append(loss_train)
			self.history["loss_test"].append(loss_test)
		
		self._is_trained = True
		
		
	def predict(self, vector_x):
		
		r"""
		Predict model
		"""
		
		vector_x = vector_x.to(self.tensor_device)
		model = self.model.to(self.tensor_device)
		
		vector_y = model(vector_x)
		
		return vector_y
		
	
	def control(self, control_loader, callback=None):
		
		r"""
		Control model
		"""
		
		model = self.model.to(self.tensor_device)
		
		# Output answers
		correct_answers = 0
		total_questions = 0
		
		# Run control dataset
		for batch_x, batch_y in control_loader:

			batch_x = batch_x.to(self.tensor_device)
			batch_y = batch_y.to(self.tensor_device)

			# Вычислим результат модели
			batch_answer = model(batch_x)
			
			if callback != None:
				correct = callback(
					batch_x = batch_x,
					batch_y = batch_y,
					batch_answer = batch_answer
				)
				if correct:
					correct_answers = correct_answers + 1
			
			total_questions = total_questions + 1
		
		return correct_answers, total_questions

