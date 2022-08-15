# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT
##

import os
from .DataSet import DataSet


class AbstractNetwork:
	
	
	def __init__(self):
		
		self.dataset = None
		self.model_name = os.path.join("data", "model")
		self.input_shape = None
		self.output_shape = None
		self._is_new = True
		
	
	def get_model_name(self):
		
		"""
		Returns model name
		"""
		
		return self.model_name
	
	
	def get_model_path(self):
		
		"""
		Returns model path
		"""
		
		return self.model_name
		
	
	def set_dataset(self, dataset: DataSet):
		
		"""
		Set dataset
		"""
		
		self.dataset = dataset
	
	
	def is_loaded(self):
		
		"""
		Returns true if model is loaded
		"""
		
		return False
	
	
	def is_new(self):
		
		"""
		Returns true if model is loaded
		"""
		
		return self._is_new	
	
	
	def create(self):
		
		"""
		Create model
		"""
		
		pass
	
	
	def load(self):
		
		"""
		Load model from file
		"""
		
		pass
	
	
	def show_summary(self):
		
		"""
		Show model info
		"""
		
		pass
	
	
	def train(self):
		
		"""
		Train model
		"""
		
		pass
	
	
	def show_train_info(self):
		
		"""
		Show train info
		"""
		
		pass
	
	
	def check(self, control_dataset, callback=None):
		
		"""
		Check model
		"""
		
		vector_x = control_dataset.get_x()
		vector_y = control_dataset.get_y()
		
		# Predict
		vector_answer = self.predict( vector_x )
		
		# Output answers
		correct_answers = 0
		total_questions = len(vector_x)
		
		if vector_answer is not None:
			for i in range(0, total_questions):
				
				if callback != None:
					correct = callback(
						question = vector_x[i],
						answer = vector_answer[i],
						control = vector_y[i],
					)
					if correct:
						correct_answers = correct_answers + 1
		
		return correct_answers, total_questions
	
	
	def predict(self, vector_x):
		
		"""
		Predict model
		"""
		
		return None