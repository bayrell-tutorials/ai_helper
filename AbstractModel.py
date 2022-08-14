# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT
##

import os
from .DataSet import DataSet


class AbstractModel:
	
	
	def __init__(self):
		
		self.dataset = None
		self.model_name = os.path.join("data", "model")
		self.input_shape = None
		self.output_shape = None
		
	
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
		
		pass
	
	
	def predict(self, vector_x):
		
		"""
			Predict model
		"""
		
		pass