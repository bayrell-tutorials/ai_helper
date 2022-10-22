# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT
##

import io, os, random, math, sqlite3, json, torch


class ModelDatabase:
	
	
	def __init__(self, model_folder_path):
		
		self.model_folder_path = model_folder_path
	
	
	def get_model_path(self, model_name, epoch_number=None):
		
		"""
		Returns model path
		"""
		
		file_name = os.path.join(self.model_folder_path, model_name, "model.data")
		
		if epoch_number is not None:
			file_name = os.path.join(self.model_folder_path,
				model_name, "model-" + str(epoch_number) + ".data"
			)
		
		return file_name
	
	
	def save_train_status(self, model_name, train_status):
		
		"""
		Save train status
		"""
		
		pass
	
	
	
	def load_train_status(self, model_name, train_status):
		
		"""
		Load train status
		"""
		
		pass
	
	
	def save_file(self, model_name, module, epoch_number=None):
		
		"""
		Save model to file
		"""
		
		file_name = self.get_model_path(model_name, epoch_number)
		
		if module:
			make_parent_dir(file_name)
			torch.save(module.state_dict(), file_name)
	
	
	def save(self, model_name, module, train_status, epoch_number=None):
		
		"""
		Save model
		"""
		
		self.save_file(model_name, module)
		
		if epoch_number is not None:
			self.save_file(model_name, module, epoch_number)
		
		self.save_train_status(model_name, train_status, epoch_number)
		
	
	def load(self, model_name, module, train_status, epoch_number=None):
		
		"""
		Load model
		"""
		
		state_dict = None
		file_name = self.get_model_path(model_name, epoch_number)
		
		try:
			if os.path.isfile(file_name):
				state_dict = torch.load(file_name)
		
		except:
			pass
		
		if state_dict:
			self.load_train_status(model_name, train_status, epoch_number)
			module.load_state_dict(state_dict)
			return True
		
		return False
	