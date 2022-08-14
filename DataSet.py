# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT
##

import io, os, random, math
import numpy as np

from PIL import Image
from sklearn.model_selection import train_test_split

from ai_helper.Utils import vector_append


class DataSet:
	
	def __init__(self):
		
		self.data = []
		self.train_x = None
		self.train_y = None
		self.test_x = None
		self.test_y = None
		self._is_new = False
	
	
	def save(self, file_name):
		
		"""
			Save dataset to file
		"""
		
		x = self.get_x()
		y = self.get_y()
		
		#data = None
		#vector_append(data, x)
		#vector_append(data, y)
		
		if (os.path.isfile(file_name + ".npz")):
			os.unlink(file_name + ".npz")
		
		np.savez_compressed(file_name, x=x, y=y)
		
		del x, y
	
	
	def load(self, file_name):
		
		"""
			Load dataset from file
		"""
		
		data = np.load(file_name + ".npz")
		x = data["x"]
		y = data["y"]
		
		for i in range(0, len(x)):
			self.append(x[i], y[i])
		
		del x, y, data
	
	
	def append(self, x, y):
		
		"""
			Add data to dataset
		"""
		
		self.data.append((x, y))
		self._is_new = True
	
	
	def get_x(self):
		
		"""
			Return x from dataset
		"""
		
		return np.asarray(list(map(lambda item: item[0], self.data)))
	
	
	def get_y(self):
		
		"""
			Return y from dataset
		"""
		
		return np.asarray(list(map(lambda item: item[1], self.data)))
	
	
	def build_train(self):
		
		"""
			Build dataset for train
		"""
		
		if self._is_new and self.data != None:
			
			train, test = train_test_split(self.data)
			
			self.train_x = np.asarray(list(map(lambda item: item[0], train)))
			self.train_y = np.asarray(list(map(lambda item: item[1], train)))
			
			self.test_x = np.asarray(list(map(lambda item: item[0], test)))
			self.test_y = np.asarray(list(map(lambda item: item[1], test)))
			
			self._is_new = False
			
			del train, test
	
	
	def get_input_shape(self):
		
		"""
			Returns input shape
		"""
		
		input_shape = self.data[0][0].shape
		return input_shape
		
		
	def get_output_shape(self):
		
		"""
			Returns output shape
		"""
		
		output_shape = self.data[0][1].shape
		return output_shape
		
		
	def get_build_train_shape(self):
		
		"""
			Returns train shape
		"""
		
		input_shape = self.train_x.shape[1:]
		output_shape = self.train_y.shape[1]
		train_count = self.train_x.shape[0]
		test_count = self.test_x.shape[0]
		
		return input_shape, output_shape, train_count, test_count
		
