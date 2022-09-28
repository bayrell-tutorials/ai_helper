# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT
##

import torch, os
from .Utils import alphanum_sort


class ChunkLoader:
	
	
	def __init__(self):
		
		"""
		Init chunk loader
		"""
		
		self.chunk_path = None
		self.total_data_count = 0
		self.chunk_number = 0
		self.chunk_size = 0
		self.chunk_prefix = ""
		self.chunk_files = []
		self.data_x = torch.tensor([])
		self.data_y = torch.tensor([])
		self.type_x = None
		self.type_y = None
	
	
	def set_type(self, x=None, y=None):
		
		"""
		Set tensor type
		"""
		
		if x is not None:
			self.type_x = x
			self.data_x = self.data_x.type(x)
		
		if y is not None:
			self.type_y = y
			self.data_y = self.data_y.type(y)
	
	
	def get_chunk_name(self, file_name):
		
		"""
		Returns chunk name
		"""
		
		chunk_name = file_name
		chunk_name = chunk_name[len(self.chunk_prefix):]
		chunk_name = chunk_name[:-len(".data")]
		
		return chunk_name
		
	
	def set_chunk_size(self, size):
		
		"""
		Set chunk size
		"""
		
		self.chunk_size = size
	
	
	def set_prefix(self, prefix):
		
		"""
		Set prefix
		"""
		
		self.chunk_prefix = prefix
	
	
	def set_chunk_path(self, path):
		
		"""
		Set chunk path
		"""
		
		self.chunk_path = path
	
	
	def clear(self):
		
		"""
		Clear chunk folder
		"""
		
		self.chunk_files = []
		self.chunk_number = 0
		self.total_data_count = 0
		
		if self.chunk_path is not None:
			for file_name in os.listdir(self.chunk_path):
				
				file_path = os.path.join(self.chunk_path, file_name)
				_, file_extension = os.path.splitext(file_name)
			
				if file_extension == ".data" and \
					file_name[:len(self.chunk_prefix)] == self.chunk_prefix:
					
					if os.path.isfile(file_path) or os.path.islink(file_path):
						os.unlink(file_path)
	
	
	def add(self, x, y):
		
		"""
		Add tensor
		"""
		
		y = y[None, :]
		x = x[None, :]
		self.data_x = torch.cat( (self.data_x, x) )
		self.data_y = torch.cat( (self.data_y, y) )
		
		if self.data_x.shape[0] >= self.chunk_size and self.chunk_size > 0:
			self.flush()
			
		self.total_data_count = self.total_data_count + 1
		
		
	def flush(self):
		
		"""
		Flush chunk to disk
		"""
		
		if not os.path.isdir(self.chunk_path):
			os.makedirs(self.chunk_path)
		
		if self.data_x.shape[0] > 0:
			
			file_name = os.path.join(self.chunk_path,
				str(self.chunk_prefix) + str(self.chunk_number) + ".data"
			)
			
			torch.save([self.data_x, self.data_y], file_name)
			
			self.data_x = torch.tensor([])
			self.data_y = torch.tensor([])
			self.chunk_number = self.chunk_number + 1
			self.set_type(x=self.type_x, y=self.type_y)
	
	
	def load_chunk(self, chunk_number):
		
		"""
		Load chunk by number
		"""
		
		file_name = os.path.join(self.chunk_path, 
			str(self.chunk_prefix) + str(chunk_number) + ".data"
		)
		
		return torch.load(file_name)
		
	
	
	def load_all_chunks(self):
		
		"""
		Load chunk loader
		"""
		
		files = os.listdir(self.chunk_path)
		alphanum_sort(files)
		
		self.chunk_files = []
		self.chunk_number = 0
		self.total_data_count = 0
		
		if self.chunk_path is not None:
			for file_name in files:
				
				file_path = os.path.join(self.chunk_path, file_name)
				_, file_extension = os.path.splitext(file_name)
			
				if file_extension == ".data" and \
					file_name[:len(self.chunk_prefix)] == self.chunk_prefix:
					
					chunk_name = self.get_chunk_name(file_name)
					
					self.chunk_files.append({
						"file_name": file_name,
						"chunk_name": chunk_name,
					})
					
					self.chunk_number = self.chunk_number + 1
		
		
		if self.chunk_number > 1:
			first_chunk = self.load_chunk(0)
			self.chunk_size = first_chunk[0].shape[0]
			
		last_chunk = self.load_chunk(self.chunk_number - 1)
		self.total_data_count = (self.chunk_number - 1) * self.chunk_size + last_chunk[0].shape[0]
