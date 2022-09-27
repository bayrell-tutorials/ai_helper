# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT
##

import torch, os


class ChunkLoader:
	
	
	def __init__(self):
		"""
		Init chunk loader
		"""
		
		self.chunk_path = None
		self.total_data_count = 0
		self.chunk_number = 0
		self.chunk_size = 256
		self.chunk_prefix = ""
		self.data_x = torch.tensor([])
		self.data_y = torch.tensor([])
	
	
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
		
		file_name = os.path.join(self.chunk_path, 
			str(self.chunk_prefix) + str(self.chunk_number) + ".data"
		)
		
		torch.save([self.data_x, self.data_y], file_name)
		
		self.data_x = torch.tensor([])
		self.data_y = torch.tensor([])
		self.chunk_number = self.chunk_number + 1