# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT
##

import torch, os, math, json
from .Utils import alphanum_sort


class ChunkLoader:
	
	
	def __init__(self):
		
		"""
		Init chunk loader
		"""
		
		self.chunk_folder_names = [1]
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
		
		folder_path_abs = os.path.join(self.chunk_path, self.chunk_prefix)
		if self.chunk_path != "" and self.chunk_prefix != "" and folder_path_abs != "":
			if os.path.isdir(folder_path_abs):
				import shutil
				shutil.rmtree(folder_path_abs)
			
		
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
		
		
	def get_chunk_number_folder(self, chunk_number):
	
		chunk_arr = []
		last = chunk_number
		
		for value in self.chunk_folder_names:
			
			name = last % pow(10, value)
			name = str(name).zfill(value)
			chunk_arr.append(name)
			
			last = math.floor(last / pow(10, value))
			
		return chunk_arr
		
		
	def flush(self):
		
		"""
		Flush chunk to disk
		"""
		
		if self.data_x.shape[0] > 0:
			
			chunk_number_folder = self.get_chunk_number_folder(self.chunk_number);
			chunk_number_folder = os.path.join(*chunk_number_folder)
			
			folder_path = os.path.join(self.chunk_prefix, chunk_number_folder)
			folder_path_abs = os.path.join(self.chunk_path,
				self.chunk_prefix, chunk_number_folder)
			
			if not os.path.isdir(folder_path_abs):
				os.makedirs(folder_path_abs)
				
			file_path = os.path.join(folder_path, str(self.chunk_number) + ".data")
			file_path_abs = os.path.join(self.chunk_path, file_path)
			
			torch.save([self.data_x, self.data_y], file_path_abs)
			
			self.chunk_files.append({
				"file_name":  str(self.chunk_number) + ".data",
				"file_path": file_path,
				"chunk_number": self.chunk_number,
			})
			
			self.save_json()
			
			self.data_x = torch.tensor([])
			self.data_y = torch.tensor([])
			self.chunk_number = self.chunk_number + 1
			self.set_type(x=self.type_x, y=self.type_y)
	
	
	def save_json(self):
		
		"""
		Save json dataset info to file
		"""
		
		json_file_path = os.path.join(self.chunk_path,
			self.chunk_prefix, "dataset.json")
		
		obj = {
			"chunk_files": self.chunk_files,
			"chunk_number": self.chunk_number,
			"chunk_size": self.chunk_size,
			"total_data_count": self.total_data_count,
		}
		
		json_object = json.dumps(obj, indent=4)
		with open(json_file_path, "w") as file:
			file.write(json_object)
	
	
	def load_chunk(self, chunk_number):
		
		"""
		Load chunk by number
		"""
		
		chunk_number_folder = self.get_chunk_number_folder(self.chunk_number);
		chunk_number_folder = os.path.join(*chunk_number_folder)
		
		file_name = os.path.join(
			self.chunk_path, self.chunk_prefix, chunk_number_folder,
			str(chunk_number) + ".data"
		)
		
		return torch.load(file_name)
		
	
	def read_json(self):
		
		"""
		Load chunk loader
		"""
		
		json_file_path = os.path.join(self.chunk_path,
			self.chunk_prefix, "dataset.json")
		
		with open(json_file_path) as f:
			obj = json.load(f)
			
			self.chunk_files = obj["chunk_files"]
			self.chunk_number = obj["chunk_number"]
			self.chunk_size = obj["chunk_size"]
			self.total_data_count = obj["total_data_count"]
		
	
	def get_dataset(self):
		
		"""
		Returns chunk loader dataset
		"""
		
		dataset = ChunkLoaderDataset(self)
		return dataset
	
	
	
class ChunkLoaderDataset():
	
	def __init__(self, chunk_loader):
		self.chunk_loader = chunk_loader

	def __getindex__(self, index):
		
		x = torch.tensor()
		y = torch.tensor()
		
		return (x, y)

	def __len__(self):
		return len(self.chunk_loader.total_data_count)
