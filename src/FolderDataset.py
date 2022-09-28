# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT
##

import torch, os, math, json


class FolderDataset():
	
	def __init__(self):
		
		"""
		Init folder dataset
		"""
		
		self.chunk_folder_names = (1, 2)
		self.total_data_count = 0
		self.folder_path = ""
	
	
	def __getindex__(self, index):
		
		x, y = self.read_tensor(index)
		
		return (x, y)
	
	
	def __len__(self):
		return len(self.total_data_count)
	
	
	def get_folder_path_by_number(self, file_number):
		
		"""
		Get forlder path by number
		"""
		
		chunk_arr = []
		last = file_number
		
		for value in self.chunk_folder_names:
			
			name = last % pow(10, value)
			name = str(name).zfill(value)
			chunk_arr.append(name)
			
			last = math.floor(last / pow(10, value))
			
		return os.path.join(*chunk_arr)
		
	
	def set_folder(self, folder_path):
		
		"""
		Set folder path
		"""
		
		self.folder_path = folder_path
		self.total_data_count = 0
	
	
	def write_json(self):
		
		"""
		Write json
		"""
		
		json_file_path = os.path.join(self.folder_path, "dataset.json")
		
		obj = {
			"total_data_count": self.total_data_count,
		}
		
		json_object = json.dumps(obj, indent=4)
		with open(json_file_path, "w") as file:
			file.write(json_object)
		
	
	
	def read_folder(self, folder_path):
		
		"""
		Set folder path
		"""
		
		self.set_folder(folder_path)
		
		json_file_path = os.path.join(self.folder_path, "dataset.json")
		
		with open(json_file_path) as f:
			obj = json.load(f)
			self.total_data_count = obj["total_data_count"]
	
	
	def add_tensor(self, x, y):
		
		"""
		Add tensor
		"""
		
		folder_path = os.path.join(
			self.folder_path,
			self.get_folder_path_by_number(self.total_data_count)
		);
		file_path = os.path.join(folder_path, str(self.total_data_count) + ".data");
		
		if not os.path.isdir(folder_path):
			os.makedirs(folder_path)
		
		torch.save([x, y], file_path)
		self.total_data_count = self.total_data_count + 1
		
		
	def read_tensor(self, file_number):
		
		"""
		Read tensor by number
		"""
		
		file_path = os.path.join(
			self.folder_path,
			self.get_folder_path_by_number(self.total_data_count),
			str(self.total_data_count) + ".data"
		);
		
		x, y = torch.load(file_path)
		return (x, y)


	def clear(self):
		
		"""
		Clear folder
		"""
		
		self.total_data_count = 0
		
		if self.folder_path != "":
			if os.path.isdir(self.folder_path):
				import shutil
				shutil.rmtree(self.folder_path)
