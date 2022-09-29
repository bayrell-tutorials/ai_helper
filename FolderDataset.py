# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT
##

import torch, os, math, json

from torch.utils.data import Dataset


class FolderDataset(Dataset):
	
	
	def __init__(self):
		
		"""
		Init folder dataset
		"""
		
		Dataset.__init__(self)
		
		self.chunk_folder_names = (1, 2)
		self.total_data_count = 0
		self.folder_path = ""
	
	
	def __getitem__(self, index):
		
		data = self.read_data(index)
		
		return ( data[0], data[1] )
	
	
	def __len__(self):
		return self.total_data_count
	
	
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
		
		if not os.path.isdir(self.folder_path):
			os.makedirs(self.folder_path)
		
		json_file_path = os.path.join(self.folder_path, "dataset.json")
		
		obj = {
			"total_data_count": self.total_data_count,
		}
		
		json_object = json.dumps(obj, indent=4)
		with open(json_file_path, "w") as file:
			file.write(json_object)
		
	
	
	def read_json(self, folder_path):
		
		"""
		Read json file
		"""
		
		self.set_folder(folder_path)
		
		json_file_path = os.path.join(self.folder_path, "dataset.json")
		
		if os.path.isfile(json_file_path):
			with open(json_file_path) as f:
				obj = json.load(f)
				self.total_data_count = obj["total_data_count"]
	
	
	def save_data(self, *data):
		
		"""
		Save data
		"""
		
		folder_path = os.path.join(
			self.folder_path,
			self.get_folder_path_by_number(self.total_data_count)
		);
		file_path = os.path.join(folder_path, str(self.total_data_count) + ".data");
		
		if not os.path.isdir(folder_path):
			os.makedirs(folder_path)
		
		torch.save(data, file_path)
		self.total_data_count = self.total_data_count + 1
		
		
	def read_data(self, file_number):
		
		"""
		Read data by number
		"""
		
		res = None
		
		file_path = os.path.join(
			self.folder_path,
			self.get_folder_path_by_number(file_number),
			str(file_number) + ".data"
		);
		
		try:
			res = torch.load(file_path)
		except Exception:
			res = None
			
		return res


	def clear(self):
		
		"""
		Clear folder
		"""
		
		self.total_data_count = 0
		
		if self.folder_path != "":
			if os.path.isdir(self.folder_path):
				import shutil
				shutil.rmtree(self.folder_path)
