# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT
##

from torch.nn import Module
from typing import Optional


class DirectModule(Module):
	
	def __init__(self):
		super().__init__()
		self._links_none = []
		self._links_next = {}
		self._links_prev = {}
		self._last_module_name = ""
		self._output_module_name = ""
		self._module_inc = 1


	def set_output_module(self, module_name):
		
		if module_name == "" or not (module_name in self._modules):
			raise KeyError("module '{}' not found".format(module_name))
		
		self._output_module_name = module_name
		
		
	def forward(self, x):
		
		values = {}
		current_list = self.links_none[:]
		current_next_list = []
		current_prev_list = []
		
		def get_prev_value(module_name, values):
			
			"""
			Returns prev value for module_name
			"""
			
			prev_value = None
			if module_name in self.links_prev:
				for prev_name in self.links_prev[module_name]:
					if prev_name in values:
						if prev_value is None:
							prev_value = values[prev_name]
						else:
							prev_value = prev_value + values[prev_name]
					
					else:
						raise KeyError("value not found for module {}".format(module_name))
							
			return prev_value
		
		
		def get_next_value(module_name, values):
			
			"""
			Predict next value for module_name
			"""
			
			prev_value = get_prev_value(module_name, values)
			current_module = self[module_name]
			next_value = current_module(prev_value)
			
			return next_value
		
		
		def clear_values(current_prev_list, values):
			
			"""
			Clear unnecessary values
			"""
			
			for module_name in current_prev_list:
				if module_name in values:
					del values[module_name]
		
		
		# Calculate value by directed graph
		while len(current_list) != 0:
			
			current_next_list = []
			
			for current_name in current_list:
				
				values[current_name] = get_next_value(current_name, values)
				
				if current_name in self.links_next:
					current_next_list = current_next_list + self.links_next[current_name]
			
			clear_values(current_prev_list, values)
			
			current_prev_list = current_list[:]
			current_list = current_next_list[:]
		
		
		y = None
		
		if self._output_module_name != "" and self._output_module_name in values:
			y = values[self._output_module_name]
		else:
			raise KeyError("output value is not setup")
		
		return y
		
	
	def add(self, module: Optional['Module'], prev_name = None):
		
		if prev_name is None and self._last_module_name != "":
			prev_name = self[self._last_module_name]
		
		new_name = "module_" + str(self._module_inc)
		
		self.add_module(new_name, module, prev_name)
		
		self._module_inc = self._module_inc + 1
		self._last_module_name = new_name
		self._output_module_name = new_name
		
		return module, new_name
	

	def add_module(self, name: str, module: Optional['Module'], prev_name = None) -> Module:
		
		if prev_name is not None:
			if name == prev_name:
				raise KeyError("prev module '{}' is same name".format(prev_name))
		
		Module.add_module(self, name, module)
		
		
		def is_exists(module_name):
			
			"""
			Check if module is exists
			"""
			
			if not(module_name in self._modules):
				raise KeyError("module '{}' not found".format(name))
		
		
		def add_module_link(src, dest):
			
			"""
			Add module link src -> dest
			"""
			
			if not (src in self.links_next):
				self._links_next[src] = []
			self._links_next[src].append(dest)
			
			if not (dest in self.links_next):
				self._links_prev[dest] = []
			self._links_prev[dest].append(src)
		
		
		if prev_name is None:
			self._links_none.append(name)
		
		if isinstance(prev_name, str):
			is_exists(prev_name)
			add_module_link(prev_name, name)
			
		if isinstance(prev_name, list) or isinstance(prev_name, tuple):
			for module_name in prev_name:
				is_exists(module_name)
				add_module_link(module_name, name)
		
		return module
		