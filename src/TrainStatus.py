# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT
##

from .AbstractModel import AbstractModel


class TrainStatus:
	
	net: AbstractModel = None
	batch_train_iter = 0
	batch_test_iter = 0
	train_count = 0
	test_count = 0
	loss_train = 0
	loss_test = 0
	acc_train = 0
	acc_test = 0
	epoch_number = 0
	do_training = True
	train_data_count = 0
	history = {}
	callbacks = []
	
	def __init__(self):
		from .TrainVerboseCallback import TrainVerboseCallback
		self.callbacks = [ TrainVerboseCallback() ]
		self.history = {
			"loss_train": [],
			"loss_test": [],
			"acc_train": [],
			"acc_test": [],
			"acc_rel": [],
		}
	
	def clear_iter(self):
		self.batch_train_iter = 0
		self.batch_test_iter = 0
		self.train_count = 0
		self.test_count = 0
		self.loss_train = 0
		self.loss_test = 0
		self.acc_train = 0
		self.acc_test = 0
	
	def get_iter_value(self):
		return self.train_count / self.train_data_count
	
	def get_loss_train(self):
		return self.loss_train / self.batch_train_iter
	
	def get_loss_test(self):
		return self.loss_test / self.batch_test_iter
	
	def get_acc_train(self):
		if self.train_count == 0:
			return 0
		return self.acc_train / self.train_count
	
	def get_acc_test(self):
		if self.test_count == 0:
			return 0
		return self.acc_test / self.test_count
	
	def get_acc_rel(self):
		acc_train = self.get_acc_train()
		acc_test = self.get_acc_test()
		if acc_test == 0:
			return 0
		return acc_train / acc_test
	
	def get_loss_rel(self):
		return self.loss_train / self.loss_test
	
	def stop_train(self):
		self.do_training = False
	
	
	"""	====================== Events ====================== """
	
	def on_start_train(self):
		"""
		Start train event
		"""
		for callback in self.callbacks:
			if hasattr(callback, "on_start_train"):
				callback.on_start_train(self)
		
	
	def on_end_train(self):
		"""
		End train event
		"""
		for callback in self.callbacks:
			if hasattr(callback, "on_end_train"):
				callback.on_end_train(self)
	
	
	def on_start_epoch(self):
		"""
		Start epoch event
		"""
		self.clear_iter()
		for callback in self.callbacks:
			if hasattr(callback, "on_start_epoch"):
				callback.on_start_epoch(self)
	
	
	def on_end_epoch(self):
		"""
		End epoch event
		"""
		loss_train = self.get_loss_train()
		loss_test = self.get_loss_test()
		acc_train = self.get_acc_train()
		acc_test = self.get_acc_test()
		acc_rel = self.get_acc_rel()
		self.history["loss_train"].append(loss_train)
		self.history["loss_test"].append(loss_test)
		self.history["acc_train"].append(acc_train)
		self.history["acc_test"].append(acc_test)
		self.history["acc_rel"].append(acc_rel)
		self.net.on_end_epoch()
		
		for callback in self.callbacks:
			if hasattr(callback, "on_end_epoch"):
				callback.on_end_epoch(self)
	
	
	def on_start_batch_train(self, batch_x, batch_y):
		"""
		Start train batch event
		"""
		for callback in self.callbacks:
			if hasattr(callback, "on_start_batch_train"):
				callback.on_start_batch_train(self)
	
	
	def on_end_batch_train(self, batch_x, batch_y):
		"""
		End train batch event
		"""
		for callback in self.callbacks:
			if hasattr(callback, "on_end_batch_train"):
				callback.on_end_batch_train(self)
	
	
	def on_start_batch_test(self, batch_x, batch_y):
		"""
		Start test batch event
		"""
		for callback in self.callbacks:
			if hasattr(callback, "on_start_batch_test"):
				callback.on_start_batch_test(self)
	
	
	def on_end_batch_test(self, batch_x, batch_y):
		"""
		End test batch event
		"""
		for callback in self.callbacks:
			if hasattr(callback, "on_end_batch_test"):
				callback.on_end_batch_test(self)
	