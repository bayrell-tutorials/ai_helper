# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT
##

import os, time, torch
from torch.utils.data import DataLoader
from .utils import get_tensor_device


class TrainStatus:
	
	def __init__(self):
		self.model = None
		self.batch_train_iter = 0
		self.batch_test_iter = 0
		self.train_count_iter = 0
		self.test_count_iter = 0
		self.loss_train_iter = 0
		self.loss_test_iter = 0
		self.acc_train_iter = 0
		self.acc_test_iter = 0
		self.epoch_number = 0
		self.do_training = True
		self.train_data_count = 0
		self.time_start = 0
		self.time_end = 0
		self.callbacks = []
	
	def set_model(self, model):
		
		self.clear()
		self.model = model
		self.epoch_number = model.history["epoch_number"]
		
		if self.epoch_number > 0:
			
			self.batch_train_iter = model.history["batch_train_iter"][-1]
			self.batch_test_iter = model.history["batch_test_iter"][-1]
			self.train_count_iter = model.history["train_count_iter"][-1]
			self.test_count_iter = model.history["test_count_iter"][-1]
			self.loss_train_iter = model.history["loss_train_iter"][-1]
			self.loss_test_iter = model.history["loss_test_iter"][-1]
			self.acc_train_iter = model.history["acc_train_iter"][-1]
			self.acc_test_iter = model.history["acc_test_iter"][-1]
	
	def clear(self):
		self.clear_iter()
	
	def clear_iter(self):
		self.batch_train_iter = 0
		self.batch_test_iter = 0
		self.train_count_iter = 0
		self.test_count_iter = 0
		self.loss_train_iter = 0
		self.loss_test_iter = 0
		self.acc_train_iter = 0
		self.acc_test_iter = 0
	
	def get_iter_value(self):
		if self.train_data_count == 0:
			return 0
		return self.train_count_iter / self.train_data_count
	
	def get_loss_train(self):
		if self.batch_train_iter == 0:
			return 0
		return self.loss_train_iter / self.batch_train_iter
	
	def get_loss_test(self):
		if self.batch_test_iter == 0:
			return 0
		return self.loss_test_iter / self.batch_test_iter
	
	def get_acc_train(self):
		if self.train_count_iter == 0:
			return 0
		return self.acc_train_iter / self.train_count_iter
	
	def get_acc_test(self):
		if self.test_count_iter == 0:
			return 0
		return self.acc_test_iter / self.test_count_iter
	
	def get_acc_rel(self):
		acc_train = self.get_acc_train()
		acc_test = self.get_acc_test()
		if acc_test == 0:
			return 0
		return acc_train / acc_test
	
	def get_loss_rel(self):
		if self.loss_test == 0:
			return 0
		return self.loss_train / self.loss_test
	
	def stop_train(self):
		self.do_training = False
	
	def get_time(self):
		return self.time_end - self.time_start
	
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
		for callback in self.callbacks:
			if hasattr(callback, "on_start_epoch"):
				callback.on_start_epoch(self)
	
	
	def on_end_epoch(self):
		
		"""
		End epoch event
		"""
		
		self.model.add_train_status_iter(self)
		
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
	

class TrainVerboseCallback:
	
	
	def on_end_batch_train(self, train_status:TrainStatus):
		
		acc_train = train_status.get_acc_train()
		loss_train = train_status.get_loss_train()
		time = train_status.get_time()
		
		msg = ("\rStep {epoch_number}, {iter_value}%" +
			", acc: .{acc}, loss: .{loss}, time: {time}s"
		).format(
			epoch_number = train_status.epoch_number,
			iter_value = round(train_status.get_iter_value() * 100),
			loss = str(round(loss_train * 10000)).zfill(4),
			acc = str(round(acc_train * 100)).zfill(2),
			time = str(round(time)),
		)
		
		print (msg, end='')
	
	
	def on_end_epoch(self, train_status:TrainStatus):
		
		"""
		Epoch
		"""
		
		loss_train = train_status.get_loss_train()
		loss_test = train_status.get_loss_test()
		acc_train = train_status.get_acc_train()
		acc_test = train_status.get_acc_test()
		acc_rel = train_status.get_acc_rel()
		time = train_status.get_time()
		
		print ("\r", end='')
		
		msg = ("Step {epoch_number}, " +
			"acc: .{acc_train}, " +
			"acc_test: .{acc_test}, " +
			"acc_rel: {acc_rel}, " +
			"loss: .{loss_train}, " +
			"loss_test: .{loss_test}, " +
			"time: {time}s, "
		).format(
			epoch_number = train_status.epoch_number,
			loss_train = str(round(loss_train * 10000)).zfill(4),
			loss_test = str(round(loss_test * 10000)).zfill(4),
			acc_train = str(round(acc_train * 100)).zfill(2),
			acc_test = str(round(acc_test * 100)).zfill(2),
			acc_rel = str(round(acc_rel * 100)).zfill(2),
			time = str(round(time)),
		)
		
		print (msg)
	

class Trainer:
	
	def __init__(self, model, *args, **kwargs):
		
		self.model = model
		
		self.train_status = TrainStatus()
		self.train_status.set_model(model)
		
		self.train_loader = None
		self.test_loader = None
		self.train_dataset = None
		self.test_dataset = None
		self.batch_size = 64
		
		self.optimizer = None
		self.loss = None
		self.verbose = True
		self.num_workers = os.cpu_count()
		
		self.max_epochs = kwargs["max_epochs"] if "max_epochs" in kwargs else 50
		self.min_epochs = kwargs["min_epochs"] if "min_epochs" in kwargs else 3
		self.max_acc = kwargs["max_acc"] if "max_acc" in kwargs else 0.95
		self.max_acc_rel = kwargs["max_acc_rel"] if "max_acc_rel" in kwargs else 1.5
		self.min_loss_test = kwargs["min_loss_test"] if "min_loss_test" in kwargs else 0.001
		self.batch_size = kwargs["batch_size"] if "batch_size" in kwargs else 64
		
		self.train_dataset = kwargs["train_dataset"] if "train_dataset" in kwargs else False
		self.test_dataset = kwargs["test_dataset"] if "test_dataset" in kwargs else False
		self.tensor_device = kwargs["tensor_device"] if "tensor_device" in kwargs else None
		
		self._check_is_trained = kwargs["check_is_trained"] \
			if "check_is_trained" in kwargs else None
		
		if "callbacks" in kwargs:
			self.train_status.callbacks = kwargs["callbacks"]
		else:
			self.train_status.callbacks = [ TrainVerboseCallback() ]
		
	
	def check_is_trained(self):
		
		"""
		Returns True if model is trained
		"""
		
		if self._check_is_trained is not None:
			return self._check_is_trained(self.train_status)
		
		epoch_number = self.train_status.epoch_number
		acc_train = self.train_status.get_acc_train()
		acc_test = self.train_status.get_acc_test()
		acc_rel = self.train_status.get_acc_rel()
		loss_test = self.train_status.get_loss_test()
		
		if epoch_number >= self.max_epochs:
			return True
		
		if acc_test > self.max_acc  and epoch_number >= self.min_epochs:
			return True
		
		if acc_rel > self.max_acc_rel and acc_train > 0.8:
			return True
		
		if loss_test < self.min_loss_test and epoch_number >= self.min_epochs:
			return True
		
		return False
	
		
	def on_end_epoch(self, **kwargs):
		
		"""
		On epoch end
		"""
		
		self._is_trained = self.check_is_trained()
		
		if self._is_trained:
			self.stop_training()
		
		self.model.save()
		
		
	def stop_training(self):
		
		"""
		Stop training
		"""
		
		self.train_status.stop_train()
		
	
	def train(self):
		
		"""
		Train model
		"""
		
		model = self.model
		
		# Adam optimizer
		if self.optimizer is None:
			self.optimizer = torch.optim.Adam(model.module.parameters(), lr=3e-4)
		
		# Mean squared error
		if self.loss is None:
			self.loss = torch.nn.MSELoss()
		
		if self.tensor_device is None:
			tensor_device = get_tensor_device()
		
		if self.train_loader is None and self.train_dataset is not None:
			self.train_loader = DataLoader(
				self.train_dataset,
				num_workers=self.num_workers,
				batch_size=self.batch_size,
				drop_last=False,
				shuffle=True
			)
		
		if self.test_loader is None and self.test_dataset is not None:
			self.test_loader = DataLoader(
				self.test_dataset,
				num_workers=self.num_workers,
				batch_size=self.batch_size,
				drop_last=False,
				shuffle=False
			)
		
		module = model.module.to(tensor_device)
		
		# Do train
		train_status = self.train_status
		train_status.do_training = True
		train_status.train_data_count = len(self.train_dataset)
		train_status.on_start_train()
		
		try:
		
			while True:
				
				train_status.clear_iter()
				train_status.epoch_number = train_status.epoch_number + 1
				train_status.time_start = time.time()
				train_status.on_start_epoch()
				
				module.train()
				
				# Train batch
				for batch_x, batch_y in self.train_loader:
					
					batch_x = batch_x.to(tensor_device)
					batch_y = batch_y.to(tensor_device)
					batch_x, batch_y = model.convert_batch(x=batch_x, y=batch_y)
					
					train_status.on_start_batch_train(batch_x, batch_y)
					
					# Predict model
					batch_predict = module(batch_x)

					# Get loss value
					loss_value = self.loss(batch_predict, batch_y)
					train_status.loss_train_iter = train_status.loss_train_iter + loss_value.item()
					
					# Gradient
					self.optimizer.zero_grad()
					loss_value.backward()
					self.optimizer.step()
					
					# Calc accuracy
					accuracy = model.check_answer_batch(
						train_status = train_status,
						batch_x = batch_x,
						batch_y = batch_y,
						batch_predict = batch_predict,
						type = "train"
					)
					train_status.acc_train_iter = train_status.acc_train_iter + accuracy
					train_status.batch_train_iter = train_status.batch_train_iter + 1
					train_status.train_count_iter = train_status.train_count_iter + batch_x.shape[0]
					
					train_status.time_end = time.time()
					train_status.on_end_batch_train(batch_x, batch_y)
					
					del batch_x, batch_y
					
					# Clear CUDA
					if torch.cuda.is_available():
						torch.cuda.empty_cache()
				
				module.eval()
				
				# Test batch
				for batch_x, batch_y in self.test_loader:
					
					batch_x = batch_x.to(tensor_device)
					batch_y = batch_y.to(tensor_device)
					batch_x, batch_y = model.convert_batch(x=batch_x, y=batch_y)
					
					train_status.on_start_batch_test(batch_x, batch_y)
					
					# Predict model
					batch_predict = module(batch_x)
					
					# Get loss value
					loss_value = self.loss(batch_predict, batch_y)
					train_status.loss_test_iter = train_status.loss_test_iter + loss_value.item()
					
					# Calc accuracy
					accuracy = model.check_answer_batch(
						train_status = train_status,
						batch_x = batch_x,
						batch_y = batch_y,
						batch_predict = batch_predict,
						type = "test"
					)
					train_status.acc_test_iter = train_status.acc_test_iter + accuracy
					train_status.batch_test_iter = train_status.batch_test_iter + 1
					train_status.test_count_iter = train_status.test_count_iter + batch_x.shape[0]
					
					train_status.time_end = time.time()
					train_status.on_end_batch_test(batch_x, batch_y)
					
					del batch_x, batch_y
					
					# Clear CUDA
					if torch.cuda.is_available():
						torch.cuda.empty_cache()
				
				# Epoch callback
				train_status.time_end = time.time()
				train_status.on_end_epoch()
				self.on_end_epoch()
				
				if not train_status.do_training:
					break
			
		except KeyboardInterrupt:
			
			print ("")
			print ("Stopped manually")
			print ("")
			
			pass
		
		train_status.on_end_train()
	

def do_train(model, *args, **kwargs):
	
	"""
	Start training
	"""
	
	trainer = Trainer(model, *args, **kwargs)
	
	# Train the model
	if not trainer.check_is_trained():
		print ("Train model " + str(model.model_name))
		trainer.train()
	
	return trainer