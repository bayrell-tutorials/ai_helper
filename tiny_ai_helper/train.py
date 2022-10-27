# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT
##

import os, time, torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
from .utils import get_tensor_device, list_files, indexOf


class TrainHistory:
	
	def __init__(self):
		
		self.epoch_number = 0
		self.epoch = {}
	
	
	def add(self, record):
		
		"""
		Add train history from record
		"""
		
		epoch_number = record["epoch_number"]
		self.epoch_number = epoch_number
		self.epoch[epoch_number] = {
			
			"loss_train": record["loss_train"],
			"loss_test": record["loss_test"],
			"acc_train": record["acc_train"],
			"acc_test": record["acc_test"],
			"acc_rel": record["acc_rel"],
			"time": record["time"],
			"batch_train_iter": record["batch_train_iter"],
			"batch_test_iter": record["batch_test_iter"],
			"train_count_iter": record["train_count_iter"],
			"test_count_iter": record["test_count_iter"],
			"loss_train_iter": record["loss_train_iter"],
			"loss_test_iter": record["loss_test_iter"],
			"acc_train_iter": record["acc_train_iter"],
			"acc_test_iter": record["acc_test_iter"],
			
		}
	
	
	def add_train_status(self, train_status):
		
		"""
		Add train history from train status
		"""
		
		loss_train = train_status.get_loss_train()
		loss_test = train_status.get_loss_test()
		acc_train = train_status.get_acc_train()
		acc_test = train_status.get_acc_test()
		acc_rel = train_status.get_acc_rel()
		time = train_status.get_time()
		
		self.add( {
			
			"epoch_number": train_status.epoch_number,
			"loss_train": loss_train,
			"loss_test": loss_test,
			"acc_train": acc_train,
			"acc_test": acc_test,
			"acc_rel": acc_rel,
			"time": time,
			"batch_train_iter": train_status.batch_train_iter,
			"batch_test_iter": train_status.batch_test_iter,
			"train_count_iter": train_status.train_count_iter,
			"test_count_iter": train_status.test_count_iter,
			"loss_train_iter": train_status.loss_train_iter,
			"loss_test_iter": train_status.loss_test_iter,
			"acc_train_iter": train_status.acc_train_iter,
			"acc_test_iter": train_status.acc_test_iter,
			
		} )
	
	
	def get_epoch(self, epoch_number):
		
		"""
		Returns epoch by number
		"""
		
		if not(epoch_number in self.epoch):
			return None
			
		return self.epoch[epoch_number]
	
	
	def get_metrics(self, metric_name, with_index=False):
		
		"""
		Returns metrics by name
		"""
		
		res = []
		for index in self.epoch:
			if with_index:
				if isinstance(metric_name, list):
					res2 = [ index ]
					for name in metric_name:
						res2.append( self.epoch[index][name] )
					res.append( tuple(res2) )
				else:
					res.append( (index, self.epoch[index][metric_name]) )
			else:
				res.append( self.epoch[index][metric_name] )
		
		return res
		
	
	def get_plot(self):
		
		"""
		Returns train history
		"""
		
		import matplotlib.pyplot as plt
		import numpy as np
		
		loss_train = self.get_metrics("loss_train")
		loss_test = self.get_metrics("loss_test")
		acc_train = self.get_metrics("acc_train")
		acc_test = self.get_metrics("acc_test")
		
		fig, axs = plt.subplots(2)
		axs[0].plot( np.multiply(loss_train, 100), label='train loss')
		axs[0].plot( np.multiply(loss_test, 100), label='test loss')
		axs[0].legend()
		axs[1].plot( np.multiply(acc_train, 100), label='train acc')
		axs[1].plot( np.multiply(acc_test, 100), label='test acc')
		axs[1].legend()
		plt.xlabel('Epoch')
		
		return plt
	

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
		self.trainer = None
		self.do_training = True
		self.train_data_count = 0
		self.time_start = 0
		self.time_end = 0
	
	def set_model(self, model):
		
		self.clear()
		self.model = model
		self.epoch_number = model.history.epoch_number
		
		if self.epoch_number > 0:
			
			record = model.history.get_epoch(self.epoch_number)
			
			self.batch_train_iter = record["batch_train_iter"]
			self.batch_test_iter = record["batch_test_iter"]
			self.train_count_iter = record["train_count_iter"]
			self.test_count_iter = record["test_count_iter"]
			self.loss_train_iter = record["loss_train_iter"]
			self.loss_test_iter = record["loss_test_iter"]
			self.acc_train_iter = record["acc_train_iter"]
			self.acc_test_iter = record["acc_test_iter"]
	
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
	

class TrainVerboseCallback:
	
	
	def on_end_batch_train(self, trainer):
		
		acc_train = trainer.train_status.get_acc_train()
		loss_train = trainer.train_status.get_loss_train()
		time = trainer.train_status.get_time()
		
		msg = ("\rStep {epoch_number}, {iter_value}%" +
			", acc: .{acc}, loss: .{loss}, time: {time}s"
		).format(
			epoch_number = trainer.train_status.epoch_number,
			iter_value = round(trainer.train_status.get_iter_value() * 100),
			loss = str(round(loss_train * 10000)).zfill(4),
			acc = str(round(acc_train * 100)).zfill(2),
			time = str(round(time)),
		)
		
		print (msg, end='')
	
	
	def on_end_epoch(self, trainer):
		
		"""
		Epoch
		"""
		
		loss_train = trainer.train_status.get_loss_train()
		loss_test = trainer.train_status.get_loss_test()
		acc_train = trainer.train_status.get_acc_train()
		acc_test = trainer.train_status.get_acc_test()
		acc_rel = trainer.train_status.get_acc_rel()
		time = trainer.train_status.get_time()
		
		print ("\r", end='')
		
		msg = ("Step {epoch_number}, " +
			"acc: .{acc_train}, " +
			"acc_test: .{acc_test}, " +
			"acc_rel: {acc_rel}, " +
			"loss: .{loss_train}, " +
			"loss_test: .{loss_test}, " +
			"time: {time}s, "
		).format(
			epoch_number = trainer.train_status.epoch_number,
			loss_train = str(round(loss_train * 10000)).zfill(4),
			loss_test = str(round(loss_test * 10000)).zfill(4),
			acc_train = str(round(acc_train * 100)).zfill(2),
			acc_test = str(round(acc_test * 100)).zfill(2),
			acc_rel = str(round(acc_rel * 100) / 100),
			time = str(round(time)),
		)
		
		print (msg)
	
	
class TrainSaveCallback:
	
	
	def detect_type(self, file_name):
	
		import re
		
		file_type = ""
		epoch_index = 0
		
		result = re.match(r'^model-(?P<id>[0-9]+)-optimizer\.data$', file_name)
		if result:
			return "optimizer", int(result.group("id"))
		
		result = re.match(r'^model-(?P<id>[0-9]+)\.data$', file_name)
		if result:
			return "model", int(result.group("id"))
		
		return file_type, epoch_index
	
	
	def save_epoch_indexes(self, model, epoch_indexes):
		
		"""
		Save epoch by indexes
		"""
		
		model_path = model.get_model_path()
		files = list_files( model_path )
		
		for file_name in files:
			
			file_type, epoch_index = self.detect_type(file_name)
			if file_type in ["model", "optimizer"] and not (epoch_index in epoch_indexes):
				
				file_path = os.path.join( model_path, file_name )
				os.unlink(file_path)
				
	
	
	def get_the_best_epoch(self, model, epoch_count=5, indexes=False):
		
		"""
		Returns teh best epoch
		"""
		
		metrics = model.history.get_metrics(["loss_test", "acc_rel"], with_index=True)
		
		def get_key(item):
			return [item[1], item[2]]

		metrics.sort(key=get_key)
		
		res = []
		res_count = 0
		metrics_len = len(metrics)
		loss_test_last = 100
		for index in range(metrics_len):
			
			res.append( metrics[index] )
			
			if loss_test_last != metrics[index][1]:
				res_count = res_count + 1
			
			loss_test_last = metrics[index][1]
			
			if res_count > epoch_count:
				break
		
		if not indexes:
			return res
			
		res_indexes = []
		for index in range(len(res)):
			res_indexes.append( res[index][0] )
			
		return res_indexes
	
	
	def save_the_best_epoch(self, model, epoch_count=5):
		
		"""
		Save the best models
		"""
		
		if model.history.epoch_number > 0 and epoch_count > 0:
			
			epoch_indexes = self.get_the_best_epoch(model, epoch_count, indexes=True)
			epoch_indexes.append( model.history.epoch_number )
			
			self.save_epoch_indexes(model, epoch_indexes)
	
	
	def on_end_epoch(self, trainer):
		
		"""
		On epoch end
		"""
		
		trainer.model.save(
			save_epoch=trainer.save_epoch
		)
		
		if trainer.save_epoch:
			trainer.model.save_optimizer(trainer.optimizer)
			
			"""
			Save best epoch
			"""
			self.save_the_best_epoch(
				trainer.model,
				epoch_count=trainer.save_epoch_count
			)
		
		

class Trainer:
	
	def __init__(self, model, *args, **kwargs):
		
		self.model = model
		
		self.train_status = TrainStatus()
		self.train_status.trainer = self
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
		self.lr = kwargs["lr"] if "lr" in kwargs else 1e-3
		
		self.train_dataset = kwargs["train_dataset"] if "train_dataset" in kwargs else False
		self.test_dataset = kwargs["test_dataset"] if "test_dataset" in kwargs else False
		self.tensor_device = kwargs["tensor_device"] if "tensor_device" in kwargs else None
		self.save_epoch = kwargs["save_epoch"] if "save_epoch" in kwargs else True
		self.save_epoch_count = kwargs["save_epoch_count"] if "save_epoch_count" in kwargs else 5
		
		self._check_is_trained = kwargs["check_is_trained"] \
			if "check_is_trained" in kwargs else None
		
		if "callbacks" in kwargs:
			self.callbacks = kwargs["callbacks"]
		else:
			self.callbacks = [
				TrainVerboseCallback(),
				TrainSaveCallback(),
			]
		
	
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
	
	
	"""	====================== Events ====================== """
	
	def on_start_train(self):
		
		"""
		Start train event
		"""
		
		for callback in self.callbacks:
			if hasattr(callback, "on_start_train"):
				callback.on_start_train(self)
		
	
	def on_start_epoch(self):
		
		"""
		Start epoch event
		"""
		
		for callback in self.callbacks:
			if hasattr(callback, "on_start_epoch"):
				callback.on_start_epoch(self)
	
	
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
	
	
	def on_end_epoch(self, **kwargs):
		
		"""
		On epoch end
		"""
		
		self._is_trained = self.check_is_trained()
		
		if self._is_trained:
			self.stop_training()
		
		self.model.history.add_train_status(self.train_status)
		
		for callback in self.callbacks:
			if hasattr(callback, "on_end_epoch"):
				callback.on_end_epoch(self)
		
	
	def on_end_train(self):
		
		"""
		End train event
		"""
		
		for callback in self.callbacks:
			if hasattr(callback, "on_end_train"):
				callback.on_end_train(self)
	
	
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
			self.optimizer = torch.optim.Adam(model.module.parameters(), lr=self.lr)
		
		if self.save_epoch:
			model.load_optimizer(self.optimizer)
			
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
		self.on_start_train()
		
		try:
		
			while True:
				
				train_status.clear_iter()
				train_status.epoch_number = train_status.epoch_number + 1
				train_status.time_start = time.time()
				self.on_start_epoch()
				
				module.train()
				
				# Train batch
				for batch_x, batch_y in self.train_loader:
					
					batch_x = batch_x.to(tensor_device)
					batch_y = batch_y.to(tensor_device)
					batch_x, batch_y = model.convert_batch(x=batch_x, y=batch_y)
					
					self.on_start_batch_train(batch_x, batch_y)
					
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
					self.on_end_batch_train(batch_x, batch_y)
					
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
					
					self.on_start_batch_test(batch_x, batch_y)
					
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
					self.on_end_batch_test(batch_x, batch_y)
					
					del batch_x, batch_y
					
					# Clear CUDA
					if torch.cuda.is_available():
						torch.cuda.empty_cache()
				
				# Epoch callback
				train_status.time_end = time.time()
				self.on_end_epoch()
				
				if not train_status.do_training:
					break
			
		except KeyboardInterrupt:
			
			print ("")
			print ("Stopped manually")
			print ("")
			
			pass
		
		self.on_end_train()
	

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


class FilesListDataset(Dataset):
	
	def __init__(self, files, files_path="", transform=None, get_tensor_from_answer=None):
		
		self.files = files
		self.files_path = files_path
		self.transform = transform
		self.get_tensor_from_answer = get_tensor_from_answer
	
	
	def __getitem__(self, index):
		
		file_path = ""
		answer = torch.tensor([])
		
		if self.get_tensor_from_answer is None:
			file_path = self.files[index]
		else:
			file_path, answer = self.files[index]
		
		if self.files_path != "":
			file_path = os.path.join(self.files_path, file_path)
		
		tensor = self.transform(file_path)
		
		if self.get_tensor_from_answer is not None:
			answer = self.get_tensor_from_answer(answer)
		
		return ( tensor, answer )
	
	def __len__(self):
		return len(self.files)
