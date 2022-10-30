# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT
##

import os, time, torch, sqlite3
import numpy as np
import matplotlib.pyplot as plt

from torchsummary import summary
from .train import TrainStatus, TrainHistory
from .layer import AbstractLayerFactory
from .utils import *


class Model(torch.nn.Module):
	
	
	def __init__(self, *args, **kwargs):
		
		torch.nn.Module.__init__(self)
		
		if not hasattr(self, "_history"):
			self._history = TrainHistory()
		
		if not hasattr(self, "_input_shape"):
			self._input_shape = (1)
		
		if not hasattr(self, "_output_shape"):
			self._output_shape = (1)
		
		if not hasattr(self, "_onnx_path"):
			self._onnx_path = os.path.join(os.getcwd(), "web")
		
		if not hasattr(self, "_onnx_opset_version"):
			self._onnx_opset_version = 9
		
		if not hasattr(self, "_model_name"):
			self._model_name = ""
		
		if not hasattr(self, "_model_path"):
			self._model_path = ""
		
		if not hasattr(self, "_repository_path"):
			self._repository_path = ""
		
		if not hasattr(self, "_is_debug"):
			self._is_debug = False
		
		if not hasattr(self, "_convert_batch"):
			self._convert_batch = None
		
		if "input_shape" in kwargs:
			self._input_shape = kwargs["input_shape"]
		
		if "output_shape" in kwargs:
			self._input_shape = kwargs["output_shape"]
		
		if "onnx_path" in kwargs:
			self._onnx_path = kwargs["onnx_path"]
		
		if "onnx_opset_version" in kwargs:
			self._onnx_opset_version = kwargs["onnx_opset_version"]
		
		if "model_name" in kwargs:
			self._model_name = kwargs["model_name"]
		
		if "repository_path" in kwargs:
			self.set_repository_path(kwargs["repository_path"])
			self._repository_path = kwargs["repository_path"]
		
		if "model_path" in kwargs:
			self.set_model_path(kwargs["model_path"])
		
		if "debug" in kwargs:
			self._is_debug = kwargs["debug"]
		
		if "convert_batch" in kwargs:
			self._convert_batch = kwargs["convert_batch"]
	
	
	def is_debug(self, value):
		
		"""
		Set debug level
		"""
		
		self._is_debug = value
	
	
	def get_model_name(self):
		
		"""
		Returns model name
		"""
		
		return self._model_name
	
	
	def get_onnx_path(self):
		
		"""
		Returns model onnx path
		"""
		
		return os.path.join(self._onnx_path, self._model_name + ".onnx")
	
	
	def get_model_path(self):
		
		"""
		Returns model folder path
		"""
		
		return self._model_path
	
	
	def get_model_file_path(self, epoch_number=None):
		
		"""
		Returns model file path
		"""
		
		file_path = os.path.join( self.get_model_path(), "model.data" )
		
		if epoch_number is not None:
			file_path = os.path.join( self.get_model_path(), "model-" + str(epoch_number) + ".data" )
		
		return file_path
	
	
	def get_optimizer_file_path(self, epoch_number=None):
		
		"""
		Returns optimizer file_path
		"""
		
		epoch_number = self._history.epoch_number
		file_path = os.path.join( self.get_model_path(), "model-" +
			str(epoch_number) + "-optimizer.data" )
		
		return file_path
	
	
	def set_model_path(self, path):
		
		"""
		Set model folder path
		"""
		
		self._model_path = path
	
	
	def set_repository_path(self, repository_path):
		
		"""
		Set repository path
		"""
		
		self.set_model_path( os.path.join(repository_path, self._model_name) )
	
	
	def convert_batch(self, x=None, y=None):
		
		"""
		Convert batch
		"""
		
		if self._convert_batch is not None:
			return self._convert_batch(self, x=x, y=y)
		
		if x is not None:
			x = x.to(torch.float)
		
		if y is not None:
			y = y.to(torch.float)
		
		return x, y
	
	
	def create_model(self):
		
		"""
		Create model
		"""
		
		self.module = None
		
		if self._layers is not None:
			
			self.module = ExtendModule(self)
			self.module.init_layers(self._layers, debug=self._is_debug)
			
	
	def summary(self, verbose=True, tensor_device=None):
		
		"""
		Show model summary
		"""
		
		print ("Model name: " + self._model_name)
		
		if tensor_device is None:
			tensor_device = get_tensor_device()
		
		model = self.to(tensor_device)
		summary(model, tuple(self._input_shape), device=str(tensor_device))
	
	
	def check_answer_batch(self, **kwargs):
		
		"""
		Check batch. Returns count right answers
		"""
		
		batch_y = kwargs["batch_y"]
		batch_predict = kwargs["batch_predict"]
		
		batch_y = torch.argmax(batch_y, dim=1)
		batch_predict = torch.argmax(batch_predict, dim=1)
		res = torch.sum( torch.eq(batch_y, batch_predict) ).item()
		
		return res
	
	
	def predict(self, x, tensor_device=None):
		
		"""
		Predict model
		"""
		
		if tensor_device is None:
			tensor_device = get_tensor_device()
		
		x = x.to(tensor_device)
		module = self.module.to(tensor_device)
		x, _ = self.convert_batch(x=x)
		
		y = module(x)
		
		return y
	
	
	def predict_dataset(self, dataset, batch_size=32, tensor_device=None):
		
		from torch.utils.data import DataLoader
		
		if tensor_device is None:
			tensor_device = get_tensor_device()
		
		num_workers = os.cpu_count()
		
		loader = DataLoader(
			dataset,
			num_workers=num_workers,
			batch_size=batch_size,
			drop_last=False,
			shuffle=False
		)
		
		res = torch.tensor([])
		module = self.module.to(tensor_device)
		
		for batch_x, _ in loader:
			
			batch_x = batch_x.to(tensor_device)
			
			batch_x, _ = self.convert_batch(x=batch_x)
			
			batch_predict = module(batch_x)
			batch_predict = batch_predict.to( res.device )
			
			res = torch.cat( (res, batch_predict) )
			
		return res
	
	
	def save_train_history(self, show=False):
		
		"""
		Save train history
		"""
		
		plt = self._history.get_plot()
		history_image = os.path.join( self.get_model_path(), "model.png" )
		make_parent_dir(history_image)
		plt.savefig(history_image)
		
		if show:
			plt.show()
		
		return plt
	
	
	def open_model_db(self, db_path):
		
		"""
		Open database
		"""
		
		is_create = False
		
		make_parent_dir(db_path)
		
		if not os.path.exists(db_path):
			is_create = True
		
		db_con = sqlite3.connect( db_path, isolation_level=None )
		db_con.row_factory = sqlite3.Row
		
		cur = db_con.cursor()
		res = cur.execute("PRAGMA journal_mode = WAL;")
		cur.close()
		
		if is_create:
			self.create_model_db(db_con)
			
		return db_con
	
	
	def create_model_db(self, db_con):
		
		"""
		Create database
		"""
		
		cur = db_con.cursor()
		
		sql = """CREATE TABLE history(
			model_name text NOT NULL,
			epoch_number integer NOT NULL,
			time real NOT NULL,
			acc_train real NOT NULL,
			acc_test real NOT NULL,
			acc_rel real NOT NULL,
			loss_train real NOT NULL,
			loss_test real NOT NULL,
			batch_train_iter integer NOT NULL,
			batch_test_iter integer NOT NULL,
			train_count_iter integer NOT NULL,
			test_count_iter integer NOT NULL,
			loss_train_iter real NOT NULL,
			loss_test_iter real NOT NULL,
			acc_train_iter real NOT NULL,
			acc_test_iter real NOT NULL,
			info text NOT NULL,
			PRIMARY KEY ("model_name", "epoch_number")
		)"""
		cur.execute(sql)
		
		cur.close()
		
		
	def save_train_status(self):
			
		"""
		Save train status
		"""
		
		epoch_number = self._history.epoch_number
		epoch_record = self._history.get_epoch(epoch_number)
		
		if epoch_number > 0 and epoch_record is not None:
			
			db_path = os.path.join( self.get_model_path(), "model.db" )
			db_con = self.open_model_db(
				db_path = db_path
			)
			
			sql = """
				insert or replace into history (
					model_name, epoch_number, acc_train,
					acc_test, acc_rel, loss_train, loss_test,
					batch_train_iter, batch_test_iter,
					train_count_iter, test_count_iter,
					loss_train_iter, loss_test_iter,
					acc_train_iter, acc_test_iter,
					time, info
				) values
				(
					:model_name, :epoch_number, :acc_train,
					:acc_test, :acc_rel, :loss_train, :loss_test,
					:batch_train_iter, :batch_test_iter,
					:train_count_iter, :test_count_iter,
					:loss_train_iter, :loss_test_iter,
					:acc_train_iter, :acc_test_iter,
					:time, :info
				)
			"""
			
			history = self._history
			obj = {
				"model_name": self._model_name,
				"epoch_number": epoch_number,
				"acc_train": epoch_record["acc_train"],
				"acc_test": epoch_record["acc_test"],
				"acc_rel": epoch_record["acc_rel"],
				"loss_train": epoch_record["loss_train"],
				"loss_test": epoch_record["loss_test"],
				"time": epoch_record["time"],
				"batch_train_iter": epoch_record["batch_train_iter"],
				"batch_test_iter": epoch_record["batch_test_iter"],
				"train_count_iter": epoch_record["train_count_iter"],
				"test_count_iter": epoch_record["test_count_iter"],
				"loss_train_iter": epoch_record["loss_train_iter"],
				"loss_test_iter": epoch_record["loss_test_iter"],
				"acc_train_iter": epoch_record["acc_train_iter"],
				"acc_test_iter": epoch_record["acc_test_iter"],
				"info": "{}",
			}
			
			cur = db_con.cursor()
			res = cur.execute(sql, obj)
			cur.close()
			
			db_con.commit()
			db_con.close()


	def load_train_status(self, epoch_number=None):
			
		"""
		Load train status
		"""
		
		db_path = os.path.join( self.get_model_path(), "model.db" )
		db_con = self.open_model_db(
			db_path = db_path
		)
		
		sql = """
			select * from "history"
			where model_name=:model_name
			order by epoch_number asc
		"""
		
		cur = db_con.cursor()
		res = cur.execute(sql, {"model_name": self._model_name})
		
		records = res.fetchall()
		
		for record in records:
			
			if epoch_number is not None:
				if record["epoch_number"] > epoch_number:
					continue
			
			self._history.add( record )
		
		cur.close()
		
		db_con.commit()
		db_con.close()


	def save_model_file(self, file_path):
			
		"""
		Save model to file
		"""
		
		state_dict = self.state_dict()
		make_parent_dir(file_path)
		torch.save(state_dict, file_path)
	
	
	def save(self, save_epoch=False):
		
		"""
		Save model
		"""
		
		file_path = self.get_model_file_path()
		self.save_model_file(file_path)
		
		if save_epoch:
			epoch_number = self._history.epoch_number
			file_path = self.get_model_file_path(epoch_number)
			self.save_model_file(file_path)
		
		self.save_train_status()

	
	def save_onnx(self, tensor_device=None):
		
		"""
		Save model to onnx file
		"""
		
		import torch, torch.onnx
		
		if tensor_device is None:
			tensor_device = get_tensor_device()
		
		onnx_model_path = self.get_onnx_path()
		
		# Prepare data input
		data_input = torch.zeros(self.input_shape).to(torch.float32)
		data_input = data_input[None,:]
		
		# Move to tensor device
		model = self.to(tensor_device)
		data_input = data_input.to(tensor_device)
		
		torch.onnx.export(
			model,
			data_input,
			onnx_model_path,
			opset_version = self.onnx_opset_version,
			input_names = ['input'],
			output_names = ['output'],
			verbose=False
		)
	
	
	def load_model_file(self, file_path):
			
		"""
		Load model
		"""
		
		state_dict = None
		
		try:
			if os.path.isfile(file_path):
				state_dict = torch.load(file_path)
		
		except:
			pass
		
		if state_dict:
			self.load_state_dict(state_dict)
			return True
		
		return False
	
	
	def load(self, model_path=None, repository_path=None, file_path=None, epoch_number=None):
		
		"""
		Load model from file
		"""
		
		if repository_path is not None:
			self.set_repository_path(repository_path)
		
		if model_path is not None:
			self.set_model_path(model_path)
		
		if file_path is None:
			file_path = self.get_model_file_path(epoch_number)
		
		is_loaded = self.load_model_file(file_path)
		
		if is_loaded:
			self.load_train_status(epoch_number)
	
	
	def save_optimizer(self, optimizer, file_path=None):
		
		"""
		Save optimizer
		"""
		
		if file_path is None:
			epoch_number = self._history.epoch_number
			file_path = self.get_optimizer_file_path(epoch_number)
		
		make_parent_dir(file_path)
		torch.save(optimizer.state_dict(), file_path)
	

	def load_optimizer(self, optimizer, file_path=None):
		
		"""
		Load optimizer
		"""
		
		if file_path is None:
			epoch_number = self._history.epoch_number
			file_path = self.get_optimizer_file_path(epoch_number)
		
		state_dict = None
		if os.path.isfile(file_path):
			try:
				if os.path.isfile(file_path):
					state_dict = torch.load(file_path)
			except:
				pass
		
		if state_dict:
			optimizer.load_state_dict(state_dict)


class ExtendModule(torch.nn.Module):
	
	def __init__(self, model):
		
		"""
		Constructor
		"""
		
		super(ExtendModule, self).__init__()
		
		self._layers = []
		self._shapes = []
		self._input_shape = kwargs["input_shape"] if "input_shape" in kwargs else (1)
		self._output_shape = kwargs["output_shape"] if "output_shape" in kwargs else (1)
		self._layers = kwargs["layers"] if "layers" in kwargs else None
		
		if self._layers is not None:
			self.init_layers(self._layers)
		
	
	def forward(self, x):
		
		"""
		Forward model
		"""
		
		for index, obj in enumerate(self._layers):
			
			if isinstance(obj, AbstractLayerFactory):
				x = obj.forward(x)
			
			elif isinstance(obj, torch.nn.Module):
				x = obj.forward(x)
		
		return x
	
	
	def init_layers(self, layers=[], debug=False):
		
		"""
		Init layers
		"""
		
		self._layers = layers
		
		input_shape = self.input_shape
		output_shape = self.output_shape
		
		arr = list(input_shape)
		arr.insert(0, 1)
		
		vector_x = torch.zeros( tuple(arr) )
		self._shapes.append( ("Input", vector_x.shape) )
		
		if debug:
			print ("Input:" + " " + str( tuple(vector_x.shape) ))
		
		index = 1
		
		for obj in self._layers:
			
			if isinstance(obj, AbstractLayerFactory):
				
				layer_factory: AbstractLayerFactory = obj
				layer_factory.parent = self
				
				name = layer_factory.get_name()
				layer_name = str( index ) + "_" + name
				layer_factory.layer_name = layer_name
				layer_factory.input_shape = vector_x.shape
				
				layer, vector_x = layer_factory.create_layer(vector_x)
				layer_factory.output_shape = vector_x.shape
				
				self._shapes.append( (layer_name, vector_x.shape) )
				
				if debug:
					print (layer_name + " => " + str(tuple(vector_x.shape)))
				
				if layer:
					self.add_module(layer_name, layer)
					
				index = index + 1
				
			else:
				
				layer_name = str( index ) + "_Layer"
				self.add_module(layer_name, obj)
