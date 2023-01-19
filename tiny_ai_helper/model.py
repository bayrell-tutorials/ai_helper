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
		
		self.optimizer = None
		
		if not hasattr(self, "_history"):
			self._history = TrainHistory()
		
		if not hasattr(self, "_input_shape"):
			self._input_shape = (1)
		
		if not hasattr(self, "_output_shape"):
			self._output_shape = (1)
		
		if not hasattr(self, "_model_name"):
			self._model_name = ""
		
		if not hasattr(self, "_is_debug"):
			self._is_debug = False
		
		if not hasattr(self, "_convert_batch"):
			self._convert_batch = None
		
		if "input_shape" in kwargs:
			self._input_shape = kwargs["input_shape"]
		
		if "output_shape" in kwargs:
			self._input_shape = kwargs["output_shape"]
		
		if "model_name" in kwargs:
			self._model_name = kwargs["model_name"]
		
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
	
	
	def get_epoch_number(self):
		
		"""
		Returns epoch number
		"""
		
		return self._history.epoch_number
	
	
	def get_epoch(self, epoch_number):
		
		"""
		Returns epoch by index
		"""
		
		return self._history.get_epoch(epoch_number)
	
	
	def convert_batch(self, x=None, y=None):
		
		"""
		Convert batch
		"""
		
		if self._convert_batch is not None:
			return self._convert_batch(self, x=x, y=y)
		
		return x, y
			
	
	def summary(self, verbose=True, device=None):
		
		"""
		Show model summary
		"""
		
		print ("Model name: " + self.get_model_name())
		
		if device is None:
			device = get_tensor_device()
		
		model = self.to(device)
		summary(model, tuple(self._input_shape), device=str(device))
	
	
	def predict(self, x, tensor_device=None):
		
		"""
		Predict model
		"""
		
		if tensor_device is None:
			tensor_device = get_tensor_device()
		
		x = x.to(tensor_device)
		module = self.to(tensor_device)
		x, _ = self.convert_batch(x=x)
		
		module.eval()
		y = module(x)
		
		return y
	
	
	def predict_dataset(self, dataset, batch_size=32, tensor_device=None, progress=None):
		
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
		module = self.to(tensor_device)
		count_total = len(dataset)
		count_iter = 0
		
		module.eval()
		
		for batch_x in loader:
			
			batch_x = batch_x.to(tensor_device)
			
			batch_x, _ = self.convert_batch(x=batch_x)
			
			batch_predict = module(batch_x)
			batch_predict = batch_predict.to( res.device )
			
			res = torch.cat( (res, batch_predict) )
			
			count_iter = count_iter + batch_x.shape[0]
			
			if progress is not None:
				progress(count_iter, count_total)
			
			del batch_x
			
			# Clear CUDA
			if torch.cuda.is_available():
				torch.cuda.empty_cache()
		
		return res
	
	
	def save_train_history(self, model_path=None, show=False):
		
		"""
		Save train history
		"""
		
		if model_path is not None:
			
			file_path = model_path.model_name(self.get_model_name()).get_model_file_path()
			dir_name = os.path.dirname(file_path)
			make_parent_dir(dir_name)
			
			# Save loss metrics
			f = create_pyplot_figure()
			ax = f.gca()
			self._history.plot(ax, "loss")
			
			file_path = os.path.join( dir_name, "model_loss.png" )
			get_pyplot_image(f).save(file_path)
			
			# Save val metrics
			f = create_pyplot_figure()
			ax = f.gca()
			self._history.plot(ax, "acc")
			
			file_path = os.path.join( dir_name, "model_acc.png" )
			get_pyplot_image(f).save(file_path)
			
			# Show image
			if show:
				f = create_pyplot_figure()
				ax = f.subplots(2)
				
				self._history.plot(ax[0], "loss")
				self._history.plot(ax[1], "acc")
				
				get_pyplot_image(f).show()
			
	
	
	def save(self, model_path=None, save_epoch=False, save_metrics={}):
		
		"""
		Save model
		"""
		
		if model_path is not None:
			model_path.save(
				model=self,
				save_epoch=save_epoch,
				save_metrics=save_metrics,
			)
		
	
	def load(self, model_path=None):
		
		"""
		Load model
		"""
		
		save_metrics = None
		
		if model_path is not None:
			save_metrics = model_path.load(self)
		
		return save_metrics
	

class ModelPath:
	
	def __init__(self, *args, **kwargs):
		
		self._onnx_path = ""
		self._file_path = ""
		self._repository_path = ""
		self._model_name = ""
		self._folder_path = ""
		self._file_name = ""
		self._epoch_number = 0
		
		if "onnx_path" in kwargs:
			self._onnx_path = kwargs["onnx_path"]
		
		if "file_path" in kwargs:
			self._file_path = kwargs["file_path"]
		
		if "repository_path" in kwargs:
			self._repository_path = kwargs["repository_path"]
		
		if "model_name" in kwargs:
			self._model_name = kwargs["model_name"]
		
		if "folder_path" in kwargs:
			self._folder_path = kwargs["folder_path"]
		
		if "file_name" in kwargs:
			self._file_name = kwargs["file_name"]
		
		if "epoch_number" in kwargs:
			self._epoch_number = kwargs["epoch_number"]
	
	
	def clone(self):
		path = ModelPath()
		path._onnx_path = self._onnx_path
		path._file_path = self._file_path
		path._repository_path = self._repository_path
		path._model_name = self._model_name
		path._folder_path = self._folder_path
		path._file_name = self._file_name
		path._epoch_number = self._epoch_number
		return path
	
	def file_path(self, file_path):
		path = self.clone()
		path._file_path = file_path
		return path
	
	def repository_path(self, repository_path):
		path = self.clone()
		path._repository_path = repository_path
		return path
	
	def model_name(self, model_name):
		path = self.clone()
		path._model_name = model_name
		return path
	
	def folder_path(self, folder_path):
		path = self.clone()
		path._folder_path = folder_path
		return path
	
	def file_name(self, file_name):
		path = self.clone()
		path._file_name = file_name
		return path
	
	def epoch_number(self, epoch_number):
		path = self.clone()
		path._epoch_number = epoch_number
		return path
	
	def onnx_path(self, onnx_path):
		path = self.clone()
		path._onnx_path = onnx_path
		return path
	
	def get_model_file_path(self):
		
		"""
		Returns model file path
		"""
		
		file_path = ""
		
		if self._file_path != "":
			file_path = self._file_path
		else:
			if self._repository_path != "":
				file_path = os.path.join( file_path, self._repository_path )
			
			if self._model_name != "":
				file_path = os.path.join( file_path, self._model_name )
			
			if self._folder_path != "":
				file_path = self._folder_path
			
			if self._file_name != "":
				file_path = os.path.join( file_path, self._file_name )
			
			elif self._epoch_number > 0:
				file_path = os.path.join( file_path, "model-" + str(self._epoch_number) + ".data" )
			
			else:
				file_path = os.path.join( file_path, "model.data" )
		
		return file_path
	
	
	def get_model_onnx_path(self):
		
		"""
		Returns model onnx path
		"""
		
		onnx_path = self._onnx_path
		if onnx_path == "":
			onnx_path = self.get_model_file_path()
			onnx_path = os.path.dirname(onnx_path)
		
		return os.path.join(onnx_path, "model.onnx")
	
	
	def create_model_db(self, db_con):
		
		"""
		Create database
		"""
		
		cur = db_con.cursor()
		
		sql = """CREATE TABLE history(
			model_name text NOT NULL,
			epoch_number integer NOT NULL,
			time real NOT NULL,
			lr text NOT NULL,
			acc_train real NOT NULL,
			acc_val real NOT NULL,
			acc_rel real NOT NULL,
			loss_train real NOT NULL,
			loss_val real NOT NULL,
			batch_train_iter integer NOT NULL,
			batch_val_iter integer NOT NULL,
			count_train_iter integer NOT NULL,
			count_val_iter integer NOT NULL,
			loss_train_iter real NOT NULL,
			loss_val_iter real NOT NULL,
			acc_train_iter real NOT NULL,
			acc_val_iter real NOT NULL,
			info text NOT NULL,
			PRIMARY KEY ("model_name", "epoch_number")
		)"""
		cur.execute(sql)
		
		cur.close()
	
	
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
	
	
	def load(self, model):
		
		"""
		Load model from file
		"""
		
		model_path = self
		save_metrics = None
		model._history.clear()
		
		# Setup epoch_number
		epoch_number = 0
		if model_path._epoch_number > 0:
			epoch_number = model_path._epoch_number
		
		# Setup model_name
		model_path = model_path.model_name(model.get_model_name())
		
		# Load train status
		if epoch_number == 0:
			model_path.load_train_status(model)
			
			# Setup new epoch_number from model history
			if model.get_epoch_number() > 0:
				model_path = model_path.epoch_number( model.get_epoch_number() )
		
		# Get file path
		file_path = model_path.get_model_file_path()
		
		if os.path.isfile(file_path):
			save_metrics = torch.load(file_path)
		
		if save_metrics is not None:
			
			if "state_dict" in save_metrics:
				model.load_state_dict(save_metrics["state_dict"])
			
			if "history" in save_metrics:
				model._history.load_state_dict(save_metrics["history"])
		
		else:
			model._history.clear()
		
		return save_metrics
	
	
	def load_train_status(self, model, epoch_number=0):
			
		"""
		Load train status
		"""
		
		file_path = self.get_model_file_path()
		dir_path = os.path.dirname(file_path)
		
		db_path = os.path.join( dir_path, "model.db" )
		
		if os.path.isfile(db_path):
			db_con = self.open_model_db(
				db_path = db_path
			)
			
			sql = """
				select * from "history"
				where model_name=:model_name
				order by epoch_number asc
			"""
			
			cur = db_con.cursor()
			res = cur.execute(sql, {"model_name": model.get_model_name()})
			
			records = res.fetchall()
			
			model._history.clear()
			
			for record in records:
				
				if epoch_number > 0:
					if record["epoch_number"] > epoch_number:
						continue
				
				model._history.add( record )
			
			cur.close()
			
			db_con.commit()
			db_con.close()
	
		
	def save_train_status(self, model):
			
		"""
		Save train status
		"""
		
		model_path = self
		model_path = model_path.model_name(model.get_model_name())
		
		epoch_number = model.get_epoch_number()
		epoch_record = model.get_epoch(epoch_number)
		
		if epoch_number > 0 and epoch_record is not None:
			
			file_path = model_path.get_model_file_path()
			dir_path = os.path.dirname(file_path)
			
			db_path = os.path.join( dir_path, "model.db" )
			db_con = self.open_model_db(
				db_path = db_path
			)
			
			sql = """
				insert or replace into history (
					model_name, epoch_number, acc_train,
					acc_val, acc_rel, loss_train, loss_val,
					batch_train_iter, batch_val_iter,
					count_train_iter, count_val_iter,
					loss_train_iter, loss_val_iter,
					acc_train_iter, acc_val_iter,
					time, lr, info
				) values
				(
					:model_name, :epoch_number, :acc_train,
					:acc_val, :acc_rel, :loss_train, :loss_val,
					:batch_train_iter, :batch_val_iter,
					:count_train_iter, :count_val_iter,
					:loss_train_iter, :loss_val_iter,
					:acc_train_iter, :acc_val_iter,
					:time, :lr, :info
				)
			"""
			
			history = model._history
			obj = {
				"model_name": model.get_model_name(),
				"epoch_number": epoch_number,
				"acc_train": epoch_record["acc_train"],
				"acc_val": epoch_record["acc_val"],
				"acc_rel": epoch_record["acc_rel"],
				"loss_train": epoch_record["loss_train"],
				"loss_val": epoch_record["loss_val"],
				"time": epoch_record["time"],
				"lr": epoch_record["lr"],
				"batch_train_iter": epoch_record["batch_train_iter"],
				"batch_val_iter": epoch_record["batch_val_iter"],
				"count_train_iter": epoch_record["count_train_iter"],
				"count_val_iter": epoch_record["count_val_iter"],
				"loss_train_iter": epoch_record["loss_train_iter"],
				"loss_val_iter": epoch_record["loss_val_iter"],
				"acc_train_iter": epoch_record["acc_train_iter"],
				"acc_val_iter": epoch_record["acc_val_iter"],
				"info": "{}",
			}
			
			cur = db_con.cursor()
			res = cur.execute(sql, obj)
			cur.close()
			
			db_con.commit()
			db_con.close()


	def save_model_file(self, model, file_path, save_metrics={}):
			
		"""
		Save model to file
		"""
		
		save_metrics["name"] = model.get_model_name()
		save_metrics["history"] = model._history.state_dict()
		save_metrics["state_dict"] = model.state_dict()
		make_parent_dir(file_path)
		torch.save(save_metrics, file_path)
	
	
	def save(self, model, save_epoch=False, save_metrics={}):
		
		"""
		Save model
		"""
		
		file_path1 = ""
		file_path2 = ""
		
		model_path = self
		model_path = model_path.model_name( model.get_model_name() )
		model_path = model_path.epoch_number( 0 )
		
		if save_epoch:
			
			# Save model with training state
			epoch_number = model.get_epoch_number()
			model_path = model_path.epoch_number(epoch_number)
			
			file_path2 = model_path.get_model_file_path()
			self.save_model_file(model, file_path2, save_metrics)
			self.save_train_status(model)
		
		# Save clear model
		file_path1 = model_path.get_model_file_path()
		
		if file_path1 != file_path2:
			self.save_model_file(model, file_path1)
	
	
	
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


class PreparedModel(torch.nn.Module):
	
	def __init__(self, model, weight_path, *args, **kwargs):
		
		torch.nn.Module.__init__(self)
		
		self.model = model
		self.weight_path = weight_path
	
		for param in self.model.parameters():
			param.requires_grad = False
	
	def forward(self, x):
		x = self.model(x)
		return x
	
	def load(self, *args, **kwargs):
		
		"""
		Load model from file
		"""
		
		state_dict = torch.load( self.weight_path )
		self.model.load_state_dict( state_dict )


class ExtendModel(Model):
	
	def __init__(self, *args, **kwargs):
		
		"""
		Constructor
		"""
		
		super(ExtendModel, self).__init__(*args, **kwargs)
		
		self._layers = []
		self._shapes = []
		
		if "input_shape" in kwargs:
			self._input_shape = kwargs["input_shape"]
		
		if "output_shape" in kwargs:
			self._output_shape = kwargs["output_shape"]
		
		if "layers" in kwargs:
			self.init_layers(kwargs["layers"])
		
	
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
	
	
	def init_layers(self, layers=None):
		
		"""
		Init layers
		"""
		
		self._layers = layers
		if self._layers is None:
			return
		
		input_shape = self._input_shape
		output_shape = self._output_shape
		
		arr = list(input_shape)
		arr.insert(0, 1)
		
		vector_x = torch.zeros( tuple(arr) )
		self._shapes.append( ("Input", vector_x.shape) )
		
		if self._is_debug:
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
				
				if self._is_debug:
					print (layer_name + " => " + str(tuple(vector_x.shape)))
				
				if layer:
					self.add_module(layer_name, layer)
					
				index = index + 1
				
			else:
				
				layer_name = str( index ) + "_Layer"
				self.add_module(layer_name, obj)


class CustomModel(Model):
	
	def __init__(self, module, *args, **kwargs):
		Model.__init__(self, *args, **kwargs)
		self.module = module
	
	def forward(self, x):
		x = self.module(x)
		return x
	
	def state_dict(self):
		return torch.nn.Module.state_dict(self.module)
	
	def load_state_dict(self, state_dict):
		self.module.load_state_dict(state_dict)
