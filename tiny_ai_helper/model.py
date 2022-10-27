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


class Model:
	
	
	def __init__(self, *args, **kwargs):
		
		self.module = None
		self.history = TrainHistory()
		
		self.input_shape = kwargs["input_shape"] if "input_shape" in kwargs else (1)
		self.output_shape = kwargs["output_shape"] if "output_shape" in kwargs else (1)
		
		self.onnx_path = kwargs["onnx_path"] \
			if "onnx_path" in kwargs else os.path.join(os.getcwd(), "web")
		self.onnx_opset_version = kwargs["onnx_opset_version"] \
			if "onnx_opset_version" in kwargs else 9
		self.model_name = kwargs["model_name"] if "model_name" in kwargs else ""
		
		self.model_path = kwargs["model_path"] \
			if "model_path" in kwargs else os.path.join(os.getcwd(), "data", "model", self.model_name)
		
		self._is_debug = kwargs["debug"] if "debug" in kwargs else False
		self._convert_batch = kwargs["convert_batch"] if "convert_batch" in kwargs else None
		
		self._layers = kwargs["layers"] if "layers" in kwargs else None
		self.create_model()
		
		
	def is_debug(self, value):
		
		"""
		Set debug level
		"""
		
		self._is_debug = value
		
	
	def get_model_name(self):
		
		"""
		Returns model name
		"""
		
		return self.model_name
	
	
	def get_onnx_path(self):
		
		"""
		Returns model onnx path
		"""
		
		return os.path.join(self.onnx_path, self.model_name + ".onnx")
	
	
	def get_model_path(self):
		
		"""
		Returns model folder path
		"""
		
		return self.model_path
	
	
	def set_model_path(self, path):
		
		"""
		Set model folder path
		"""
		
		self.model_path = path
	
	
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
		
		print ("Model name: " + self.model_name)
		
		if tensor_device is None:
			tensor_device = get_tensor_device()
		
		module = self.module.to(tensor_device)
		summary(self.module, tuple(self.input_shape), device=str(tensor_device))
		
		if (verbose and isinstance(self.module, ExtendModule)):
			for arr in self.module._shapes:
				print ( arr[0] + " => " + str(tuple(arr[1])) )
	
	
	def save(self, optimizer=None, save_epoch=False):
		
		"""
		Save model to file
		"""
		
		save_model(self, save_epoch=save_epoch)
	
	
	def save_optimizer(self, optimizer):
		
		"""
		Save optimizer to file
		"""
		
		save_optimizer_file(self, optimizer)
	
	
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
		model = self.module.to(tensor_device)
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
	
	
	def load(self, folder_path=None, repository_path=None, epoch_number=None):
		
		"""
		Load model from file
		"""
		
		if folder_path is not None:
			self.set_model_path(folder_path)
		
		if repository_path is not None:
			self.set_model_path( os.path.join(repository_path, self.model_name) )
		
		load_model(self, epoch_number=epoch_number)
	
	
	def load_optimizer(self, optimizer):
		
		"""
		Load optimizer
		"""
		
		load_optimizer(self, optimizer)
	
	
	def check_answer(self, **kwargs):
		
		"""
		Check answer
		"""
		
		tensor_y = kwargs["tensor_y"]
		tensor_predict = kwargs["tensor_predict"]
		
		y = get_answer_from_vector(tensor_y)
		predict = get_answer_from_vector(tensor_predict)
		
		return predict == y
		
		
	def check_answer_batch(self, **kwargs):
		
		"""
		Check batch. Returns count right answers
		"""
		
		res = 0
		
		type = kwargs["type"]
		batch_x = kwargs["batch_x"]
		batch_y = kwargs["batch_y"]
		batch_predict = kwargs["batch_predict"]
		
		for i in range(batch_x.shape[0]):
			
			tensor_x = batch_x[i]
			tensor_y = batch_y[i]
			tensor_predict = batch_predict[i]
			
			flag = self.check_answer(
				tensor_x=tensor_x,
				tensor_y=tensor_y,
				tensor_predict=tensor_predict,
				type=type,
			)
			
			if flag:
				res = res + 1
		
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
		
		plt = self.history.get_plot()
		history_image = os.path.join( self.get_model_path(), "model.png" )
		make_parent_dir(history_image)		
		plt.savefig(history_image)
		
		if show:
			plt.show()
		
		return plt
	
	
class ExtendModule(torch.nn.Module):
	
	def __init__(self, model):
		
		"""
		Constructor
		"""
		
		super(ExtendModule, self).__init__()
		
		self._model = model
		self._layers = []
		self._shapes = []
		self._saves = {}
		
	
	def forward(self, x):
		
		"""
		Forward model
		"""
		
		for index, obj in enumerate(self._layers):
			
			if isinstance(obj, AbstractLayerFactory):
				
				layer_factory: AbstractLayerFactory = obj
				x = layer_factory.forward(x)
				
				
		return x
	
	
	def init_layers(self, layers=[], debug=False):
		
		"""
		Init layers
		"""
		
		self._layers = layers
		
		input_shape = self._model.input_shape
		output_shape = self._model.output_shape
		
		arr = list(input_shape)
		arr.insert(0, 1)
		
		vector_x = torch.zeros( tuple(arr) )
		self._shapes.append( ("Input", vector_x.shape) )
		
		if debug:
			print ("Debug model " + str(self._model.model_name))
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


def open_model_db(db_path):
		
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
		create_model_db(db_con)
		
	return db_con


def create_model_db(db_con):
	
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
	
	
def save_train_status(model):
		
	"""
	Save train status
	"""
	
	epoch_number = model.history.epoch_number
	epoch_record = model.history.get_epoch(epoch_number)
	
	if epoch_number > 0 and epoch_record is not None:
		
		db_con = open_model_db(
			db_path = os.path.join( model.get_model_path(), "model.db" )
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
		
		history = model.history
		obj = {
			"model_name": model.model_name,
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


def load_train_status(model, epoch_number=None):
		
	"""
	Load train status
	"""
	
	db_con = open_model_db(
		db_path = os.path.join( model.get_model_path(), "model.db" )
	)
	
	sql = """
		select * from "history"
		where model_name=:model_name
		order by epoch_number asc
	"""
	
	cur = db_con.cursor()
	res = cur.execute(sql, {"model_name": model.model_name})
	
	records = res.fetchall()
	
	for record in records:
		
		if epoch_number is not None:
			if record["epoch_number"] > epoch_number:
				continue
		
		model.history.add( record )
	
	cur.close()
	
	db_con.commit()
	db_con.close()


def save_model_file(model, epoch_number=None):
		
	"""
	Save model to file
	"""
	
	file_path = os.path.join( model.get_model_path(), "model.data" )
	if epoch_number is not None:
		file_path = os.path.join( model.get_model_path(), "model-" + str(epoch_number) + ".data" )
	
	make_parent_dir(file_path)
	torch.save(model.module.state_dict(), file_path)


def save_optimizer_file(model, optimizer):
	
	epoch_number = model.history.epoch_number
	file_path = os.path.join( model.get_model_path(), "model-" +
		str(epoch_number) + "-optimizer.data" )
	
	make_parent_dir(file_path)
	torch.save(optimizer.state_dict(), file_path)


def save_model(model, save_epoch=False):
		
	"""
	Save model
	"""
	
	save_model_file(model)
	save_train_status(model)
	
	if save_epoch:
		save_model_file(model, model.history.epoch_number)


def load_model(model, epoch_number=None):
		
	"""
	Load model
	"""
	
	state_dict = None
	
	file_path = os.path.join( model.get_model_path(), "model.data" )
	if epoch_number is not None:
		file_path = os.path.join( model.get_model_path(), "model-" + str(epoch_number) + ".data" )
	
	try:
		if os.path.isfile(file_path):
			state_dict = torch.load(file_path)
	
	except:
		pass
	
	if state_dict:
		load_train_status(model, epoch_number)
		model.module.load_state_dict(state_dict)
		return True
	
	return False


def load_optimizer(model, optimizer):
	
	epoch_number = model.history.epoch_number
	file_path = os.path.join( model.get_model_path(), "model-" +
		str(epoch_number) + "-optimizer.data" )
	
	state_dict = None
	if os.path.isfile(file_path):
		state_dict = torch.load(file_path)
	
	if state_dict:
		optimizer.load_state_dict(state_dict)
