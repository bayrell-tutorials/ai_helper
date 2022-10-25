# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT
##

import io, os, random, math, sqlite3, json, torch
from .Utils import make_parent_dir


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
	
	sql = """CREATE TABLE models(
		model_name text NOT NULL PRIMARY KEY,
		epoch_number integer NOT NULL,
		acc_train real NOT NULL,
		acc_test real NOT NULL,
		acc_rel real NOT NULL,
		loss_train real NOT NULL,
		loss_test real NOT NULL,
		info text NOT NULL
	)"""
	cur.execute(sql)
	
	sql = """CREATE TABLE history(
		model_name text NOT NULL,
		epoch_number integer NOT NULL,
		acc_train real NOT NULL,
		acc_test real NOT NULL,
		acc_rel real NOT NULL,
		loss_train real NOT NULL,
		loss_test real NOT NULL,
		acc_train_iter real NOT NULL,
		acc_test_iter real NOT NULL,
		loss_train_iter real NOT NULL,
		loss_test_iter real NOT NULL,
		batch_train_iter integer NOT NULL,
		batch_test_iter integer NOT NULL,
		train_count integer NOT NULL,
		test_count integer NOT NULL,
		time real NOT NULL,
		info text NOT NULL,
		PRIMARY KEY ("model_name", "epoch_number")
	)"""
	cur.execute(sql)
	
	cur.close()
	


class ModelDatabase:
	
	
	def __init__(self, folder_path=None):
		
		self.folder_path = folder_path
	
	
	def set_path(self, folder_path):
		
		"""
		Set folder path
		"""
		
		self.folder_path = folder_path
	
	
	def get_model_path(self, model_name, epoch_number=None):
		
		"""
		Returns model path
		"""
		
		file_name = os.path.join(self.folder_path, model_name, "model.data")
		
		if epoch_number is not None:
			file_name = os.path.join(self.folder_path,
				model_name, "model-" + str(epoch_number) + ".data"
			)
		
		return file_name
	
	
	def save_train_status(self, model_name, train_status):
		
		"""
		Save train status
		"""
		
		db_con = open_model_db(
			db_path = os.path.join( self.folder_path, "model.db" )
		)
		
		
		"""
		Save model info
		"""
		
		sql = """
			insert or replace into models (
				model_name, epoch_number, acc_train,
				acc_test, acc_rel, loss_train, loss_test,
				info
			) values
			(
				:model_name, :epoch_number, :acc_train,
				:acc_test, :acc_rel, :loss_train, :loss_test,
				:info
			)
		"""
		
		cur = db_con.cursor()
		res = cur.execute(sql, {
			"model_name": model_name,
			"epoch_number": train_status.epoch_number,
			"acc_train": train_status.get_acc_train(),
			"acc_test": train_status.get_acc_test(),
			"acc_rel": train_status.get_acc_rel(),
			"loss_train": train_status.get_loss_train(),
			"loss_test": train_status.get_loss_test(),
			"info": "{}",
		})
		cur.close()
		
		
		"""
		Save train status info
		"""
		
		if train_status.epoch_number > 0:
		
			sql = """
				insert or replace into history (
					model_name, epoch_number, acc_train,
					acc_test, acc_rel, loss_train, loss_test,
					acc_train_iter, acc_test_iter,
					loss_train_iter, loss_test_iter,
					batch_train_iter, batch_test_iter,
					train_count, test_count, time,
					info
				) values
				(
					:model_name, :epoch_number, :acc_train,
					:acc_test, :acc_rel, :loss_train, :loss_test,
					:acc_train_iter, :acc_test_iter,
					:loss_train_iter, :loss_test_iter,
					:batch_train_iter, :batch_test_iter,
					:train_count, :test_count, :time,
					:info
				)
			"""
			
			cur = db_con.cursor()
			res = cur.execute(sql, {
				"model_name": model_name,
				"epoch_number": train_status.epoch_number,
				"acc_train": train_status.get_acc_train(),
				"acc_test": train_status.get_acc_test(),
				"acc_rel": train_status.get_acc_rel(),
				"loss_train": train_status.get_loss_train(),
				"loss_test": train_status.get_loss_test(),
				"acc_train_iter": train_status.acc_train_iter,
				"acc_test_iter": train_status.acc_test_iter,
				"loss_train_iter": train_status.loss_train_iter,
				"loss_test_iter": train_status.loss_test_iter,
				"batch_train_iter": train_status.batch_train_iter,
				"batch_test_iter": train_status.batch_test_iter,
				"train_count": train_status.train_count_iter,
				"test_count": train_status.test_count_iter,
				"time": train_status.get_time(),
				"info": "{}",
			})
			cur.close()
		
		db_con.commit()
		db_con.close()
		
	
	
	def load_train_status(self, model_name, train_status, epoch_number=None):
		
		"""
		Load train status
		"""
		
		train_status.clear()
		
		db_con = open_model_db(
			db_path = os.path.join( self.folder_path, "model.db" )
		)
		
		sql = """
			select * from "history"
			where model_name=:model_name
			order by epoch_number asc
		"""
		
		cur = db_con.cursor()
		res = cur.execute(sql, {"model_name": model_name})
		
		records = res.fetchall()
		
		for record in records:
			
			if epoch_number is not None:
				if record["epoch_number"] > epoch_number:
					continue
			
			train_status.epoch_number = record["epoch_number"]
			train_status.batch_train_iter = record["batch_train_iter"]
			train_status.batch_test_iter = record["batch_test_iter"]
			train_status.train_count_iter = record["train_count"]
			train_status.test_count_iter = record["test_count"]
			train_status.loss_train_iter = record["loss_train_iter"]
			train_status.loss_test_iter = record["loss_test_iter"]
			train_status.acc_train_iter = record["acc_train_iter"]
			train_status.acc_test_iter = record["acc_test_iter"]
			
			loss_train = train_status.get_loss_train()
			loss_test = train_status.get_loss_test()
			acc_train = train_status.get_acc_train()
			acc_test = train_status.get_acc_test()
			acc_rel = train_status.get_acc_rel()
			train_status.history["loss_train"].append(loss_train)
			train_status.history["loss_test"].append(loss_test)
			train_status.history["acc_train"].append(acc_train)
			train_status.history["acc_test"].append(acc_test)
			train_status.history["acc_rel"].append(acc_rel)
		
		cur.close()
		
		db_con.commit()
		db_con.close()
	
	
	def save_file(self, model_name, module, epoch_number=None):
		
		"""
		Save model to file
		"""
		
		file_name = self.get_model_path(model_name, epoch_number)
		
		if module:
			make_parent_dir(file_name)
			torch.save(module.state_dict(), file_name)
	
	
	def save(self, model_name, module, train_status):
		
		"""
		Save model
		"""
		
		self.save_file(model_name, module)
		self.save_file(model_name, module, train_status.epoch_number)
		self.save_train_status(model_name, train_status)
		
	
	def load(self, model_name, module, train_status, epoch_number=None):
		
		"""
		Load model
		"""
		
		state_dict = None
		file_name = self.get_model_path(model_name, epoch_number)
		
		try:
			if os.path.isfile(file_name):
				state_dict = torch.load(file_name)
		
		except:
			pass
		
		if state_dict:
			self.load_train_status(model_name, train_status, epoch_number)
			module.load_state_dict(state_dict)
			return True
		
		return False
	