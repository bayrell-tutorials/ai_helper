# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT
##

import io, os, random, math, sqlite3, json

from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from .Utils import alphanum_sort, list_dirs, list_files


class FolderDatabase:
	
	def __init__(self):
		
		self.folder_path = ""
		self.db_con = ""
		self.answers = {}
		self.records = {}
		self.layers = []
	
	
	
	def set_folder(self, path):
		
		"""
		Setup files folder path
		"""
		
		self.folder_path = path
	
	
	
	def get_folder(self):
		
		"""
		Returns folder path
		"""
		
		return self.folder_path
	
	
	
	def get_db_path(self):
		
		"""
		Returns database path
		"""
		
		return os.path.join(self.folder_path, "main.db")
	
	
	
	def get_data_path(self):
		
		"""
		Returns folder data path
		"""
		
		return os.path.join(self.folder_path, "data")
	
	
	
	def clear_data(self):
		
		"""
		Clear data
		"""
		
		self.answers = {}
		self.records = {}
		self.layers = []
	
	
	
	def create_db(self):
		
		"""
		Create database
		"""
		
		cur = self.db_con.cursor()
		
		sql = """CREATE TABLE dataset(
			id integer NOT NULL PRIMARY KEY AUTOINCREMENT,
			layer integer NOT NULL,
			type text NOT NULL,
			file_name text NOT NULL,
			file_index text NOT NULL,
			answer text NOT NULL,
			predict text NOT NULL,
			width integer NOT NULL,
			height integer NOT NULL,
			info text NOT NULL
		)"""
		cur.execute(sql)
		
		sql = """CREATE TABLE answers(
			id integer NOT NULL PRIMARY KEY AUTOINCREMENT,
			layer integer NOT NULL,
			answer text NOT NULL
		)"""
		cur.execute(sql)
		
		sql = """CREATE TABLE layers(
			layer integer NOT NULL,
			PRIMARY KEY ("layer")
		)"""
		cur.execute(sql)
		cur.close()
		
		self.clear_data()
	
	
	
	def create(self):
		
		"""
		Create dataset
		"""
		
		db_path = self.get_db_path()
		
		if os.path.exists(db_path):
			os.unlink(db_path)
		
		if os.path.exists(db_path + "-shm"):
			os.unlink(db_path + "-shm")
		
		if os.path.exists(db_path + "-wal"):
			os.unlink(db_path + "-wal")
		
		self.open(read_db=False)
		self.create_db()
	
	
	
	def open(self, read_db=True):
		
		"""
		Open dataset
		"""
		
		db_path = self.get_db_path()
		
		self.db_con = sqlite3.connect( db_path )
		self.db_con.row_factory = sqlite3.Row
		
		cur = self.db_con.cursor()
		res = cur.execute("PRAGMA journal_mode = WAL;")
		cur.close()
		
		if read_db:
			self.read_database()
	
	
	
	def read_database(self, path):
		
		"""
		Read database
		"""
		
		pass
	
	
	
	def get_answer(self, answer="", layer=0):
		
		"""
		Find answer
		"""
		
		sql = """
			select * from "answers"
			where layer=:layer and answer=:answer
		"""
		
		cur = self.db_con.cursor()
		res = cur.execute(sql, {"answer": answer, "layer": layer})
		record = res.fetchone();
		cur.close()
		
		return record
	
	
	
	def add_layer(self, layer=0):
		
		"""
		Add layer
		"""
		
		if (layer in self.layers):
			return
		
		try:
			sql = """
				INSERT INTO 'layers'
				('layer')
				VALUES
				(:layer)
			"""
			
			cur = self.db_con.cursor()
			res = cur.execute(sql, {
				"layer": layer,
			})
			cur.close()
		
		except Exception:
			pass
	
		
		self.layers.append(layer)
	
	
	
	def add_answer(self, answer="", layer=0):
		
		"""
		Add answer
		"""
		
		if (layer in self.answers) and (answer in self.answers[layer]):
			return
		
		record = self.get_answer(answer)
		if record is not None:
			return
			
		sql = """
			INSERT INTO 'answers'
			('layer', 'answer')
			VALUES
			(:layer, :answer)
		"""
		
		cur = self.db_con.cursor()
		res = cur.execute(sql, {
			"layer": layer,
			"answer": answer,
		})
		cur.close()
		
		if not (layer in self.answers):
			self.add_layer(layer)
			self.answers[layer] = []
		
		self.answers[layer].append(answer)
		
	
	
	def get_record_by_index(self, index, layer=0):
		
		"""
		Returns record by index
		"""
		
		if not (layer in self.records):
			return None
		
		if not (index in self.records[layer]):
			return None
		
		return self.records[layer][index]
		
	
	
	def get_record_by_file_name(self, file_name="", layer=0):
		
		"""
		Find record by name
		"""
		
		sql = """
			select * from dataset
			where layer=:layer and file_name=:file_name
		"""
		
		cur = self.db_con.cursor()
		res = cur.execute(sql, {"file_name": file_name, "layer": layer})
		record = res.fetchone();
		cur.close()
		
		return record
	
	
	
	def add_record(self, type="", file_name="", file_index="",
		answer="", width=-1, height=-1, layer=-1, info=None, record=None
	):
		
		"""
		Add record
		"""
		
		find_record = self.get_record_by_file_name(file_name)
		if find_record is None:
			
			if record is None:
				record = {
					"layer": 0,
					"type": "",
					"file_name": "",
					"file_index": "",
					"answer": "",
					"predict": "",
					"width": 0,
					"height": 0,
					"info": "{}",
				}
			
			if layer >= 0:
				record["layer"] = layer
			
			if type != "":
				record["type"] = type
			
			if file_name != "":
				record["file_name"] = file_name
			
			if file_index != "":
				record["file_index"] = file_index
			
			if answer != "":
				record["answer"] = answer
			
			if width >= 0:
				record["width"] = width
			
			if height >= 0:
				record["height"] = height
			
			if info is not None:
				if isinstance(info, str):
					record["info"] = info
				if isinstance(info, dict):
					record["info"] = json.dumps(info)
			
			
			sql = """
				INSERT INTO 'dataset'
				('layer', 'type', 'file_name', 'file_index',
					'answer', 'predict', 'width', 'height', 'info')
				VALUES
				(:layer, :type, :file_name, :file_index,
					:answer, :predict, :width, :height, :info)
			"""
			
			cur = self.db_con.cursor()
			res = cur.execute(sql, record)
			cur.close()
			
			layer = record["layer"]
			
			if not (layer in self.records):
				self.add_layer(layer)
				self.records[layer] = []
			
			self.records[layer].append(record)
		
		pass
	
	
	
	def flush(self):
		
		"""
		Flush database
		"""
		
		self.db_con.commit()
	
	
	
	def add_tensor(self, type="", file_name="", file_index="",
		answer="", width=-1, height=-1, layer=-1, info=None, record=None
	):
		
		"""
		Add tensor
		"""
		
		pass
	
	
	
	def read_tensor(self, index, layer=0):
		
		"""
		Read tensor
		"""
		
		return None
	
	
	
	def get_layer_count(self, layer):
		
		"""
		Returns layer count
		"""
		
		if not (layer in self.records):
			return None
		
		return len(self.records[layer])



class FolderDataset(Dataset):
	
	def __init__(self, database=None, layer=0):
		
		self.database = database
		self.layer = layer
	
	
	def __getitem__(self, index):
		data = self.database.read_tensor(index, layer=self.layer)
		return ( data[0], data[1] )
	
	
	def __len__(self):
		return self.database.get_layer_count(self.layer)



def init_folder_database(type, path, layer=0):
	
	"""
	Init database folder
	"""
	
	db = FolderDatabase()
	db.set_folder(path)
	db.create()
	
	if type == "answer/images":
	
		dir_name_pos = 1
		dataset_names = list_dirs( os.path.join(path, "data") )
		alphanum_sort(dataset_names)
		
		for dir_name in dataset_names:
			
			file_names = list_files( os.path.join(path, "data", dir_name) )
			alphanum_sort(file_names)
			
			print (str(dir_name_pos) + ") " + dir_name)
			
			iterator_pos = 0
			iterator_count = len(file_names)
			
			db.add_answer(dir_name)
			
			for file_name in file_names:
				
				file_path = os.path.join(path, "data", dir_name, file_name)
				
				im = Image.open(file_path)
				width, height = im.size
				del im
				
				db.add_record(
					type="image",
					file_name=dir_name + "/" + file_name,
					file_index=dir_name + "/" + file_name,
					answer=dir_name,
					width=width,
					height=height,
				)
				
				msg = str(round(iterator_pos / iterator_count * 100))
				iterator_pos = iterator_pos + 1
				print ("\r", end='')
				print (msg + "%", end='')
			
			dir_name_pos = dir_name_pos + 1
			print ("\r", end='')
			
			db.flush()



def convert_folder_database(src_path, dest_path,
	convert=None, type="", train_k=0.95
):
	
	"""
	Convert database folder
	"""
	
	src = FolderDatabase()
	src.set_folder(src_path)
	src.open()
	
	dest = FolderDatabase()
	dest.set_folder(dest_path)
	dest.create()
	
	layer_count = src.get_layer_count(0)
	for index in range(layer_count):
		
		record = src.get_record_by_index(index)
		
		kind = ""
		if type == "train_test":
			rand = random.random()
			if rand > train_k:
				kind = "train"
			else:
				kind = "test"
		
		if convert is not None:
			convert(
				record=record,
				src=src,
				dest=dest,
				kind=kind,
			)

