# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT
##

import io, os, random, math, sqlite3, json, torch

from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from .Utils import *


class FolderDatabase:
	
	def __init__(self, read_tensor=None, get_tensor_from_answer=None):
		
		self.chunk_folder_names = (1, 2)
		self.folder_path = ""
		self.db_con = ""
		self.answers = {}
		self.records = {}
		self.layers = []
		self._get_tensor_from_answer = get_tensor_from_answer
	
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
	
	
	def get_file_path(self, file_name):
		
		"""
		Returns file data path
		"""
		
		return os.path.join(self.folder_path, "data", file_name)
	
	
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
	
	
	
	def create(self, folder_path=None):
		
		"""
		Create dataset
		"""
		
		if folder_path is not None:
			self.set_folder(folder_path)
		
		db_path = self.get_db_path()
		
		make_parent_dir(db_path)
		
		if os.path.exists(db_path):
			os.unlink(db_path)
		
		if os.path.exists(db_path + "-shm"):
			os.unlink(db_path + "-shm")
		
		if os.path.exists(db_path + "-wal"):
			os.unlink(db_path + "-wal")
		
		self.open(is_read=False)
	
	
	def open(self, folder_path=None, is_read=True):
		
		"""
		Open dataset
		"""
		
		if folder_path is not None:
			self.set_folder(folder_path)
		
		is_create = False
		db_path = self.get_db_path()
		
		make_parent_dir(db_path)
		
		if not os.path.exists(db_path):
			is_create = True
		
		self.db_con = sqlite3.connect( db_path, isolation_level=None )
		self.db_con.row_factory = sqlite3.Row
		
		cur = self.db_con.cursor()
		res = cur.execute("PRAGMA journal_mode = WAL;")
		cur.close()
		
		if is_create:
			self.create_db()
		
		if is_read:
			self.read_database()
	
	
	
	def flush(self):
		
		"""
		Flush database
		"""
		
		if self.db_con:
			self.db_con.commit()
	
	
	
	def close(self):
		
		"""
		Close database
		"""
		
		if self.db_con:
			self.db_con.commit()
			self.db_con.close()
		
		self.db_con = None
		
	
	def read_database(self):
		
		"""
		Read database
		"""
		
		sql = """
			select * from "dataset"
			order by id asc
		"""
		
		cur = self.db_con.cursor()
		res = cur.execute(sql)
		batch_size = 1024
		
		while True:
			
			records = res.fetchmany( batch_size )
			if records is None or len(records) == 0:
				break
			
			for record in records:
				self.add_record( dict(record) )
		
			del records
		
		cur.close()
		
		pass
	
	
	
	def find_answer(self, answer="", layer=0):
		
		"""
		Find answer
		"""
		
		sql = """
			select * from "answers"
			where layer=:layer and answer=:answer
		"""
		
		cur = self.db_con.cursor()
		res = cur.execute(sql, {"answer": answer, "layer": layer})
		record = res.fetchone()
		cur.close()
		
		return record
	
	
	
	def add_layer(self, layer):
		
		"""
		Add layer
		"""
		
		if not (layer in self.layers):
			self.layers.append(layer)
	
	
	
	def save_layer(self, layer):
		
		"""
		Save layer
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
	
		
		self.add_layer(layer)
	
	
	
	def add_answer(self, answer="", layer=0):
		
		"""
		Add answer
		"""
		
		if answer == "":
			return
		
		if not (layer in self.answers):
			self.add_layer(layer)
			self.answers[layer] = []
		
		if not (answer in self.answers[layer]):
			self.answers[layer].append(answer)
	
	
	
	def save_answer(self, answer="", layer=0):
		
		"""
		Add answer
		"""
		
		if answer == "":
			return
		
		if (layer in self.answers) and (answer in self.answers[layer]):
			return
		
		record = self.find_answer(answer, layer)
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
		
		self.add_answer(answer, layer)
		
	
	
	def get_record_by_index(self, index, layer=0):
		
		"""
		Returns record by index
		"""
		
		if not (layer in self.records):
			return None
		
		if index < 0 or index >= len(self.records[layer]):
			return None
		
		return self.records[layer][index]
		
		
	
	def find_record_by_id(self, id):
		
		"""
		Returns record by id
		"""
		
		sql = """
			select * from dataset
			where id=:id
		"""
		
		cur = self.db_con.cursor()
		res = cur.execute(sql, {"id": id})
		record = res.fetchone();
		cur.close()
		
		return record
	
	
	
	def find_record_by_file_name(self, file_name="", layer=0):
		
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
	
	
	
	def add_record(self, record):
		
		"""
		Add record
		"""
		
		layer = record["layer"]
		
		self.add_layer(layer)
		self.add_answer( record["answer"] )
		
		if not (layer in self.records):
			self.records[layer] = []
		
		self.records[layer].append(record)
	
	
	
	def save_record(self, type="", file_name="", file_index="",
		answer="", width=-1, height=-1, layer=-1, info=None, record=None
	):
		
		"""
		Add record
		"""
		
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
		
		layer = record["layer"]
		answer = record["answer"]
		file_name = record["file_name"]
		
		find_record = self.find_record_by_file_name(file_name, layer=layer)
		if find_record is None:
			
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
			
			self.save_layer(layer)
			self.save_answer(answer, layer)
			self.add_record(record)
		
		pass
	
	
	
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
	
	
	
	def save_file(self, file_content=None, file_ext="", **kwargs):
		
		"""
		Save file
		"""
		
		if isinstance(file_content, Image.Image):
			file_ext="jpg"
		
		else:
			file_ext="data"
		
		layer = kwargs["layer"] if "layer" in kwargs else 0
		layer_count = self.get_layer_count(layer)
		
		folder_data_path = os.path.join( self.folder_path, "data" )
		file_name = os.path.join(
			self.get_folder_path_by_number(layer_count),
			str(layer_count) + "-" + str(layer) + "." + file_ext
		)
		
		file_path = os.path.join(folder_data_path, file_name);
		
		make_parent_dir(file_path)
		
		if isinstance(file_content, Image.Image):
			file_content.save(file_path, quality=100)
		
		else:
			torch.save(file_content, file_path)
		
		kwargs["file_name"] = file_name
		self.save_record(**kwargs)
		
		pass
	
	
	def read_tensor_by_id(self, id):
		
		"""
		Read tensor by id
		"""
		
		record = self.find_record_by_id(id)
		if record is None:
			return None, None, None
		
		file_path = self.get_file_path(record["file_name"])
		
		_, file_ext = os.path.splitext( file_path )
		
		if file_ext == ".data":
			tensor = torch.load(file_path)
		
		else:
			tensor = convert_image_to_tensor(file_path)
		
		answer = record["answer"]
		
		if self._get_tensor_from_answer:
			answer = self._get_tensor_from_answer(answer)
		
		return (tensor, answer, record)
	
	
	
	def read_tensor(self, index, layer=0):
		
		"""
		Read tensor by index
		"""
		
		record = self.get_record_by_index(index, layer=layer)
		if record is None:
			return None, None, None
		
		file_path = self.get_file_path(record["file_name"])
		
		_, file_ext = os.path.splitext( file_path )
		
		if file_ext == ".data":
			tensor = torch.load(file_path)
		
		else:
			tensor = convert_image_to_tensor(file_path)
		
		answer = record["answer"]
		
		if self._get_tensor_from_answer:
			answer = self._get_tensor_from_answer(answer)
		
		return (tensor, answer, record)
	
	
	
	def get_layer_count(self, layer):
		
		"""
		Returns layer count
		"""
		
		if not (layer in self.records):
			return 0
		
		return len(self.records[layer])



class FolderDataset(Dataset):
	
	def __init__(self, database=None, layer=0):
		
		self.database = database
		self.layer = layer
	
	
	def __getitem__(self, index):
		tensor, answer, _ = self.database.read_tensor(index, layer=self.layer)
		return ( tensor, answer )
	
	
	def __len__(self):
		return self.database.get_layer_count(self.layer)



def init_folder_database(type, path, layer=0, db_class=FolderDatabase):
	
	"""
	Init database folder
	"""
	
	db = db_class()
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
			
			db.save_answer(dir_name, layer)
			
			for file_name in file_names:
				
				file_path = os.path.join(path, "data", dir_name, file_name)
				
				im = Image.open(file_path)
				width, height = im.size
				del im
				
				db.save_record(
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
	
	db.close()


def convert_folder_record(**kwargs):
		
	"""
	Convert folder record
	"""
	
	src = kwargs["src"]
	dest = kwargs["dest"]
	record = kwargs["record"]
	kind = kwargs["kind"] if "kind" in kwargs else ""
	transform = kwargs["transform"] if "transform" in kwargs else None
	transform_train = kwargs["transform_train"] if "transform_train" in kwargs else None
	transform_test = kwargs["transform_test"] if "transform_test" in kwargs else None
	layer = 0
	
	if kind == "train":
		# Train dataset
		transform = transform_train
	
	elif kind == "test":
		# Test dataset
		layer = 1
		transform = transform_test
	
	image = src.get_file_path(record["file_name"])
	
	if transform:
		image = transform(image)
	
	dest.save_file(
		layer=layer,
		file_content=image,
		file_index=record["file_index"],
		record=record,
	)


def convert_folder_database(src_path, dest_path,
	convert=None, type="", train_k=0.95,
	db_class=FolderDatabase, **kwargs
):
	
	"""
	Convert database folder
	"""
	
	src = db_class()
	src.set_folder(src_path)
	src.open()
	
	dest = db_class()
	dest.set_folder(dest_path)
	dest.create()
	
	iterator_pos = 0
	layer_count = src.get_layer_count(0)
	for index in range(layer_count):
		
		record = src.get_record_by_index(index, layer=0)
		if record is None:
			continue
		
		kind = ""
		if type == "train_test":
			rand = random.random()
			if rand > train_k:
				kind = "test"
			else:
				kind = "train"
		
		if convert is not None:
			convert(
				record=record,
				src=src,
				dest=dest,
				kind=kind,
				type=type,
				**kwargs,
			)
			
		else:
			convert_folder_record(
				record=record,
				src=src,
				dest=dest,
				kind=kind,
				type=type,
				**kwargs,
			)
		
		if iterator_pos % 10 == 0:
			
			msg = str(round(iterator_pos / layer_count * 100))
			print ("\r", end='')
			print (msg + "%", end='')
			
			dest.flush()
		
		iterator_pos = iterator_pos + 1
			
	dest.flush()
	print ("")
	
	src.close()
	dest.close()
