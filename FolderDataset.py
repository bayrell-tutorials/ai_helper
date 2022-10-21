# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT
##

import io, os, random, math, sqlite3, json


class FolderDataset:
	
	def __init__(self):
		
		self.db_path = ""
		self.db_con = ""
		self.answers = {}
		
	
	def create_db(self):
		
		"""
		Create database
		"""
		
		cur = self.db_con.cursor()
		
		sql = """CREATE TABLE dataset(
			id integer NOT NULL PRIMARY KEY AUTOINCREMENT,
			db integer NOT NULL,
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
			db integer NOT NULL,
			answer text NOT NULL
		)"""
		cur.execute(sql)
		cur.close()
	
	
	def create(self, path):
		
		"""
		Create dataset
		"""
		
		if os.path.exists(path):
			os.unlink(path)
		
		if os.path.exists(path + "-shm"):
			os.unlink(path + "-shm")
		
		if os.path.exists(path + "-wal"):
			os.unlink(path + "-wal")
		
		self.open(path)
		self.create_db()
	
	
	def open(self, path):
		
		"""
		Open dataset
		"""
		
		self.db_path = path
		self.db_con = sqlite3.connect(path)
		self.db_con.row_factory = sqlite3.Row
		
		cur = self.db_con.cursor()
		res = cur.execute("PRAGMA journal_mode = WAL;")
		cur.close()
		
	
	
	def getAnswer(self, answer="", db=0):
		
		"""
		Find answer
		"""
		
		sql = """
			select * from "answers"
			where db=:db and answer=:answer
		"""
		
		cur = self.db_con.cursor()
		res = cur.execute(sql, {"answer": answer, "db": db})
		record = res.fetchone();
		cur.close()
		
		return record
	
	
	
	def addAnswer(self, answer="", db=0):
		
		"""
		Add answer
		"""
		
		if (db in self.answers) and (answer in self.answers[db]):
			return
		
		record = self.getAnswer(answer)
		if record is not None:
			return
			
		sql = """
			INSERT INTO 'answers'
			('db', 'answer')
			VALUES
			(:db, :answer)
		"""
		
		cur = self.db_con.cursor()
		res = cur.execute(sql, {
			"db": db,
			"answer": answer,
		})
		cur.close()
		
		if not (db in self.answers):
			self.answers[db] = []
		
		self.answers[db].append(answer)
		
	
	def getRecord(self, file_name="", db=0):
		
		"""
		Find record
		"""
		
		sql = """
			select * from dataset
			where db=:db and file_name=:file_name
		"""
		
		cur = self.db_con.cursor()
		res = cur.execute(sql, {"file_name": file_name, "db": db})
		record = res.fetchone();
		cur.close()
		
		return record
	
	
	def addRecord(self, type="", file_name="", file_index="",
		answer="", width=0, height=0, db=0, info={}):
		
		"""
		Add record
		"""
		
		record = self.getRecord(file_name)
		if record is None:
			
			sql = """
				INSERT INTO 'dataset'
				('db', 'type', 'file_name', 'file_index',
				'answer', 'predict', 'width', 'height', 'info')
				VALUES
				(:db, :type, :file_name, :file_index, :answer, :predict, :width, :height, :info)
			"""
			
			cur = self.db_con.cursor()
			res = cur.execute(sql, {
				"db": db,
				"type": type,
				"file_name": file_name,
				"file_index": file_index,
				"answer": answer,
				"predict": "",
				"width": width,
				"height": height,
				"db": db,
				"info": json.dumps(info),
			})
			cur.close()
			
			pass
		
		pass
	
	
	
	def flush(self):
		
		self.db_con.commit()
		