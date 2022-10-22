# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT
##


import os, math

import numpy as np
import matplotlib.pyplot as plt

from .AbstractModel import AbstractModel


class KerasModel(AbstractModel):
	
	def __init__(self):
		
		AbstractModel.__init__(self)
		
		self._is_new = True
		self.model = None
		self.history = None
		self.batch_size = 64
		self.epochs = 30
		self.train_verbose = 1
		
		
	def is_loaded(self):
		
		"""
			Returns true if model is loaded
		"""
		
		return self.model != None
	
	
	
	def create(self):
		
		"""
			Create model
		"""
		
		from tensorflow.keras.models import Sequential
		from tensorflow.keras.layers import Dense, Input, Dropout
		
		model_name = self.get_model_name()
		self.model = Sequential(name=model_name)
		
		# Input layer
		self.model.add(Input(self.input_shape, name='input'))
		
		# Hidden layer
		self.model.add(Dense(256, activation='relu'))
		self.model.add(Dropout(0.5))
		
		# Output layer
		self.model.add(Dense(self.output_shape[0], name='output', activation='softmax'))
		
		# Compile model
		self.model.compile(
			loss='mean_squared_error', 
			optimizer='adam',
			metrics=['accuracy'])
		
		self._is_new = True
		
		
	def create_model_parent_dir(self):
		
		"""
			Create folder for model
		"""
		
		# Create model folder
		model_path = self.get_model_path()
		model_dir = os.path.dirname(model_path)
		
		if not os.path.isdir(model_dir):
			os.makedirs(model_dir)
		
		
	def load(self):
		
		"""
			Load model from file
		"""
		
		import tensorflow.keras as keras
		
		model_path = self.get_model_path()
		model_file_path = model_path + '.h5'
		
		self.model = None
		if os.path.isfile(model_file_path):
			self.model = keras.models.load_model(model_file_path)
			
		if self.model:
			self._is_new = False
		
		
	def show_summary(self):
		
		"""
			Show model info
		"""
		
		import tensorflow.keras as keras
		
		self.create_model_parent_dir()
		model_path = self.get_model_path()
		
		# Вывод на экран информация о модели
		self.model.summary()
		
		file_name = model_path + "_plot.png"
		keras.utils.plot_model(
			self.model,
			to_file=file_name,
			show_shapes=True)
		
		
	def train(self):
		
		"""
			Train model
		"""
		
		import tensorflow.keras as keras
		
		model_path = self.get_model_path()
		checkpoint_path = os.path.join(model_path, "training", "cp.ckpt")
		checkpoint_dir = os.path.dirname(checkpoint_path)
		
		# Create folder for checkpoints
		if not os.path.isdir(checkpoint_dir):
			os.makedirs(checkpoint_dir)
		
		# Callback function
		cp_callback = keras.callbacks.ModelCheckpoint(
			filepath=checkpoint_path,
			save_weights_only=True,
			verbose=1
		)
		
		self.dataset.build_train()
		
		self.history = self.model.fit(
		
			# Input dataset
			self.dataset.train_x,
			
			# Output dataset
			self.dataset.train_y,
			
			# Batch size
			batch_size=self.batch_size,
			
			# Count epochs
			epochs=self.epochs,
			
			# Test data
			validation_data=(self.dataset.test_x, self.dataset.test_y),
			
			# Verbose
			verbose=self.train_verbose,
			
			# Сохраняем контрольные точки
			callbacks=[cp_callback]
		) 
		
		# Save model to file
		self.model.save(model_path)
		self.model.save(model_path + ".h5")
		
		pass
	
	
	def show_train_info(self):
		
		"""
			Show train info
		"""
		
		model_path = self.get_model_path()
		total_val_accuracy = math.ceil(self.history.history['val_accuracy'][-1] * 100)
	
		# Save picture
		plt.title("Total: " + str(total_val_accuracy) + "%")
		plt.plot( np.multiply(self.history.history['accuracy'], 100), label='acc')
		plt.plot( np.multiply(self.history.history['val_accuracy'], 100), label='val_acc')
		plt.plot( np.multiply(self.history.history['val_loss'], 100), label='val_loss')
		plt.ylabel('Percent')
		plt.xlabel('Epochs')
		plt.legend()
		plt.savefig(model_path + '_history.png')
		plt.show()
		
	
	def predict(self, vector_x):
		
		"""
			Predict model
		"""
		
		vector_answer = self.model.predict( vector_x )
		return vector_answer
		