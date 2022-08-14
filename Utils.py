# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT
##

import math, io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


def sign(x):
	"""
		Sign function
	"""
	if x >= 0: return 1
	return -1
	
	
def indexOf(arr, item):
	"""
		Index of
	"""
	try:
		index = arr.index(item)
		return index
	except Exception:
		pass
	return -1
	
	
def vector_append(res, data):
	
	"""
		Append 2 numpy vectors
	"""
	
	if res is None:
		res = np.expand_dims(data, axis=0)
	else:
		res = np.append(res, [data], axis=0)
	
	return res
	
	
def tensorflow_gpu_init(memory_limit=1024):
	"""
		Init tensorflow GPU
	"""
	import tensorflow as tf
	gpus = tf.config.list_physical_devices('GPU')
	tf.config.experimental.set_memory_growth(gpus[0], True)
	tf.config.experimental.set_virtual_device_configuration(
	    gpus[0],
	    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])



def image_resize_canvas(image, size, color=None):
	
	"""
		Resize image canvas
	"""
	
	width, height = size
	
	if color == None:
		pixels = image.load()
		color = pixels[0, 0]
		del pixels
		
	image_new = Image.new(image.mode, (width, height), color = color)
	draw = ImageDraw.Draw(image_new)
	
	position = (
		math.ceil((width - image.size[0]) / 2),
		math.ceil((height - image.size[1]) / 2),
	)
	
	image_new.paste(image, position)
	
	del draw, image
	
	return image_new
	
	
def image_to_vector(image_bytes, mode=None):
	
	"""
		Convert image to numpy vector
	"""
	
	image = None
	
	try:
		
		if isinstance(image_bytes, bytes):
			image = Image.open(io.BytesIO(image_bytes))
		
		if isinstance(image_bytes, Image.Image):
			image = image_bytes
	
	except Exception:
		image = None
	
	if image is None:
		return None
	
	if mode is not None:
		image = image.convert(mode)
	
	image_vector = np.asarray(image)

	return image_vector
	
	
def plot_show_image(image):
	"""
		Plot show image
	"""
	plt.imshow(image, cmap='gray')
	plt.show()
	
	
def get_answer_vector_by_number(number, count):
	"""
		Returns anwer vector
	"""
	res = [0.0] * count
	if (number >=0 and number < count):
		res[number] = 1.0
	return np.asarray(res)


def get_answer_from_vector(vector):
	"""
		Returns answer from vector
	"""
	value_max = -math.inf
	value_index = 0
	for i in range(0, len(vector)):
		value = vector[i]
		if value_max < value:
			value_index = i
			value_max = value
	
	return value_index