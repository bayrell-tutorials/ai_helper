# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT
##

from .TrainStatus import TrainStatus


class TrainVerboseCallback:
	
	
	def on_end_batch_train(self, train_status:TrainStatus):
		
		if train_status.model.verbose:
			
			acc_train = train_status.get_acc_train()
			loss_train = train_status.get_loss_train()
			time = train_status.get_time()
			
			msg = ("\rStep {epoch_number}, {iter_value}%" +
				", acc: .{acc}, loss: .{loss}, time: {time}s"
			).format(
				epoch_number = train_status.epoch_number,
				iter_value = round(train_status.get_iter_value() * 100),
				loss = str(round(loss_train * 10000)).zfill(4),
				acc = str(round(acc_train * 100)).zfill(2),
				time = str(round(time)),
			)
			
			print (msg, end='')
	
	
	def on_end_epoch(self, train_status:TrainStatus):
		
		"""
		Epoch
		"""
		
		if train_status.model.verbose:
			
			loss_train = train_status.get_loss_train()
			loss_test = train_status.get_loss_test()
			acc_train = train_status.get_acc_train()
			acc_test = train_status.get_acc_test()
			acc_rel = train_status.get_acc_rel()
			time = train_status.get_time()
			
			print ("\r", end='')
			
			msg = ("Step {epoch_number}, " +
				"acc: .{acc_train}, " +
				"acc_test: .{acc_test}, " +
				"acc_rel: {acc_rel}, " +
				"loss: .{loss_train}, " +
				"loss_test: .{loss_test}, " +
				"time: {time}s, "
			).format(
				epoch_number = train_status.epoch_number,
				loss_train = str(round(loss_train * 10000)).zfill(4),
				loss_test = str(round(loss_test * 10000)).zfill(4),
				acc_train = str(round(acc_train * 100)).zfill(2),
				acc_test = str(round(acc_test * 100)).zfill(2),
				acc_rel = str(round(acc_rel * 100)).zfill(2),
				time = str(round(time)),
			)
			
			print (msg)
	