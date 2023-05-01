# -*- coding: utf-8 -*-

##
# Tiny ai helper
# Copyright (с) Ildar Bikmamatov 2022 - 2023 <support@bayrell.org>
# License: MIT
##

import torch, time
from .utils import get_default_device, batch_to, get_acc_class, get_acc_binary
from torch.utils.data import DataLoader, TensorDataset


class Trainer:
    
    def __init__(self):
        
        self.device = None
        self.model = None
        self.epoch = 0
        self.loss_train = 1
        self.loss_val = 1
        self.acc_train = 0
        self.acc_val = 0
        self.count_train = 0
        self.count_val = 0
        self.batch_iter = 0
        self.time_start = 0
        self.time_end = 0
        self.train_loader = None
        self.val_loader = None
        self.max_best_models = 5
        self.min_epoch = 5
        self.max_epoch = 10
        self.min_loss_val = -1
        self.min_lr = 1e-5
        self.do_training = False
        
    
    def check_is_trained(self):
        
        """
        Returns True if model is trained
        """
        
        if self.epoch >= self.max_epoch:
            return True
        
        if self.count_val > 0 and \
            (self.loss_val / self.count_val) < self.min_loss_val and \
            self.epoch >= self.min_epoch:
            return True
        
        if self.model.optimizer.param_groups[0]["lr"] < self.min_lr:
            return True
        
        return False
    
    
    def on_start_train(self):
        pass
    
    
    def on_start_epoch(self):
        pass
    
    
    def on_start_batch_train(self, batch_x, batch_y):
        pass
    
    
    def on_end_batch_train(self):
        
        # Лог обучения
        acc_train = str(round(self.acc_train / self.count_train * 100))
        batch_iter_value = round(self.batch_iter / (self.len_train + self.len_val) * 100)
        print (f"\rEpoch {self.epoch}, {batch_iter_value}%, acc: {acc_train}%", end='')
    
    
    def on_start_batch_val(self, batch_x, batch_y):
        pass
    
    
    def on_end_batch_val(self):
        
        # Лог обучения
        acc_train = str(round(self.acc_train / self.count_train * 100))
        batch_iter_value = round(self.batch_iter / (self.len_train + self.len_val) * 100)
        print (f"\rEpoch {self.epoch}, {batch_iter_value}%, acc: {acc_train}%", end='')
    
    
    def on_end_epoch(self):
        
        # Получить текущий lr
        res_lr = []
        for param_group in self.model.optimizer.param_groups:
            res_lr.append(param_group['lr'])
        res_lr_str = str(res_lr)
        
        # Результат обучения
        loss_train = '%.3e' % (self.loss_train / self.batch_train)
        loss_val = '%.3e' % (self.loss_val / self.batch_val)
        
        acc_train = self.acc_train / self.count_train
        acc_val = self.acc_val / self.count_val
        acc_rel = acc_train / acc_val if acc_val > 0 else 0
        acc_train = str(round(acc_train * 10000) / 100)
        acc_val = str(round(acc_val * 10000) / 100)
        acc_train = acc_train.ljust(5, "0")
        acc_val = acc_val.ljust(5, "0")
        acc_rel_str = str(round(acc_rel * 100) / 100).ljust(4, "0")
        time = str(round(self.time_end - self.time_start))
        
        print ("\r", end='')
        print (f"Epoch {self.epoch}, " +
            f"acc: {acc_train}%, acc_val: {acc_val}%, rel: {acc_rel_str}, " +
            f"loss: {loss_train}, loss_val: {loss_val}, lr: {res_lr_str}, " +
            f"t: {time}s"
        )
        
        # Update model history
        self.model.epoch = self.epoch
        self.model.history[self.epoch] = {
            "loss_train": self.loss_train / self.batch_train,
            "loss_val": self.loss_val / self.batch_val,
            "acc_train": self.acc_train / self.count_train,
            "acc_val": self.acc_val / self.count_val,
            "acc_rel": acc_rel,
            "count_train": self.count_train,
            "count_val": self.count_val,
            "batch_iter": self.batch_iter,
            "res_lr": res_lr,
            "time": time,
        }
        
        # Save model
        self.model.save_epoch()
        self.model.save_the_best_models(epoch_count=self.max_best_models)
    
    
    def on_end_train(self):
        pass
    
    
    def stop_training(self):
        self.do_training = False
    
    
    def calc_metrics(self, kind, batch_x, batch_y, batch_predict, loss_value):
        
        """
        Calc metrics
        """
        
        get_acc_fn = self.model.acc_fn
        acc = get_acc_fn(batch_predict, batch_y)
        loss_value_item = loss_value.item()
        batch_count = len(batch_x[0]) if isinstance(batch_x, list) else len(batch_x)
        
        if kind == "train":
            self.acc_train = self.acc_train + acc
            self.loss_train = self.loss_train + loss_value_item
            self.count_train = self.count_train + batch_count
            self.batch_train = self.batch_train + 1
            self.batch_iter = self.batch_iter + batch_count
        
        elif kind == "valid":
            self.acc_val = self.acc_val + acc
            self.loss_val = self.loss_val + loss_value_item
            self.count_val = self.count_val + batch_count
            self.batch_val = self.batch_val + 1
            self.batch_iter = self.batch_iter + batch_count
    
    
    def fit(self, model, train_dataset, val_dataset, batch_size=64, epochs=10):
        
        """
        Fit model
        """
        
        self.device = model.device
        self.model = model
        self.len_train = len(train_dataset)
        self.len_val = len(val_dataset)
        self.max_epoch = epochs
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            drop_last=False,
            shuffle=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            drop_last=False,
            shuffle=False
        )
        
        if self.device is None:
            self.device = get_default_device()
            self.model.to(self.device)
        
        module = self.model.module
        
        try:
            self.epoch = self.model.epoch
            self.do_training = True
            
            # Start train
            self.on_start_train()
            
            while self.do_training and not self.check_is_trained():
                
                self.loss_train = 0
                self.loss_val = 0
                self.acc_train = 0
                self.acc_val = 0
                self.count_train = 0
                self.count_val = 0
                self.batch_train = 0
                self.batch_val = 0
                self.batch_iter = 0
                self.epoch = self.epoch + 1
                self.time_start = time.time()
                
                self.on_start_epoch()
                module.train()
                
                # Обучение
                for batch_x, batch_y in self.train_loader:
                    
                    if self.model.transform_x:
                        batch_x = self.model.transform_x(batch_x)
                    
                    if self.model.transform_y:
                        batch_y = self.model.transform_y(batch_y)
                    
                    batch_x = batch_to(batch_x, self.device)
                    batch_y = batch_to(batch_y, self.device)
                    
                    # Start batch
                    self.on_start_batch_train(batch_x, batch_y)
                    
                    # Predict
                    batch_predict = module(batch_x)
                    loss_value = self.model.loss(batch_predict, batch_y)
                    
                    # Calc metrics
                    self.calc_metrics("train", batch_x, batch_y, batch_predict, loss_value)
                    del batch_x, batch_y, batch_predict
                    
                    # Вычислим градиент
                    self.model.optimizer.zero_grad()
                    loss_value.backward()
                    del loss_value
                    
                    # Оптимизируем
                    self.model.optimizer.step()
                    
                    # End batch
                    self.on_end_batch_train()
                  
                    # Clear cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                module.eval()
                
                # Вычислим ошибку на проверочном датасете
                for batch_x, batch_y in self.val_loader:
                    
                    if self.model.transform_x:
                        batch_x = self.model.transform_x(batch_x)
                    
                    if self.model.transform_y:
                        batch_y = self.model.transform_y(batch_y)
                    
                    batch_x = batch_to(batch_x, self.device)
                    batch_y = batch_to(batch_y, self.device)
                    
                    # Start batch
                    self.on_start_batch_val(batch_x, batch_y)
                    
                    # Predict
                    batch_predict = module(batch_x)
                    loss_value = self.model.loss(batch_predict, batch_y)
                    
                    # Calc metrics
                    self.calc_metrics("valid", batch_x, batch_y, batch_predict, loss_value)
                    del batch_x, batch_y, batch_predict, loss_value
                    
                    # End batch
                    self.on_end_batch_val()
                    
                    # Clear cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                # Двигаем шедулер
                self.model.scheduler.step(self.loss_val)
                self.time_end = time.time()
                
                self.on_end_epoch()
            
            self.on_end_train()
            self.do_training = False
            
        except KeyboardInterrupt:
            
            print ("")
            print ("Stopped manually")
            print ("")
            
            pass
