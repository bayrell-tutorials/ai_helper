# -*- coding: utf-8 -*-

##
# Tiny ai helper
# Copyright (с) Ildar Bikmamatov 2022 - 2023 <support@bayrell.org>
# License: MIT
##

import torch, time
from .utils import TransformDataset, get_default_device, batch_to
from torch.utils.data import DataLoader, TensorDataset


class Trainer:
    
    def __init__(self):
        
        self.device = None
        self.model = None
        self.step = 0
        self.loss_train = 0
        self.loss_val = 0
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
        self.min_step = 5
        self.max_step = 10
        self.min_loss_val = -1
        self.do_training = False
        
    
    def check_is_trained(self):
        
        """
        Returns True if model is trained
        """
        
        if self.step >= self.max_step:
            return True
        
        if self.loss_val < self.min_loss_val and self.step >= self.min_step:
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
        print (f"\rStep {self.step}, {batch_iter_value}%, acc: {acc_train}%", end='')
    
    
    def on_start_batch_val(self, batch_x, batch_y):
        pass
    
    
    def on_end_batch_val(self):
        
        # Лог обучения
        acc_train = str(round(self.acc_train / self.count_train * 100))
        batch_iter_value = round(self.batch_iter / (self.len_train + self.len_val) * 100)
        print (f"\rStep {self.step}, {batch_iter_value}%, acc: {acc_train}%", end='')
    
    
    def on_end_epoch(self):
        
        # Получить текущий lr
        res_lr = []
        for param_group in self.model.optimizer.param_groups:
            res_lr.append(param_group['lr'])
        res_lr = str(res_lr)
        
        # Результат обучения
        loss_train = '%.3e' % (self.loss_train / self.count_train)
        loss_val = '%.3e' % (self.loss_val / self.count_val)
        
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
        print (f"Step {self.step}, " +
            f"acc: {acc_train}%, acc_val: {acc_val}%, rel: {acc_rel_str}, " +
            f"loss: {loss_train}, loss_val: {loss_val}, lr: {res_lr}, " +
            f"t: {time}s"
        )
        
        # Update model history
        self.model.step = self.step
        self.model.history[self.step] = {
            "loss_train": self.loss_train / self.count_train,
            "loss_val": self.loss_val / self.count_val,
            "acc_train": self.acc_train / self.count_train,
            "acc_val": self.acc_val / self.count_val,
            "acc_rel": acc_rel,
            "count_train": self.count_train,
            "count_val": self.count_val,
            "batch_iter": self.batch_iter,
            "time": time,
        }
        
        # Save model
        self.model.save_step()
        self.model.save_the_best_models(epoch_count=self.max_best_models)
    
    
    def on_end_train(self):
        pass
    
    
    def stop_training(self):
        self.do_training = False
    
    
    def fit(self, model, train_dataset, val_dataset, batch_size=64, epochs=10):
        
        """
        Fit model
        """
        
        self.device = model.device
        self.model = model
        self.len_train = len(train_dataset)
        self.len_val = len(val_dataset)
        self.max_step = epochs
        
        # Create new transform dataset
        if self.model.transform_x is not None or self.model.transform_y is not None:
            
            train_dataset = TransformDataset(
                train_dataset,
                transform_x=self.model.transform_x,
                transform_y=self.model.transform_y
            )
            
            val_dataset = TransformDataset(
                val_dataset,
                transform_x=self.model.transform_x,
                transform_y=self.model.transform_y
            )
        
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
        get_acc_fn = self.model.acc_fn
        
        try:
            self.step = self.model.step
            self.do_trainin = True
            
            # Start train
            self.on_start_train()
            
            while self.do_trainin and not self.check_is_trained():
                
                self.loss_train = 0
                self.loss_val = 0
                self.acc_train = 0
                self.acc_val = 0
                self.count_train = 0
                self.count_val = 0
                self.batch_iter = 0
                self.step = self.step + 1
                self.time_start = time.time()
                
                self.on_start_epoch()
                module.train()
                
                # Обучение
                for batch_x, batch_y in self.train_loader:
                    
                    batch_count = len(batch_x[0]) if isinstance(batch_x, list) else len(batch_x)
                    batch_x = batch_to(batch_x, self.device)
                    batch_y = batch_to(batch_y, self.device)
                    
                    self.on_start_batch_train(batch_x, batch_y)
                    
                    # Predict
                    model_predict = module(batch_x)
                    loss_value = self.model.loss(model_predict, batch_y)
                    loss_value_item = loss_value.item()
                    acc = get_acc_fn(model_predict, batch_y)
                    
                    del batch_x, batch_y
                    
                    # Вычислим градиент
                    self.model.optimizer.zero_grad()
                    loss_value.backward()
                    del loss_value
                    
                    # Оптимизируем
                    self.model.optimizer.step()
                    
                    self.acc_train = self.acc_train + acc
                    self.loss_train = self.loss_train + loss_value_item
                    self.count_train = self.count_train + batch_count
                    self.batch_iter = self.batch_iter + batch_count
                    
                    self.on_end_batch_train()
                  
                # Clear cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                module.eval()
                
                # Вычислим ошибку на проверочном датасете
                for batch_x, batch_y in self.val_loader:
                    
                    batch_x = batch_to(batch_x, self.device)
                    batch_y = batch_to(batch_y, self.device)
                    
                    self.on_start_batch_val(batch_x, batch_y)
                    
                    # Predict
                    model_predict = module(batch_x)
                    loss_value = self.model.loss(model_predict, batch_y)
                    acc = get_acc_fn(model_predict, batch_y)
                    
                    batch_count = len(batch_x[0]) if isinstance(batch_x, list) else len(batch_x)
                    self.acc_val = self.acc_val + acc
                    self.loss_val = self.loss_val + loss_value.item()
                    self.count_val = self.count_val + batch_count
                    self.batch_iter = self.batch_iter + batch_count
                    
                    self.on_end_batch_val()
                    
                    # Clear cache
                    del batch_x, batch_y, loss_value
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                # Двигаем шедулер
                self.model.scheduler.step(self.loss_val)
                self.time_end = time.time()
                
                self.on_end_epoch()
            
            self.on_end_train()
            
        except KeyboardInterrupt:
            
            print ("")
            print ("Stopped manually")
            print ("")
            
            pass
