# -*- coding: utf-8 -*-

##
# Tiny ai helper
# Copyright (с) Ildar Bikmamatov 2022 - 2023 <support@bayrell.org>
# License: MIT
##

import torch, time, json, math, gc, os
from torch.utils.data import DataLoader, TensorDataset
from .utils import TransformDataset, list_files, \
    get_default_device, batch_to, tensor_size, load_json, summary


class Model:
    
    def __init__(self, name=None):
        self.device = None
        self.transform_x = None
        self.transform_y = None
        self.module = None
        self.optimizer = None
        self.scheduler = None
        self.loss = None
        self.acc_fn = None
        self.name = name
        self.model_path = None
        self.epoch = 0
        self.history = {}
    
    
    def set_transform_x(self, transform_x):
        self.transform_x = transform_x
        return self
    
    
    def set_transform_y(self, transform_y):
        self.transform_y = transform_y
        return self
    
    
    def set_module(self, module):
        self.module = module
        return self
    
    
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
        return self
    
    
    def set_loss(self, loss):
        self.loss = loss
        return self
    
    
    def set_scheduler(self, scheduler):
        self.scheduler = scheduler
        return self
    
    
    def set_acc(self, acc):
        self.acc_fn = acc
        return self
    
    
    def set_name(self, name):
        self.name = name
        return self
    
    
    def set_path(self, model_path):
        self.model_path = model_path
        return self
    
    
    def init(self, acc=None, optimizer=None, loss=None, scheduler=None, lr=1e-3,
        transform_x=None, transform_y=None):
        
        """
        Init model
        """
        
        if acc is not None:
            self.acc_fn = acc
        
        if transform_x is not None:
            self.transform_x = transform_x
        
        if transform_y is not None:
            self.transform_y = transform_y
        
        if loss is not None:
            self.loss = loss
        
        if optimizer is not None:
            self.optimizer = optimizer
        
        if scheduler is not None:
            self.scheduler = scheduler
        
        if self.loss is None:
            self.loss = torch.nn.MSELoss()
        
        if self.optimizer is None:
            try:
                self.optimizer = torch.optim.Adam(self.module.parameters(), lr=lr)
            except:
                pass
        
        if self.scheduler is None and self.optimizer is not None:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( self.optimizer )
        
    
    def to(self, device):
        self.module = self.module.to(device)
        self.device = device
    
    
    def to_gpu(self):
        self.to( get_default_device() )
    
    
    def to_cpu(self):
        self.to( torch.device("cpu") )
    
    
    def train(self):
        self.module.train()
    
    
    def eval(self):
        self.module.eval()
    
    
    def load_file(self, file_path):
        
        """
        Load model from file
        """
        
        save_metrics = torch.load(file_path)
        self.epoch = save_metrics["epoch"]
        
        # Load history
        if "history" in save_metrics:
            self.history = save_metrics["history"].copy()
        
        # Load module
        if "module" in save_metrics:
            state_dict = save_metrics["module"]
            self.module.load_state_dict(state_dict, strict=False)
        
        # Load optimizer
        if "optimizer" in save_metrics:
            state_dict = save_metrics["optimizer"]
            self.optimizer.load_state_dict(state_dict)
        
        # Load scheduler
        if "scheduler" in save_metrics:
            state_dict = save_metrics["scheduler"]
            self.scheduler.load_state_dict(state_dict)
        
        # Load loss
        if "loss" in save_metrics:
            state_dict = save_metrics["loss"]
            self.loss.load_state_dict(state_dict)
        
    
    def load(self, file_name):
        
        """
        Load model by file name
        """
        
        file_path = os.path.join(self.model_path, file_name)
        self.load_file(file_path)
    
    
    def load_epoch(self, epoch):
        
        """
        Load current epoch
        """
        
        file_path = os.path.join(self.model_path, "model-" + str(epoch) + ".data")
        self.load_file(file_path)
        
        
    def load_last(self):
        
        """
        Load last model
        """
        
        file_name = os.path.join(self.model_path, "history.json")
        
        if not os.path.exists(file_name):
            return
        
        obj = load_json(file_name)
        
        if obj is not None:
            epoch = obj["epoch"]
            self.load_epoch(epoch)
        
    
    def load_best(self):
        
        """
        Load best model
        """
        
        file_name = os.path.join(self.model_path, "history.json")
        
        if not os.path.exists(file_name):
            return
        
        obj = load_json(file_name)
        
        if obj is not None:
            best_epoch = obj["best_epoch"]
            self.load_epoch(best_epoch)
        
    
    def save_epoch(self):
        
        """
        Save train status
        """
        
        # Get metrics
        save_metrics = {}
        save_metrics["name"] = self.name
        save_metrics["epoch"] = self.epoch
        save_metrics["history"] = self.history.copy()
        save_metrics["module"] = self.module.state_dict()
        save_metrics["optimizer"] = self.optimizer.state_dict()
        save_metrics["scheduler"] = self.scheduler.state_dict()
        save_metrics["loss"] = self.loss.state_dict()
        
        # Create folder
        if not os.path.isdir(self.model_path):
            os.makedirs(self.model_path)
        
        # Save model to file
        file_name = os.path.join(self.model_path, "model-" + str(self.epoch) + ".data")
        torch.save(save_metrics, file_name)
        
        # Save history to json
        self.save_history()
    
    
    def save_history(self):
        
        """
        Save history to json
        """
        
        best_epoch = self.get_the_best_epoch()
        file_name = os.path.join(self.model_path, "history.json")
        obj = {
            "epoch": self.epoch,
            "best_epoch": best_epoch,
            "history": self.history.copy(),
        }
        json_str = json.dumps(obj, indent=2)
        file = open(file_name, "w")
        file.write(json_str)
        file.close()
    
    
    def predict(self, x, batch_size=64):
        
        """
        Predict
        """
         
        if self.transform_x is not None:
            x = self.transform_x(x)
        
        if self.device:
            x = x.to( self.device )
        
        y = self.module(x)
        
        return y
    
    
    def predict_dataset(self, dataset, predict, batch_size=64, predict_obj=None):
        
        """
        Predict dataset
        """
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            drop_last=False,
            shuffle=False
        )
        
        self.module.eval()
        
        pos = 0
        next_pos = 0
        dataset_count = len(dataset)
        time_start = time.time()
        
        for batch_x, batch_y in loader:
            
            if self.transform_x:
                batch_x = self.transform_x(batch_x)
            
            if self.device:
                batch_x = batch_to(batch_x, self.device)
            
            batch_predict = self.module(batch_x)
            predict(batch_x, batch_y, batch_predict, predict_obj)
            
            # Show progress
            pos = pos + len(batch_x)
            if pos > next_pos:
                next_pos = pos + 16
                t = str(round(time.time() - time_start))
                print ("\r" + str(math.floor(pos / dataset_count * 100)) + "% " + t + "s", end='')
            
            del batch_x, batch_y, batch_predict
            
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            gc.collect()
        
        print ("\nOk")
    
    
    def get_metrics(self, metric_name):
        
        """
        Returns metrics by name
        """
        
        res = []
        epochs = list(self.history.keys())
        for index in epochs:
            
            epoch = self.history[index]
            res2 = [ index ]
            
            if isinstance(metric_name, list):
                for name in metric_name:
                    res2.append( epoch[name] if name in epoch else 0 )
            
            else:
                res2.append( epoch[metric_name] if metric_name in epoch else 0 )
            
            res.append(res2)
            
        return res
    
    
    def get_the_best_epoch(self):
        
        """
        Returns the best epoch
        """
        
        epoch_indexes = self.get_the_best_epochs_indexes(1)
        best_epoch = epoch_indexes[0] if len(epoch_indexes) > 0 else 0
        return best_epoch
    
    
    def get_the_best_epochs_indexes(self, epoch_count=5):
        
        """
        Returns best epoch indexes
        """
        
        metrics = self.get_metrics(["acc_val", "acc_rel"])
        
        def get_key(item):
            return [100 - item[1], item[2]]

        metrics.sort(key=get_key)
        
        res = []
        res_count = 0
        metrics_len = len(metrics)
        loss_val_last = 100
        for index in range(metrics_len):
            
            res.append( metrics[index] )
            
            if loss_val_last != metrics[index][1]:
                res_count = res_count + 1
            
            loss_val_last = metrics[index][1]
            
            if res_count > epoch_count:
                break
        
        res = [ res[index][0] for index in range(len(res)) ]
        
        return res
    
    
    def save_the_best_models(self, epoch_count=5):
        
        """
        Save the best models
        """
        
        def detect_type(file_name):
            
            import re
            
            file_type = ""
            epoch_index = 0
            
            result = re.match(r'^model-(?P<id>[0-9]+)\.data$', file_name)
            if result:
                return "model", int(result.group("id"))
            
            return file_type, epoch_index
        
        
        if self.epoch > 0 and epoch_count > 0 and os.path.isdir(self.model_path):
            
            epoch_indexes = self.get_the_best_epochs_indexes(epoch_count)
            epoch_indexes.append( self.epoch )
            
            files = list_files( self.model_path )
            
            for file_name in files:
                
                file_type, epoch_index = detect_type(file_name)
                if file_type in ["model"] and \
                    epoch_index > 0 and \
                    not (epoch_index in epoch_indexes):
                    
                    file_path = os.path.join( self.model_path, file_name )
                    os.unlink(file_path)
    
    
    def summary(self, x):
        
        """
        Show model summary
        """
        
        summary(self.module, x,
            device=self.device,
            model_name=self.name,
            transform_x=self.transform_x,
        )
    
    
    def draw_history_ax(self, ax, metrics=[], label=None, legend=True, convert=None):
        
        """
        Draw history to axes
        """
        
        metrics_values = self.get_metrics(metrics)
        for index, name in enumerate(metrics):
            values = [ item[index + 1] for item in metrics_values ]
            if convert:
                values = list(map(convert, values))
            ax.plot( values, label=name)
        
        if label:
            ax.set_xlabel( label )
        
        if legend:
            ax.legend()
    
    
    def draw_history(self, metrics=[]):
        
        """
        Draw history
        """
        
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        self.draw_history_ax(ax[0],
            ["acc_train", "acc_val"],
            label="Accuracy",
            convert=lambda x: x * 100
        )
        self.draw_history_ax(ax[1],
            ["loss_train", "loss_val"],
            label="Loss"
        )
        plt.show()
    
    
    def show_history(self):
        
        h = list(self.history.keys())
        h.sort()
        
        for epoch in h:
            
            res = self.history[epoch]
            
            acc_train = res["acc_train"] if "acc_train" in res else 0
            acc_val = res["acc_val"] if "acc_val" in res else 0
            acc_rel = res["acc_rel"] if "acc_rel" in res else 0
            loss_train = res["loss_train"] if "loss_train" in res else 0
            loss_val = res["loss_val"] if "loss_val" in res else 0
            res_lr = res["res_lr"] if "res_lr" in res else 0
            time = res["time"] if "time" in res else 0
            
            acc_train = str(round(acc_train * 10000) / 100)
            acc_val = str(round(acc_val * 10000) / 100)
            acc_train = acc_train.ljust(5, "0")
            acc_val = acc_val.ljust(5, "0")
            acc_rel_str = str(round(acc_rel * 100) / 100).ljust(4, "0")
            loss_train = '%.3e' % loss_train
            loss_val = '%.3e' % loss_val
            res_lr_str = str(res_lr)
            
            print (f"Epoch {epoch}, " +
                f"acc: {acc_train}%, acc_val: {acc_val}%, rel: {acc_rel_str}, " +
                f"loss: {loss_train}, loss_val: {loss_val}, lr: {res_lr_str}, " +
                f"t: {time}s"
            )