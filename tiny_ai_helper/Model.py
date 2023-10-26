# -*- coding: utf-8 -*-

##
# Tiny ai helper
# Copyright (с) Ildar Bikmamatov 2022 - 2023 <support@bayrell.org>
# License: MIT
##

import torch, time, json, math, gc, os
import numpy as np
from torch.utils.data import DataLoader, Dataset
from .utils import TransformDataset, list_files, \
    get_default_device, batch_to, tensor_size, \
    load_json, summary, fit


class Model:
    
    def __init__(self, module=None):
        self.device = 'cpu'
        self.transform_x = None
        self.transform_y = None
        self.module = module
        self.optimizer = None
        self.scheduler = None
        self.loss = None
        self.loss_reduction = 'mean'
        self.best_metrics = ["epoch"]
        self.acc_fn = None
        self.acc_reduction = 'sum'
        self.name = module.__class__.__name__
        self.prefix_name = ""
        self.epoch = 1
        self.history = {}
        self.min_lr = 1e-5
        self.model_path = ""
        self.repository_path = ""
        self.set_repository_path("model")
    
    
    def set_module(self, module):
        self.module = module
        return self
    
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
        return self
    
    def set_loss(self, loss, reduction = 'mean'):
        self.loss = loss
        self.loss_reduction = reduction
        return self
    
    def set_scheduler(self, scheduler):
        self.scheduler = scheduler
        return self
    
    def set_acc(self, acc, reduction = 'sum'):
        self.acc_fn = acc
        self.acc_reduction = reduction
        return self
    
    def set_name(self, name):
        self.name = name
        self.set_repository_path(self.repository_path)
        return self
    
    def set_prefix_name(self, prefix_name):
        self.prefix_name = prefix_name
        self.set_repository_path(self.repository_path)
        return self
    
    def set_path(self, model_path):
        self.model_path = model_path
        return self
    
    def set_best_metrics(self, best_metrics):
        self.best_metrics = best_metrics
        return self
    
    def set_repository_path(self, repository_path):
        self.repository_path = repository_path
        self.model_path = os.path.join(repository_path, self.get_model_name())
        return self
    
    def get_model_name(self):
        if self.prefix_name != "":
            return self.name + "_" + self.prefix_name
        return self.name
    
    def get_epoch(self):
        return self.epoch - 1
    
    
    def to(self, device):
        self.module = self.module.to(device)
        self.device = device
        return self
    
    def to_cuda(self):
        self.to( torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') )
        return self
    
    def to_cpu(self):
        self.to( torch.device("cpu") )
        return self
    
    def train(self):
        self.module.train()
        return self
    
    def eval(self):
        self.module.eval()
        return self
    
    
    def load_state_dict(self, save_metrics, strict=False):
        
        if "epoch" in save_metrics:
            self.epoch = save_metrics["epoch"] + 1
            
            # Load history
            if "history" in save_metrics:
                self.history = save_metrics["history"].copy()
            
            # Load module
            if "module" in save_metrics:
                state_dict = save_metrics["module"]
                self.module.load_state_dict(state_dict, strict=strict)
            
            # Load optimizer
            if "optimizer" in save_metrics:
                state_dict = save_metrics["optimizer"]
                self.optimizer.load_state_dict(state_dict)
            
            # Load scheduler
            if "scheduler" in save_metrics:
                state_dict = save_metrics["scheduler"]
                self.scheduler.load_state_dict(state_dict)
            
            # Load loss
            #if "loss" in save_metrics:
            #    state_dict = save_metrics["loss"]
            #    self.loss.load_state_dict(state_dict)
        
        else:
            self.module.load_state_dict(save_metrics, strict=strict)
        
        return self
    
    
    def load_model(self, file_path, full_path=False):
        
        """
        Load model from file
        """
        
        if not os.path.exists(file_path) and not full_path:
            file_path = os.path.join(self.model_path, file_path)
        
        save_metrics = torch.load(file_path)
        self.load_state_dict(save_metrics)
        
        return self
        
    
    def load_epoch(self, epoch):
        
        """
        Load epoch
        """
        
        file_name = self.get_model_name() + "-" + str(epoch) + ".data"
        file_path = os.path.join(self.model_path, file_name)
        
        if not os.path.exists(file_path):
            file_name = self.get_model_name() + "-" + str(epoch) + ".pth"
            file_path = os.path.join(self.model_path, file_name)
        
        self.load_model(file_path, full_path=True)
        
        return self
        
        
    def load_last(self):
        
        """
        Load last model
        """
        
        file_name = self.get_model_name() + ".data"
        file_path = os.path.join(self.model_path, file_name)
        
        if not os.path.exists(file_path):
            file_name = self.get_model_name() + ".pth"
            file_path = os.path.join(self.model_path, file_name)
        
        if os.path.exists(file_path):
            self.load_model(file_path, full_path=True)
            return self
        
        file_name = os.path.join(self.model_path, "history.json")
        if os.path.exists(file_name):
        
            obj = load_json(file_name)
            
            if obj is not None:
                epoch = obj["epoch"]
                self.load_epoch(epoch)
        
        return self
        
    
    def load_best(self):
        
        """
        Load best model
        """
        
        file_path = os.path.join(self.model_path, "history.json")
        
        if not os.path.exists(file_path):
            return self
        
        obj = load_json(file_path)
        
        if obj is not None:
            best_epoch = obj["best_epoch"]
            self.load_epoch(best_epoch)
        
        return self
    
    
    def save_weights_epoch(self, file_path=None):
        
        """
        Save weights
        """
        
        # Create folder
        if not os.path.isdir(self.model_path):
            os.makedirs(self.model_path)
        
        if file_path is None:
            file_name = self.get_model_name() + "-" + str(self.epoch) + ".pth"
            file_path = os.path.join(self.model_path, file_name)
        
        torch.save(self.module.state_dict(), file_path)
        
        return self
    
    
    def save_weights(self, file_path=None):
        
        """
        Save weights
        """
        
        # Create folder
        if not os.path.isdir(self.model_path):
            os.makedirs(self.model_path)
        
        if file_path is None:
            file_name = self.get_model_name() + ".pth"
            file_path = os.path.join(self.model_path, file_name)
        
        torch.save(self.module.state_dict(), file_path)
        
        return self
    
    
    def save_train_epoch(self):
        
        """
        Save train epoch
        """
        
        # Create folder
        if not os.path.isdir(self.model_path):
            os.makedirs(self.model_path)
        
        file_name = self.get_model_name() + "-" + str(self.epoch) + ".data"
        file_path = os.path.join(self.model_path, file_name)
        self.save_model(file_path)
        
        return self
    
        
    def save_model(self, file_path=None):
        
        """
        Save train status
        """
        
        # Get metrics
        save_metrics = {}
        save_metrics["name"] = self.get_model_name()
        save_metrics["epoch"] = self.epoch
        save_metrics["history"] = self.history.copy()
        save_metrics["module"] = self.module.state_dict()
        
        if self.optimizer is not None:
            save_metrics["optimizer"] = self.optimizer.state_dict()
        
        if self.scheduler is not None:
            save_metrics["scheduler"] = self.scheduler.state_dict()
        
        #if self.loss is not None:
        #    save_metrics["loss"] = self.loss.state_dict()
        
        # Create folder
        if not os.path.isdir(self.model_path):
            os.makedirs(self.model_path)
        
        # Save model to file
        if file_path is None:
            model_file_name = self.get_model_name() + ".data"
            file_path = os.path.join(self.model_path, model_file_name)
        
        torch.save(save_metrics, file_path)
        
        return self
    
    
    def save_history(self):
        
        """
        Save history to json
        """
        
        best_epoch = self.get_the_best_epoch(self.best_metrics)
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
        
        return self
    
    
    def do_training(self, max_epochs):
        
        """
        Returns True if model is need to train
        """
        
        if self.epoch > max_epochs:
            return False
        
        for item in self.optimizer.param_groups:
            if item["lr"] >= self.min_lr:
                return True
        
        return False
    
    
    def set_new_lr(self, lr):
        
        for index, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = lr[index]
        
        return self
    
    
    def __call__(self, x):
        return self.module(x)
    
    
    def predict(self, x):
        
        """
        Predict
        """
        
        batch = {"x":x}
        batch_transform = getattr(self.module, "batch_transform", None)
        
        with torch.no_grad():
            
            self.module.eval()
            
            if batch_transform:
                batch = batch_transform(batch, self.device)
            
            x = batch["x"]
            
            if hasattr(self.module, "step_forward"):
                _, _, y_predict = self.module.step_forward(x, self.device)
            
            else:
                x = batch_to(x, self.device)
                y = self.module(x)
        
        return y
    
    
    def predict_dataset(self, dataset, predict, batch_size=64, obj=None, collate_fn=None):
        
        """
        Predict dataset
        """
        
        loader = None
        device = self.device
        
        if isinstance(dataset, Dataset):
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=collate_fn,
                drop_last=False,
                shuffle=False
            )
        
        if isinstance(dataset, DataLoader):
            loader = dataset
        
        batch_transform = getattr(self.module, "batch_transform", None)
        
        with torch.no_grad():
        
            self.module.eval()
            
            pos = 0
            next_pos = 0
            dataset_count = len(dataset)
            time_start = time.time()
            
            for batch in loader:
                
                if batch_transform:
                    batch = batch_transform(batch, self.device)
                
                if hasattr(self.module, "step_forward"):
                    _, _, y_predict = self.module.step_forward(batch, self.device)
                
                else:
                    x_batch = batch_to(batch["x"], device)
                    y_predict = self.module(x_batch)
                    del x_batch
                
                predict(batch, y_predict, obj)
                
                # Show progress
                pos = pos + len(batch['x'])
                if pos > next_pos:
                    next_pos = pos + 16
                    t = str(round(time.time() - time_start))
                    print (
                        "\r" + str(math.floor(pos / dataset_count * 10000) / 100) + "% " +
                        t + "s", end=''
                    )
                
                del y_predict, batch
                
                # Clear cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                gc.collect()
            
            print ("\nOk")
    
    
    def get_metrics(self, metric_name, convert=False):
        
        """
        Returns metrics by name
        """
        
        def convert_value(value, metric_name):
            if (
                metric_name == "train_acc" or
                metric_name == "train_acc_value" or
                metric_name == "val_acc" or
                metric_name == "val_acc_value" or
                metric_name == "epoch"
            ):
                return -value
            return value
        
        res = []
        epochs = list(self.history.keys())
        for index in epochs:
            
            epoch = self.history[index]
            res2 = [ index ]
            
            if isinstance(metric_name, list):
                for name in metric_name:
                    value = epoch[name] if name in epoch else 0
                    if convert:
                        value = convert_value(value, name)
                    res2.append( value )
            
            else:
                value = epoch[metric_name] if metric_name in epoch else 0
                if convert:
                    value = convert_value(value, metric_name)
                res2.append( value )
            
            res.append(res2)
            
        return res
    
    
    def get_metric(self, metric_name, convert=False):
        
        """
        Returns metrics by name
        """
        
        res = []
        epochs = list(self.history.keys())
        for index in epochs:
            
            epoch = self.history[index]
                        
            value = epoch[metric_name] if metric_name in epoch else 0
            if convert:
                value = convert_value(value, metric_name)
            res.append( value )
            
        return res
    
    
    def get_the_best_epoch(self, best_metrics=None):
        
        """
        Returns the best epoch
        """
        
        epoch_indexes = self.get_the_best_epochs_indexes(1, best_metrics)
        best_epoch = epoch_indexes[0] if len(epoch_indexes) > 0 else 0
        return best_epoch
    
    
    def get_the_best_epochs_indexes(self, epoch_count=5, best_metrics=None):
        
        """
        Returns best epoch indexes
        """
        
        if best_metrics is None:
            best_metrics = self.best_metrics
        
        metrics = self.get_metrics(best_metrics, convert=True)
        metrics.sort(key=lambda x: x[1:])
        
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
    
    
    def get_best_epoch(self, best_metrics=None):
        
        """
        Returns the best epoch
        """
        
        return self.get_the_best_epoch(best_metrics)
    
    
    def get_best_epochs(self, epoch_count=5, best_metrics=None):
        
        """
        Returns best epoch indexes
        """
        
        return self.get_the_best_epochs_indexes(epoch_count, best_metrics)
    
    
    def save_the_best_models(self, max_best_models=10):
        
        """
        Save the best models
        """
        
        def detect_type(file_name):
            
            import re
            
            file_type = ""
            epoch_index = 0
            model_name = self.get_model_name()
            
            result = re.match(r'^'+model_name+'-(?P<id>[0-9]+)\.data$', file_name)
            if result:
                return "model", int(result.group("id"))
            
            result = re.match(r'^'+model_name+'-(?P<id>[0-9]+)\.pth$', file_name)
            if result:
                return "model", int(result.group("id"))
            
            return file_type, epoch_index
        
        
        if self.epoch > 0 and max_best_models > 0 and os.path.isdir(self.model_path):
            
            epoch_indexes = self.get_the_best_epochs_indexes(max_best_models)
            epoch_indexes.append( self.epoch )
            
            files = list_files( self.model_path )
            
            for file_name in files:
                
                file_type, epoch_index = detect_type(file_name)
                if file_type in ["model"] and \
                    epoch_index > 0 and \
                    not (epoch_index in epoch_indexes):
                    
                    file_path = os.path.join( self.model_path, file_name )
                    os.unlink(file_path)
    
    
    def summary(self, x, batch_size=2, collate_fn=None):
        
        """
        Show model summary
        """
        
        summary(self.module, x,
            device=self.device,
            collate_fn=collate_fn,
            model_name=self.get_model_name(),
            batch_size=batch_size
        )
    
        
    def draw_history_ax(self, ax, metrics=[], label=None, legend=True, convert=None, start=0):
        
        """
        Draw history to axes
        """
        
        metrics_values = self.get_metrics(metrics)
        metrics_values = metrics_values[start:]
        for index, name in enumerate(metrics):
            values = [ item[index + 1] for item in metrics_values ]
            if convert:
                values = list(map(convert, values))
            if len(values) > 0 and values[0] is not None:
                ax.plot( values, label=name)
        
        if label:
            ax.set_xlabel( label )
        
        if legend:
            ax.legend()
    
    
    def draw_history(self, show_acc=False, show_loss=True, start=0):
        
        """
        Draw history
        """
        
        import matplotlib.pyplot as plt
        
        pos = 0
        fig, ax = plt.subplots(1, show_acc + show_loss, figsize=(10, 4))
        if show_acc:
            self.draw_history_ax(
                ax[pos] if isinstance(ax, np.ndarray) else ax,
                ["train_acc", "val_acc"],
                label="Accuracy",
                convert=lambda x: x * 100 if x is not None else None,
                start=start
            )
            pos += 1
        if show_loss:
            self.draw_history_ax(
                ax[pos] if isinstance(ax, np.ndarray) else ax,
                ["train_loss", "val_loss"],
                label="Loss",
                start=start
            )
        plt.show()
    
    
    def print_history(self, progress):
        
        h = list(self.history.keys())
        h.sort()
        
        for epoch in h:
            s = progress.get_epoch_string(self.history[epoch].copy())
            print(s)
    
    
    def get_epoch_train_status(self):
        
        status = {
            "epoch": self.epoch,
            "time_start": 0,
            "time_end": 0,
            "train_acc": 0,
            "train_acc_items": [],
            "train_count": 0,
            "train_loss": 0,
            "train_loss_items": [],
            "train_batch_iter": 0,
            "val_acc": 0,
            "val_acc_items": [],
            "val_count": 0,
            "val_loss": 0,
            "val_loss_items": [],
            "val_batch_iter": 0,
            "iter_value": 0,
            "total_count": 0,
            "pos": 0,
            "t": 0,
        }
        
        return status
    
    
    def add_epoch(self, params):
        status = params["status"]
        epoch = status["epoch"]
        self.history[epoch] = status.copy()
    
    
    def on_train_iter(self, params):
        
        status = params["status"]
        train_count = status["train_count"]
        train_acc_items = status["train_acc_items"]
        train_loss_items = status["train_loss_items"]
        
        if self.loss_reduction == "mean":
            if len(train_loss_items) > 0:
                status["train_loss"] = sum(train_loss_items) / len(train_loss_items)
        
        elif self.loss_reduction == "sum":
            if len(train_loss_items) > 0 and train_count > 0:
                status["train_loss"] = sum(train_loss_items) / train_count
        
        if self.acc_reduction == "mean":
            if len(train_acc_items) > 0:
                status["train_acc"] = sum(train_acc_items) / len(train_acc_items)
                status["train_acc_percent"] = status["train_acc"] * 100
        
        elif self.acc_reduction == "sum":
            if len(train_acc_items) > 0 and train_count > 0:
                status["train_acc"] = sum(train_acc_items) / train_count
                status["train_acc_percent"] = status["train_acc"] * 100
        
        if status["total_count"] > 0:
            status["iter_value"] = (status["pos"] / status["total_count"]) * 100
        
    
    def on_val_iter(self, params):
        
        status = params["status"]
        val_count = status["val_count"]
        val_acc_items = status["val_acc_items"]
        val_loss_items = status["val_loss_items"]
        
        if self.loss_reduction == "mean":
            if len(val_loss_items) > 0:
                status["val_loss"] = sum(val_loss_items) / len(val_loss_items)
        
        elif self.loss_reduction == "sum":
            if len(val_loss_items) > 0 and val_count > 0:
                status["val_loss"] = sum(val_loss_items) / val_count
        
        if self.acc_reduction == "mean":
            if len(val_acc_items) > 0:
                status["val_acc"] = sum(val_acc_items) / len(val_acc_items)
                status["val_acc_percent"] = status["val_acc"] * 100
        
        elif self.acc_reduction == "sum":
            if len(val_acc_items) > 0 and val_count > 0:
                status["val_acc"] = sum(val_acc_items) / val_count
                status["val_acc_percent"] = status["val_acc"] * 100
        
        if status["total_count"] > 0:
            status["iter_value"] = (status["pos"] / status["total_count"]) * 100
        
    
    def on_end_epoch(self, params):
        
        self.on_train_iter(params)
        self.on_val_iter(params)
        
        status = params["status"]
        
        lr = []
        for param_group in self.optimizer.param_groups:
            lr.append( param_group['lr'] )
        
        if not(status["train_acc"] is None) and not(status["val_acc"] is None):
            status["rel"] = (status["train_acc"] / status["val_acc"]) if status["val_acc"] > 0 else 0
        
        status["lr"] = lr
        status["lr_str"] = "[" + ",".join([ str(round(item,7)) for item in lr ]) + "]"
    
    
    def get_train_loss(self, epoch=None):
        if epoch is None:
            epoch = self.epoch
        value = self.history[epoch]["train_loss"]
        if value is None:
            value = 0
        return value
    
    
    def get_val_loss(self, epoch=None):
        if epoch is None:
            epoch = self.epoch
        value = self.history[epoch]["train_loss"]
        if value is None:
            value = 0
        return value
    
    
    def get_train_acc(self, epoch=None):
        if epoch is None:
            epoch = self.epoch
        value = self.history[epoch]["train_acc"]
        if value is None:
            value = 0
        return value
    
    
    def get_val_acc(self, epoch=None):
        if epoch is None:
            epoch = self.epoch
        value = self.history[epoch]["train_acc"]
        if value is None:
            value = 0
        return value
    
    
    def get_epoch_metric(self, metric_name, epoch=None):
        if epoch is None:
            epoch = self.epoch
        value = self.history[epoch][metric_name]
        if value is None:
            value = 0
        return value
    
    
    def upload_to_google_drive(self, epoch, repository_path):
        
        import shutil
    
        if not os.path.exists('/content/drive'):
            from google.colab import drive
            drive.mount('/content/drive')
        
        dest_path = os.path.join(repository_path, self.get_model_name())
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        
        def upload(file_name):
            src_file_path = os.path.join(self.model_path, file_name)
            if os.path.exists(src_file_path):
                dest_file_path = os.path.join(dest_path, file_name)
                shutil.copy(src_file_path, dest_file_path)

        upload(self.get_model_name() + "-" + str(epoch) + ".data")
        upload(self.get_model_name() + "-" + str(epoch) + ".pth")
    
    
    def download_from_google_drive(self, epoch, repository_path):
        
        import shutil
        
        if not os.path.exists('/content/drive'):
            from google.colab import drive
            drive.mount('/content/drive')
        
        src_path = os.path.join(repository_path, self.get_model_name())
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        
        def download(file_name):
            src_file_path = os.path.join(src_path, file_name)
            if os.path.exists(src_file_path):
                dest_file_path = os.path.join(self.model_path, file_name)
                shutil.copy(src_file_path, dest_file_path)

        download(self.get_model_name() + "-" + str(epoch) + ".data")
        download(self.get_model_name() + "-" + str(epoch) + ".pth")
    
    
    def upload_history_to_google_drive(self, repository_path):
    
        import shutil
        
        if not os.path.exists('/content/drive'):
            from google.colab import drive
            drive.mount('/content/drive')
        
        if not os.path.exists(repository_path):
            os.makedirs(repository_path)
        
        dest_path = os.path.join(repository_path, self.get_model_name())
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        
        src_file_path = os.path.join(self.model_path, "history.json")
        dest_file_path = os.path.join(dest_path, "history.json")
        shutil.copy(src_file_path, dest_file_path)
    
    
    def download_history_from_google_drive(self, repository_path):
    
        import shutil
        
        if not os.path.exists('/content/drive'):
            from google.colab import drive
            drive.mount('/content/drive')
        
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        
        src_path = os.path.join(repository_path, self.get_model_name())
        src_file_path = os.path.join(src_path, "history.json")
        dest_file_path = os.path.join(self.model_path, "history.json")
        shutil.copy(src_file_path, dest_file_path)


class ReAccuracyCallback():
    
    def on_start_epoch(self, params):
        
        self.train_y_batch = []
        self.train_y_pred = []
        self.val_y_batch = []
        self.val_y_pred = []
    
    def on_train_iter(self, params):
        
        self.train_y_batch += params["iter"]["y_batch"].detach().cpu()
        self.train_y_pred += params["iter"]["y_pred"].detach().cpu()

    def on_val_iter(self, params):
        
        self.val_y_batch += params["iter"]["y_batch"].detach().cpu()
        self.val_y_pred += params["iter"]["y_pred"].detach().cpu()
    
    def on_train(self, params):
        
        acc_fn = params["model"].acc_fn
        params["status"]["train_acc_items"] = [
            acc_fn(torch.vstack(self.train_y_pred), torch.vstack(self.train_y_batch))
        ]
        
    def on_val(self, params):
        
        acc_fn = params["model"].acc_fn
        params["status"]["val_acc_items"] = [
            acc_fn(torch.vstack(self.val_y_pred), torch.vstack(self.val_y_batch))
        ]


class SaveCallback():
    
    def __init__(self, count=20, save_weights=True, save_train=True, save_last=False):
        self.count = count
        self.save_weights = save_weights
        self.save_train = save_train
        self.save_last = save_last
    
    def on_save(self, params):
        
        model = params["model"]
        
        is_save = False
        
        if self.save_weights:
            model.save_train_epoch()
            is_save = True
        
        if self.save_train:
            model.save_weights_epoch()
            is_save = True
        
        if self.count >= 0:
            model.save_the_best_models(self.count)
            is_save = True
        
        if self.save_last:
            model.save_model()
            is_save = True
        
        if is_save:
            model.save_history()


class ProgressCallback():
    
    def __init__(self, one_line=False, progress_iter=True, show_lr=True, show_acc=True):
        self.one_line = one_line
        self.progress_iter = progress_iter
        
        self.progress_string_train = [
            "Epoch: {epoch}",
            "{iter_value:.0f}%",
            "train_acc: {train_acc_percent:.2f}%" if show_acc else "",
            "train_loss: {train_loss:.7f}",
            "{t}s",
        ]
        self.progress_string_train = list(filter(lambda item: item != "", self.progress_string_train))
        self.progress_string_train = ", ".join(self.progress_string_train)
        
        self.progress_string_val = [
            "Epoch: {epoch}",
            "{iter_value:.0f}%",
            "val_acc: {val_acc_percent:.2f}%" if show_acc else "",
            "val_loss: {val_loss:.7f}",
            "{t}s",
        ]
        self.progress_string_val = list(filter(lambda item: item != "", self.progress_string_val))
        self.progress_string_val = ", ".join(self.progress_string_val)
        
        self.epoch_string = [
            "Epoch: {epoch}",
            "train_acc: {train_acc_percent:.2f}%" if show_acc else "",
            "val_acc: {val_acc_percent:.2f}%" if show_acc else "",
            "rel: {rel:.3f}" if show_acc else "",
            "train_loss: {train_loss:.7f}",
            "val_loss: {val_loss:.7f}",
            "lr: {lr_str}" if show_lr else "",
            "t: {t}s",
        ]
        self.epoch_string = list(filter(lambda item: item != "", self.epoch_string))
        self.epoch_string = ", ".join(self.epoch_string)
    
    
    def get_status(self, status):
        return status
    
    
    def get_epoch_string(self, status):
        status = self.get_status(status)
        return self.epoch_string.format(**status)
    
    
    def get_progress_string(self, kind, status):
        
        if kind == "train":
            return self.progress_string_train.format(**status)
        
        if kind == "val":
            return self.progress_string_val.format(**status)
    
    
    def on_train_iter(self, params):
        
        status = params["status"]
        
        if self.progress_iter:
            print ("\r" + self.get_progress_string("train", status), end="")
    
    
    def on_val_iter(self, params):
        
        status = params["status"]
        
        if self.progress_iter:
            print ("\r" + self.get_progress_string("val", status), end="")
    
    
    def on_end_epoch(self, params):
        
        status = params["status"]
        
        if self.one_line:
            print( "\r" + self.get_epoch_string(status), end="" )
        else:
            print( "\r" + self.get_epoch_string(status) )
    
    
    def on_end(self, params):
        if self.one_line:
            print ("")
        
        print ("Ok")


class ReloadDatasetCallback():
    
    def on_start_epoch(self, params):
        
        # Train loader
        params["train_loader"] = DataLoader(
            params["train_dataset"],
            batch_size=params["batch_size"],
            collate_fn=params["collate_fn"],
            drop_last=False,
            shuffle=True
        )
        
        # Val loader
        if "val_dataset" in params:
            params["val_loader"] = DataLoader(
                params["val_dataset"],
                batch_size=params["batch_size"],
                collate_fn=params["collate_fn"],
                drop_last=False,
                shuffle=False
            )


class RandomDatasetCallback():

    def __init__(self, train_count, val_count=None):
        self.train_count = train_count
        self.val_count = val_count

    def get_indices(self, total_count, iter_count):
        indices = list(np.random.permutation(total_count))
        return indices[:iter_count]

    def on_start_epoch(self, params):
        
        from torch.utils.data import SubsetRandomSampler
        
        status = params['status']
        batch_size = params['batch_size']

        params['train_loader'] = DataLoader(
            params['train_dataset'],
            batch_size=batch_size,
            collate_fn=params["collate_fn"],
            sampler=SubsetRandomSampler(
                self.get_indices(len(params['train_dataset']), self.train_count)
            )
        )
        
        status["total_count"] = self.train_count
        
        if self.val_count is not None:
            params['val_loader'] = DataLoader(
                params['val_dataset'],
                batch_size=batch_size,
                collate_fn=params["collate_fn"],
                sampler=SubsetRandomSampler(
                    self.get_indices(len(params['val_dataset']), self.val_count)
                )
            )
            
            status["total_count"] += self.val_count

