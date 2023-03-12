# -*- coding: utf-8 -*-

##
# Tiny ai helper
# Copyright (с) Ildar Bikmamatov 2022 - 2023 <support@bayrell.org>
# License: MIT
##

import torch, json, os
from torch.utils.data import DataLoader, TensorDataset
from .utils import TransformDataset, list_files, get_default_device, batch_to, tensor_size


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
        self.step = 0
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
        
        if self.loss == None:
            self.loss = nn.MSELoss()
        
        if self.optimizer == None:
            self.optimizer = torch.optim.Adam(self.module.parameters(), lr=lr)
        
        if self.scheduler == None:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( self.optimizer )
        
    
    def to(self, device):
        self.module = self.module.to(device)
        self.device = device
    
    
    def to_gpu(self):
        self.to( get_default_device() )
    
    
    def to_cpu(self):
        self.to( torch.device("cpu") )
    
    
    def load_file(self, file_path):
        
        """
        Load model from file
        """
        
        save_metrics = torch.load(file_path)
        self.step = save_metrics["step"]
        
        # Load history
        if "history" in save_metrics:
            self.history = save_metrics["history"].copy()
        
        # Load module
        if "module" in save_metrics:
            state_dict = save_metrics["module"]
            self.module.load_state_dict(state_dict)
        
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
    
    
    def load_step(self, step):
        
        """
        Load current step
        """
        
        file_path = os.path.join(self.model_path, "model-" + str(step) + ".data")
        self.load_file(file_path)
        
        
    def load_last(self):
        
        """
        Load last model
        """
        
        file_name = os.path.join(self.model_path, "history.json")
        
        if not os.path.exists(file_name):
            return
        
        obj = None
        file = None
        
        try:
            
            file = open(file_name, "r")
            s = file.read()
            obj = json.loads(s)
            
        except Exception:
            pass
        
        finally:
            if file:
                file.close()
                file = None
        
        if obj is not None:
            step = obj["step"]
            self.load_step(step)
        
    
    def load_best(self):
        
        """
        Load best model
        """
        
        file_name = os.path.join(self.model_path, "history.json")
        
        if not os.path.exists(file_name):
            return
        
        obj = None
        file = None
        
        try:
            
            file = open(file_name, "r")
            s = file.read()
            obj = json.loads(s)
            
        except Exception:
            pass
        
        finally:
            if file:
                file.close()
                file = None
        
        if obj is not None:
            best_step = obj["best_step"]
            self.load_step(best_step)
        
    
    def save_step(self):
        
        """
        Save train status
        """
        
        # Get metrics
        save_metrics = {}
        save_metrics["name"] = self.name
        save_metrics["step"] = self.step
        save_metrics["history"] = self.history.copy()
        save_metrics["module"] = self.module.state_dict()
        save_metrics["optimizer"] = self.optimizer.state_dict()
        save_metrics["scheduler"] = self.scheduler.state_dict()
        save_metrics["loss"] = self.loss.state_dict()
        
        # Create folder
        if not os.path.isdir(self.model_path):
            os.makedirs(self.model_path)
        
        # Save model to file
        file_name = os.path.join(self.model_path, "model-" + str(self.step) + ".data")
        torch.save(save_metrics, file_name)
        
        # Save history to json
        epoch_indexes = self.get_the_best_epoch(1)
        best_step = epoch_indexes[0] if len(epoch_indexes) > 0 else 0
        file_name = os.path.join(self.model_path, "history.json")
        obj = {
            "step": self.step,
            "best_step": best_step,
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
        
        y = None
        
        if isinstance(x, torch.utils.data.Dataset):
            
            y = torch.tensor([])
            
            if self.transform_x is not None:
                x = TransformDataset(
                    x,
                    transform_x=self.transform_x
                )
            
            loader = DataLoader(
                x,
                batch_size=batch_size,
                drop_last=False,
                shuffle=False
            )
            
            self.module.eval()
            
            for batch_x, _ in loader:
                
                if self.device:
                    batch_x = batch_to(batch_x, self.device)
                
                batch_predict = self.module(batch_x)
                y = torch.cat( (y, batch_predict) )
        
        else:
        
            if self.device:
                x = x.to( self.device )
            
            if self.transform_x is not None:
                x = self.transform_x(x)
            
            self.module.eval()
            y = self.module(x)
        
        return y
    
    
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
                res2 = [ index ]
                for name in metric_name:
                    res2.append( epoch[name] if name in epoch else 0 )
            
            else:
                res2.append( epoch[metric_name] if metric_name in epoch else 0 )
            
            res.append(res2)
            
        return res
    
    
    def get_the_best_epoch(self, epoch_count=5):
        
        """
        Returns teh best epoch
        """
        
        metrics = self.get_metrics(["loss_val", "acc_rel"])
        
        def get_key(item):
            return [item[1], item[2]]

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
        
        
        if self.step > 0 and epoch_count > 0 and os.path.isdir(self.model_path):
            
            epoch_indexes = self.get_the_best_epoch(epoch_count)
            epoch_indexes.append( self.step )
            
            files = list_files( self.model_path )
            
            for file_name in files:
                
                file_type, epoch_index = detect_type(file_name)
                if file_type in ["model"] and \
                    epoch_index > 0 and \
                    not (epoch_index in epoch_indexes):
                    
                    file_path = os.path.join( self.model_path, file_name )
                    os.unlink(file_path)
    
    
    def summary(self, dataset: TensorDataset):
        
        """
        Show model summary
        """
        
        hooks = []
        layers = []
        res = {
            "params_count": 0,
            "params_train_count": 0,
            "total_size": 0,
        }
        
        def forward_hook(module, input, output):
            
            class_name = module.__class__.__module__ + "." + module.__class__.__name__
            layer = {
                "name": module.__class__.__name__,
                "class_name": module.__class__.__module__ + "." + module.__class__.__name__,
                "shape": output.shape,
                "params": 0
            }
            
            # Get weight
            if hasattr(module, "weight"):
                params, size = tensor_size(module.weight)
                res["params_count"] += params
                res["total_size"] += size
                layer["params"] += params
                
                if module.weight.requires_grad:
                    res["params_train_count"] += params
            
            # Get bias
            if hasattr(module, "bias"):
                params, size = tensor_size(module.bias)
                res["params_count"] += params
                res["total_size"] += size
                layer["params"] += params
                
                if module.bias.requires_grad:
                    res["params_train_count"] += params
            
            # Add output size
            params, size = tensor_size(output)
            res["total_size"] += size
            
            # Add layer
            layers.append(layer)
                
        def add_hooks(module):
            hooks.append(module.register_forward_hook(forward_hook))
        
        # Get input tensor
        if self.transform_x is not None:
            dataset = TransformDataset(
                dataset,
                transform_x=self.transform_x
            )
        
        loader = DataLoader(
            dataset,
            batch_size=2,
            drop_last=False,
            shuffle=False
        )
        it = loader._get_iterator()
        
        x, _ = next(it)
        x = batch_to(x, self.device)
        
        # Add input size
        params, size = tensor_size(x)
        res["total_size"] += size
        
        # Module predict
        module: torch.nn.Module = self.module
        module.apply(add_hooks)
        y = module(x)
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Remove hooks
        for item in hooks:
            item.remove()
        
        # Print info
        def format_row(args):
            return "{:<5} {:>20} {:>20} {:>15}".format(*args)
        
        res['total_size'] = round(res['total_size'] / 1024 / 1024 * 100) / 100
        
        width = 63
        print( "=" * width )
        print( format_row(["", "Layer", "Output", "Params"]) )
        print( "-" * width )
        
        for i, layer in enumerate(layers):
            shape = "(" + ", ".join(map(str,layer["shape"])) + ")"
            print( format_row([i + 1, layer["name"], shape, layer["params"]]) )
        
        print( "-" * width )
        print( f"Model name: {self.name}" )
        print( f"Total params: {res['params_count']}" )
        print( f"Trainable params: {res['params_train_count']}" )
        print( f"Total size: {res['total_size']} MiB" )
        print( "=" * width )
        