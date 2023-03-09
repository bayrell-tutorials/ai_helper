# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT
##

import json


class Model:
    
    def __init__(self, name=None):
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
        self.history = []
    
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
    
    
    def load_file(self, file_path):
        """
        Load model from file
        """
        
        save_metrics = torch.load(file_path)
        self.step = save_metrics["step"]
        self.history = save_metrics["history"].copy()
        
        state_dict = save_metrics["state_dict"]
        self.module.load_state_dict(state_dict)
    
    
    def save_train(self, trainer):
        """
        Save train status
        """
        
        # Get metrics
        save_metrics["name"] = self.name
        save_metrics["step"] = self.step
        save_metrics["history"] = self.history.copy()
        save_metrics["state_dict"] = self.module.state_dict()
        
        # Create folder
        if not os.path.isdir(self.model_path):
            os.makedirs(self.model_path)
        
        # Save model to file
        file_name = os.path.join(self.model_path, "model-" + str(self.step) + ".data")
        torch.save(save_metrics, file_name)
        
        # Save history to json
        file_name = os.path.join(self.model_path, "history.json")
        json_str = json.dumps(self.history, indent=2)
        file = open(file_name, "w")
        outfile.write(json_str)
        file.close()
    
    
    def predict(self, x, device=None):
        """
        Predict
        """
        
        if device is None:
            device = get_default_device()
        
        module = self.module.to(device)
        x = x.to(device)
        
        if self.transform_x is not None:
            x = self.transform_x(x)
        
        module.eval()
        y = module(x)
        
        return y
        