# -*- coding: utf-8 -*-

##
# Tiny ai helper
# Copyright (с) Ildar Bikmamatov 2022 - 2023 <support@bayrell.org>
# License: MIT
##

import torch, time
import torch.multiprocessing as mp
from .utils import batch_to


def dataset_predict_work(obj):
        
    """
    One thread worker
    """
    
    loader = obj["loader"]
    module = obj["module"]
    device = obj["device"]
    predict = obj["predict"]
    queue = obj["queue"]
    predict_obj = obj["predict_obj"]
    
    queue.put(1)
    
    for batch_x, batch_y in loader:
        
        if device:
            batch_x = batch_to(batch_x, device)
        
        batch_predict = module(batch_x)
        
        if predict:
            predict(batch_x, batch_y, batch_predict, predict_obj)
        
        del batch_x, batch_y, batch_predict
    
    queue.get()
    

class DatasetPredict():
    
    def __init__(self, dataset, model,
        batch_size=4, num_workers=None,
        predict=None, predict_obj=None
    ):
        
        if num_workers is None:
            num_workers = mp.cpu_count()
        
        self.dataset = dataset
        self.model = model
        self.batch_size = batch_size
        self.predict = predict
        self.predict_obj = predict_obj
        self.loader = None
        self.workers = None
        self.num_workers = num_workers
        self.pos = 0
    
    
    def init(self):
        
        """
        Init
        """
        
        # Init model to eval
        self.model.module.share_memory()
        self.model.module.eval()
        
        # Init dataset
        self.loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.num_workers
        )
        
        # Init vars       
        self.queue = mp.Queue()
    
    
    def start(self):
        
        """
        Start one thread predict
        """
        
        obj={
            "module": self.model.module,
            "device": self.model.device,
            "loader": self.loader,
            "predict": self.predict,
            "predict_obj": self.predict_obj,
            "queue": self.queue,
        }
        
        dataset_predict_work(obj)
        
    
    def start_mp(self):
        
        """
        Start multiprocessing predict
        """
        
        q = mp.Queue()
        
        obj={
            "module": self.model.module,
            "device": self.model.device,
            "loader": self.loader,
            "predict": self.predict,
            "predict_obj": self.predict_obj,
            "queue": self.queue,
        }
        
        # Run workers
        self.workers = []
        for _ in range(self.num_workers):
            self.start_worker(
                mp.Process(target=dataset_predict_work, args=(obj,))
            )
    
    
    def start_worker(self, worker):
        
        """
        Start worker
        """
        
        worker.start()
        self.workers.append(worker)
    
    
    def join(self):
        
        """
        Join all workers
        """
        
        if self.workers is not None:
            for p in self.workers:
                p.join()


def get_features_save_file(
    queue, pipe_recv, file_name,
    dataset_count, features_count
):
    
    # Open file to write
    h = ["label", "image_id"] + [ "f_" + str(i) for i in range(features_count) ]
    h = ",".join(h)
    file = open(file_name, "w")
    file.write(h + "\n")
    
    pos = 0
    next_pos = 0
    time_start = time.time()
    while not queue.empty():
        
        if pipe_recv.poll():
            s = pipe_recv.recv()
            file.write(s + "\n")
            
            pos = pos + 1
            if pos > next_pos:
                next_pos = pos + 16
                t = str(round(time.time() - time_start))
                print ("\r" + str(pos) + " " +
                    str(round(pos / dataset_count * 100)) + "% " + t + "s", end='')
                
                file.flush()
                
        time.sleep(0.1)
    
    # Закрыть файл для записи
    file.close()


def get_features_predict(batch_x, batch_y, batch_predict, predict_obj):
        
    pipe_send = predict_obj["pipe_send"]
    
    # Отправляет фичи в Pipe
    sz = len(batch_y)
    for i in range(sz):
        
        s = []
        if isinstance(batch_y, list):
            s = [ batch_item[i] for batch_item in batch_y ]
        s = batch_predict[i].tolist()
        s = list(map(str,s))
        s = ",".join(s)
        pipe_send.send(s)  


def save_features_mp(
    dataset, model, file_name, features_count,
    num_workers=2, batch_size=4
):
    
    """
    Multiprocess save features to CSV
    """
    
    # Variables
    pipe_recv, pipe_send = mp.Pipe()
    
    # Create dataset
    predict = DatasetPredict(
        dataset=dataset,
        model=model,
        batch_size=4,
        num_workers=num_workers,
        predict=get_features_predict,
        predict_obj={
            "pipe_send": pipe_send,
        }
    )
    
    # Init
    predict.init()
    
    # Start
    predict.start_mp()
    predict.start_worker(
        mp.Process(
            target=get_features_save_file,
            args=(
                predict.queue, pipe_recv,
                file_name, len(dataset), features_count
            )
        )
    )
    
    # Join threads
    predict.join()


def save_features(
    dataset, model, file_name,
    features_count, batch_size=4
):
    
    """
    One thread save features to CSV
    """
    
    # Init
    model.module.eval()
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False
    )
    
    # Open file to write
    h = ["label", "image_id"] + [ "f_" + str(i) for i in range(features_count) ]
    h = ",".join(h)
    file = open(file_name, "w")
    file.write(h + "\n")
    
    pos = 0
    next_pos = 0
    dataset_count = len(dataset)
    time_start = time.time()
    device = model.device
    module = model.module
    
    for batch_x, batch_y in loader:
        
        # Predict batch
        if device:
            batch_x = batch_to(batch_x, device)
        
        batch_predict = module(batch_x)
        
        # Save predict to file
        sz = len(batch_y)
        for i in range(sz):
            s = []
            if isinstance(batch_y, list):
                s = [ batch_item[i] for batch_item in batch_y ]
            s = batch_predict[i].tolist()
            s = list(map(str,s))
            s = ",".join(s)
            file.write(s + "\n")
        
        # Delete batch
        del batch_x, batch_y, batch_predict
        
        # Show progress
        pos = pos + sz
        if pos > next_pos:
            next_pos = pos + 16
            t = str(round(time.time() - time_start))
            print ("\r" + str(pos) + " " +
                str(round(pos / dataset_count * 100)) + "% " + t + "s", end='')
            
            file.flush()
    
    file.close()