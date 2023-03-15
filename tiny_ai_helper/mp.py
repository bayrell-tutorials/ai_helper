# -*- coding: utf-8 -*-

##
# Tiny ai helper
# Copyright (с) Ildar Bikmamatov 2022 - 2023 <support@bayrell.org>
# License: MIT
##

import torch, time, math
import torch.multiprocessing as mp
from .utils import batch_to
  

class MultiProcessPredict():
    
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
        self.workers = []
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
            shuffle=False
        )
        
        # Init vars
        self.finish_queue = mp.Queue()
        self.worker_queue = mp.Queue()
        self.loader_queue = mp.Queue( self.num_workers * 4 )
        
        # Setup obj
        obj={
            "module": self.model.module,
            "device": self.model.device,
            "loader": self.loader,
            "predict": self.predict,
            "predict_obj": self.predict_obj,
            "worker_queue": self.worker_queue,
            "loader_queue": self.loader_queue,
            "finish_queue": self.finish_queue,
            "transform_x": self.model.transform_x,
            "transform_y": self.model.transform_y,
        }
        
        # Add workers
        for _ in range(self.num_workers):
            self.add_worker(
                mp.Process(target=dataset_predict_worker, args=(obj,))
            )
    
    
    def add_worker(self, worker):
        
        """
        Add worker
        """
        
        self.workers.append(worker)
    
    
    def start(self):
        
        """
        Start multiprocessing predict
        """
        
        # Set work flag to start
        self.finish_queue.put(1)
        
        # Start workers
        for p in self.workers:
            p.start()
        
        # Loader loop
        for batch_x, batch_y in self.loader:
            
            if self.worker_queue.empty():
                break
            
            # Send batch to workers
            success = False
            while not success and not self.worker_queue.empty():
                try:
                    self.loader_queue.put( (batch_x, batch_y), True, 0.5 )
                    success = True
                except:
                    pass
                
        
        # Set finish flag to True
        if not self.finish_queue.empty():
            self.finish_queue.get()
        
        # Join to all workers
        for p in self.workers:
            p.join()


def dataset_predict_worker(obj):
        
    """
    One thread worker
    """
    
    import gc
    
    loader = obj["loader"]
    module = obj["module"]
    device = obj["device"]
    predict = obj["predict"]
    predict_obj = obj["predict_obj"]
    worker_queue = obj["worker_queue"]
    loader_queue = obj["loader_queue"]
    finish_queue = obj["finish_queue"]
    transform_x = obj["transform_x"]
    transform_y = obj["transform_y"]
    
    worker_queue.put(1)
    
    try:
        while not ( finish_queue.empty() and loader_queue.empty() ):
            
            batch_x = None
            batch_y = None
            
            try:
                batch_x, batch_y = loader_queue.get(True, 0.5)
            except:
                pass
            
            if batch_x is None:
                continue
            
            if transform_x:
                batch_x = transform_x(batch_x)
            
            if transform_y:
                batch_y = transform_y(batch_y)
            
            if device:
                batch_x = batch_to(batch_x, device)
            
            batch_predict = module(batch_x)
            
            if predict:
                predict(batch_x, batch_y, batch_predict, predict_obj)
            
            del batch_x, batch_y, batch_predict
            
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            gc.collect()
    
    except Exception as e:
        print(e)
        pass
    
    finally:
        time.sleep(1)
        worker_queue.get()


def get_features_predict(batch_x, batch_y, batch_predict, predict_obj):
        
    pipe_send = predict_obj["pipe_send"]
    
    # Send features to pipe
    sz = len(batch_x)
    for i in range(sz):
        
        s = []
        if isinstance(batch_y, list):
            s = [ batch_item[i] for batch_item in batch_y ]
        s += batch_predict[i].tolist()
        s = list(map(str,s))
        s = ",".join(s)
        
        pipe_send.send(s)


def get_features_save_file(
    worker_queue, finish_queue, pipe_recv, file_name,
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
    while not finish_queue.empty() or not worker_queue.empty() or pipe_recv.poll():
        
        if pipe_recv.poll():
            s = pipe_recv.recv()
            file.write(s + "\n")
            
            pos = pos + 1
            if pos > next_pos:
                next_pos = pos + 16
                t = str(round(time.time() - time_start))
                print ("\r" + str(pos) + " " +
                    str(math.floor(pos / dataset_count * 100)) + "% " + t + "s", end='')
                
                file.flush()
                
        time.sleep(0.1)
    
    t = str(round(time.time() - time_start))
    print ("\r" + str(pos) + " " +
        str(math.floor(pos / dataset_count * 100)) + "% " + t + "s", end='')
    
    # Закрыть файл для записи
    file.flush()
    file.close()
    
    print("\nOk")


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
    predict = MultiProcessPredict(
        dataset=dataset,
        model=model,
        batch_size=batch_size,
        num_workers=num_workers,
        predict=get_features_predict,
        predict_obj={
            "pipe_send": pipe_send,
        }
    )
    
    # Init
    predict.init()
    
    # Add save file worker
    predict.add_worker(
        mp.Process(
            target=get_features_save_file,
            args=(
                predict.worker_queue, predict.finish_queue, pipe_recv,
                file_name, len(dataset), features_count
            )
        )
    )
    
    # Start
    predict.start()


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
        
        if model.transform_x:
            batch_x = model.transform_x(batch_x)
        
        if model.transform_y:
            batch_y = model.transform_y(batch_y)
        
        # Predict batch
        if device:
            batch_x = batch_to(batch_x, device)
        
        batch_predict = module(batch_x)
        
        # Save predict to file
        sz = len(batch_x)
        for i in range(sz):
            s = []
            if isinstance(batch_y, list):
                s = [ batch_item[i] for batch_item in batch_y ]
            s += batch_predict[i].tolist()
            s = list(map(str,s))
            s = ",".join(s)
            file.write(s + "\n")
        
        file.flush()
        
        # Delete batch
        del batch_x, batch_y, batch_predict
        
        # Show progress
        pos = pos + sz
        if pos > next_pos:
            next_pos = pos + 16
            t = str(round(time.time() - time_start))
            print ("\r" + str(pos) + " " +
                str(math.floor(pos / dataset_count * 100)) + "% " + t + "s", end='')
    
    file.close()