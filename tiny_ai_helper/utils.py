# -*- coding: utf-8 -*-

##
# Tiny ai helper
# Copyright (с) Ildar Bikmamatov 2022 - 2023 <support@bayrell.org>
# License: MIT
##

import torch, math, json, os, re, time
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw


class TransformDataset(torch.utils.data.Dataset):
    
    def __init__(self, dataset, transform_x=None, transform_y=None):
        self.dataset = dataset
        self.transform_x = transform_x
        self.transform_y = transform_y
        
    def __getitem__(self, index):
        
        x, y = self.dataset[index]
        
        if x is not None and self.transform_x:
            x = self.transform_x(x)
        
        if y is not None and self.transform_y:
            y = self.transform_y(y)
        
        return x, y
        
    def __len__(self):
        return len(self.dataset)


def append_tensor(res, t):
    
    """
    Append tensor
    """
    
    t = t[None, :]
    res = torch.cat( (res, t) )
    return res


def make_index(arr, field_name=None):
    
    """
    Make index from arr. Returns dict of positions values in arr
    """
    
    res = {}
    for index in range(len(arr)):
        value = arr[index]
        if field_name is not None:
            value = value[field_name]
        res[value] = index
    
    return res


def one_hot_encoder(num_class):
    
    """
    Returns one hot encoder to num class
    """
    
    def f(t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        t = nn.functional.one_hot(t.to(torch.int64), num_class).to(torch.float32)
        return t
    
    return f


def label_encoder(labels):
    
    """
    Returns one hot encoder from label
    """
    
    labels = make_index(labels)
    
    def f(label_name):
        
        index = labels[label_name] if label_name in labels else -1
        
        if index == -1:
            return torch.zeros( len(labels) )
        
        t = torch.tensor(index)
        return nn.functional.one_hot(t.to(torch.int64), len(labels)).to(torch.float32)
    
    return f


def bag_of_words_encoder(dictionary_sz):
    
    """
    Returns bag of words encoder from dictionary indexes.
    """
    
    def f(text_index):
        
        t = torch.zeros(dictionary_sz - 1)
        for index in text_index:
            if index > 0:
                t[index - 1] = 1
        
        return t
        
    return f


def dictionary_encoder(dictionary, max_words):
    
    """
    Returns one hot encoder from text.
    In dictionary 0 pos is empty value, if does not exists in dictionary
    """
    
    def f(text_arr):
        
        t = torch.zeros(max_words).to(torch.int64)
        text_arr_sz = min(len(text_arr), max_words)
        
        pos = 0
        for i in range(text_arr_sz):
            word = text_arr[i]
            
            if word in dictionary:
                index = dictionary[word]
                t[pos] = index
                pos = pos + 1
        
        return t
    
    return f


def batch_map(f):
    
    def transform(batch_x):
        
        res = torch.tensor([])
        
        for i in range(len(batch_x)):
            x = f(batch_x[i])
            x = x[None, :]
            res = torch.cat( (res, x) )
        
        return res.to(batch_x.device)
    
    return transform


def batch_to(x, device):
    
    """
    Move batch to device
    """
    
    if isinstance(x, list) or isinstance(x, tuple):
        for i in range(len(x)):
            x[i] = x[i].to(device)
    else:
        x = x.to(device)
    
    return x


def tensor_size(t):

    """
    Returns tensor size
    """
    
    if not isinstance(t, torch.Tensor):
        return 0, 0
    
    sz = t.element_size()
    shape = t.shape
    params = 1

    for c in shape:
        params = params * c

    size = params * sz

    return params, size


def create_dataset_indexes(dataset, file_name):
    
    """
    Load dataset indexes
    """
    
    index = load_json( file_name )
    if index is None:
        index = list(np.random.permutation( len(dataset) ))
        save_json(file_name, index, indent=None)
    
    return index


def split_dataset(dataset, k=0.2, indexes=None):
    
    """
    Split dataset for train and validation
    """
    
    sz = len(dataset)
    train_count = round(sz * (1 - k))
    val_count = sz - train_count
    
    if indexes is None:
        indexes = list(np.random.permutation(sz))
    
    from torch.utils.data import Subset
    return [Subset(dataset, indexes[0 : train_count]), Subset(dataset, indexes[train_count : ])]
    


def get_default_device():
    
    """
    Returns default device
    """
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    return device


def get_acc_class():
    
    def f(batch_predict, batch_y):
        
        """
        Returns class accuracy
        """
        
        if len(batch_y.shape) == 2:
            batch_y = torch.argmax(batch_y, dim=1)
        
        batch_predict = torch.argmax(batch_predict, dim=1)
        acc = torch.sum( torch.eq(batch_y, batch_predict) ).item()
        
        return acc
    
    return f


def get_acc_binary(treshold=0.5):
    
    def f(batch_predict, batch_y):
        
        batch_predict = (batch_predict >= treshold) * 1.0
        acc = torch.sum( torch.eq(batch_y, batch_predict) ).item()
        
        if len(batch_y.shape) == 2:
            return acc / batch_y.shape[1]
            
        return acc
        
    return f
    
    
    """
    Returns binary accuracy
    """
    
    from torcheval.metrics import BinaryAccuracy
    
    batch_predict = batch_predict.reshape(batch_predict.shape[0])
    batch_y = batch_y.reshape(batch_y.shape[0])
    
    acc = BinaryAccuracy() \
        .to(batch_predict.device) \
        .update(batch_predict, batch_y) \
        .compute().item()
    
    return round(acc * len(batch_y))


def resize_image(image, new_size, contain=True, color=None):
   
    """
    Resize image
    """
    
    if isinstance(image, str):
        image = Image.open(image)
    
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    w1 = image.size[0]
    h1 = image.size[1]
    w2 = new_size[0]
    h2 = new_size[1]

    k1 = w1 / h1
    k2 = w2 / h2
    w_new = 0
    h_new = 0
    
    if k1 > k2 and contain or k1 < k2 and not contain:
        h_new = round(w2 * h1 / w1)
        w_new = w2
        
    else:
        h_new = h2
        w_new = round(h2 * w1 / h1)
    
    image_new = image.resize( (w_new, h_new) )
    image_resize = resize_image_canvas(image_new, new_size, color=color)
    del image_new
    
    return image_resize
    

def resize_image_canvas(image, size, color=None):
   
    """
    Resize image canvas
    """
    
    width, height = size
    
    if color == None:
        pixels = image.load()
        color = pixels[0, 0]
        del pixels
    
    image_new = Image.new(image.mode, (width, height), color = color)
    
    position = (
        math.ceil((width - image.size[0]) / 2),
        math.ceil((height - image.size[1]) / 2),
    )
    
    image_new.paste(image, position)
    return image_new


def load_image(file_name, convert=None, load_as=""):
    
    image = Image.open(file_name)
    
    if convert:
        image = image.convert(convert)
    
    if load_as == "numpy":
        image = np.array(image)
    
    elif load_as == "torch":
        image = torch.from_numpy(np.array(image))
    
    return image


def draw_images(images, ncols=4, cmap=None, first_channel=False):

    import matplotlib.pyplot as plt
    
    count_images = len(images)
    
    fig, ax = plt.subplots(
        (count_images - 1) // ncols + 1, ncols,
        figsize=(10, 6), sharey=True, sharex=True
    )
    fig.subplots_adjust(0, 0, 1, 1)
    ax = ax.flatten()

    # Рисуем картинки
    for i in range(count_images):
        image = images[i]
        
        if isinstance(image, str):
            image = load_image(image, load_as="torch")
        
        if isinstance(image, np.ndarray):
            pass
        
        if isinstance(image, torch.Tensor):
            if first_channel == True:
                image = move_rgb_to_end(image)
        
        ax[i].imshow(image, cmap)

    # Скрываем лишние картинки
    for i in range(count_images, len(ax)):
        ax[i].set_visible(False)
    
    plt.show()


def draw_images_grid(images, ncols=4, first_channel=False, *args, **kwargs):
    
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid
    
    if isinstance(images, list):
        images = images.copy()
        for index in range(len(images)):
            
            if isinstance(images[index], Image.Image):
                images[index] = torch.from_numpy( np.array(images[index]) )
            
            if isinstance(images[index], np.ndarray):
                images[index] = torch.from_numpy( images[index] )
            
            if not first_channel:
                images[index] = move_rgb_to_begin(images[index])
    
    elif isinstance(images, torch.Tensor):
        if not first_channel:
            images = move_rgb_to_begin(images)
    
    plt.imshow(move_rgb_to_end(make_grid(images, nrow=ncols, *args, **kwargs)))
    plt.show()


def draw_image(image, cmap=None, first_channel=False):
    
    """
    Plot show image
    """
    
    if isinstance(image, str):
        image = Image.open(image)
    
    elif isinstance(image, np.ndarray):
        pass
    
    elif torch.is_tensor(image):
        if first_channel == True:
            image = move_rgb_to_end(image)
    
    import matplotlib.pyplot as plt
    
    plt.imshow(image, cmap)
    plt.show()


def move_rgb_to_end(t):
    l = len(t.shape)
    t = torch.moveaxis(t, l-3, l-1)
    return t


def move_rgb_to_begin(t):
    l = len(t.shape)
    t = torch.moveaxis(t, l-1, l-3)
    return t

def swap_hw(t):
    l = len(t.shape)
    t = torch.moveaxis(t, l-1, l-2)
    return t


def list_files(path="", recursive=True, full_path=False):
    
    """
        Returns files in folder
    """
    
    def read_dir(path, recursive=True):
        res = []
        items = os.listdir(path)
        for item in items:
            
            item_path = os.path.join(path, item)
            
            if item_path == "." or item_path == "..":
                continue
            
            if os.path.isdir(item_path):
                if recursive:
                    res = res + read_dir(item_path, recursive)
            else:
                res.append(item_path)
            
        return res
    
    try:
        items = read_dir( path, recursive )
            
        def f(item):
            return item[len(path + "/"):]
        
        items = list( map(f, items) )
    
    except Exception:
        items = []
    
    if full_path:
        items = list( map(lambda x: os.path.join(path, x), items) )
    
    return items


def get_sort_alphanum_key(name):
    
    """
    Returns sort alphanum key
    """
    
    arr = re.split("([0-9]+)", name)
    
    for key, value in enumerate(arr):
        try:
            value = int(value)
        except:
            pass
        arr[key] = value
    
    arr = list(filter(lambda item: item != "", arr))
    
    return arr


def alphanum_sort(files):
    
    """
    Alphanum sort
    """
    
    files.sort(key=get_sort_alphanum_key)


def list_dirs(path=""):
    
    """
        Returns dirs in folder
    """
    
    try:
        items = os.listdir(path)
    except Exception:
        items = []
    
    return items


class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save_json(file_name, obj, indent=2):
    
    """
    Save json to file
    """
    
    json_str = json.dumps(obj, indent=indent, cls=JSONEncoder, ensure_ascii=False)
    file = open(file_name, "w")
    file.write(json_str)
    file.close()


def load_json(file_name):
    
    """
    Load json from file
    """
    
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
    
    return obj


def load_epoch(model, model_name, epoch, repository_path="model"):
        
    """
    Load epoch
    """
    
    model_path = os.path.join(repository_path, model_name)
    
    file_name = model_name + "-" + str(epoch) + ".data"
    file_path = os.path.join(model_path, file_name)
    
    if not os.path.exists(file_path):
        file_name = model_name + "-" + str(epoch) + ".pth"
        file_path = os.path.join(model_path, file_name)
    
    load_model_from_file(model, file_path)


def load_model(model, model_name, file_name, repository_path="model"):
    
    """
    Load model
    """
    
    model_path = os.path.join(repository_path, model_name)
    file_path = os.path.join(model_path, file_name)
    
    if not os.path.exists(file_path):
        file_path = os.path.join(model_path, file_name)
    
    load_model_from_file(model, file_path)
    

def load_model_from_file(model, file_path):
        
    """
    Load model from file
    """
    
    save_metrics = torch.load(file_path)
    state_dict = save_metrics
    
    if "epoch" in save_metrics:
        if isinstance(model, nn.Module):
            state_dict = save_metrics["module"]
    
    model.load_state_dict(state_dict, strict=False)


def summary(module, x, model_name=None, device=None, batch_size=2, collate_fn=None):
        
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
        
        output = output[0] if isinstance(output, tuple) else output
        output_shape = "(?)"
        if hasattr(output, "shape") and isinstance(output, torch.Tensor):
            output_shape = output.shape
            
        class_name = module.__class__.__module__ + "." + module.__class__.__name__
        layer = {
            "name": module.__class__.__name__,
            "class_name": module.__class__.__module__ + "." + module.__class__.__name__,
            "shape": output_shape,
            "params": 0
        }
        
        if layer["name"] == model_name:
            layer["name"] = "Output"
        
        # Calc parameters
        for name, p in module.named_parameters():
            
            if "." in name:
                continue
            
            params = p.numel()
            size = p.numel() * p.element_size()
            
            res["params_count"] += params
            res["total_size"] += size
            layer["params"] += params
            
            if p.requires_grad:
                res["params_train_count"] += params
        
        # Add output size
        params, size = tensor_size(output)
        #res["total_size"] += size
        
        # Add layer
        layers.append(layer)
            
    def add_hooks(module):
        hooks.append(module.register_forward_hook(forward_hook))
    
    # Get batch from Dataset
    batch = None
    if isinstance(x, torch.utils.data.Dataset):
        loader = torch.utils.data.DataLoader(
            x,
            batch_size=batch_size,
            collate_fn=collate_fn,
            drop_last=False,
            shuffle=False
        )
        it = loader._get_iterator()
        
        batch = next(it)
        
        if hasattr(module, "batch_transform"):
            batch = module.batch_transform(batch, device)
        
        x = batch["x"]
    
    # Add input size
    if isinstance(x, list) or isinstance(x, tuple):
        shapes = []
        for i in range(len(x)):
            params, size = tensor_size(x[i])
            shapes.append(x[i].shape)
        #res["total_size"] += size
        layers.append({
            "name": "Input",
            "shape": shapes,
            "params": 0
        })
    
    else:
        params, size = tensor_size(x)
        #res["total_size"] += size
        layers.append({
            "name": "Input",
            "shape": x.shape,
            "params": 0
        })
    
    module.apply(add_hooks)
    
    if hasattr(module, "step_forward"):
        _, _, y = module.step_forward(batch, device)
    
    else:
        
        # Move to device
        if device is not None:
            x = batch_to(x, device)
        
        # Module predict
        with torch.no_grad():
            module.eval()
            y = module(x)
    
    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Remove hooks
    for item in hooks:
        item.remove()
    
    res['total_size'] = round(res['total_size'] / 1024 / 1024 * 100) / 100
    
    # Calc info
    values = []
    for i, layer in enumerate(layers):
        shape = layer["shape"]
        shape_str = ""
        if isinstance(shape, list):
            shape = [ "(" + ", ".join(map(str,s)) + ")" for s in shape ]
            shape_str = "[" + ", ".join(shape) + "]"
        else:
            shape_str = "(" + ", ".join(map(str,shape)) + ")"
        
        values.append([i + 1, layer["name"], shape_str, layer["params"]])
    
    # Print info
    info_sizes = [2, 7, 7, 5]
    for _, value in enumerate(values):
        for i in range(4):
            sz = len(str(value[i]))
            if info_sizes[i] < sz:
                info_sizes[i] = sz
        
    def format_row(arr, size):
        s = "{:<"+str(size[0] + 1)+"} {:>"+str(size[1] + 2)+"}" + \
            "{:>"+str(size[2] + 5)+"} {:>"+str(size[3] + 5)+"}"
        return s.format(*arr)
    
    width = info_sizes[0] + 1 + info_sizes[1] + 2 + info_sizes[2] + 5 + info_sizes[3] + 5 + 2
    print( "=" * width )
    print( format_row(["", "Layer", "Output", "Params"], info_sizes) )
    print( "-" * width )
    
    for _, value in enumerate(values):
        print( format_row(value, info_sizes) )
    
    print( "-" * width )
    if model_name is not None:
        print( f"Model name: {model_name}" )
    print( f"Total params: {res['params_count']:_}".replace('_', ' ') )
    if res['params_count'] != res['params_train_count'] and res['params_train_count'] > 0:
        print( f"Trainable params: {res['params_train_count']:_}".replace('_', ' ') )
    print( f"Total size: {res['total_size']} MiB" )
    print( "=" * width )


def compile(module):
    from .Model import Model
    return Model(module)


def fit(
    model, train_dataset=None, val_dataset=None,
    batch_size=64, epochs=10, collate_fn=None,
    callbacks=None, do_train=True, do_val=True,
    **params
):
    if callbacks is None:
        callbacks = []
    
    callbacks.insert(0, model.module)
    callbacks.insert(0, model)
    
    params["model"] = model
    params["train_dataset"] = train_dataset
    params["val_dataset"] = val_dataset
    params["batch_size"] = batch_size
    params["epochs"] = epochs
    params["status"] = None
    params["callbacks"] = callbacks
    params["collate_fn"] = collate_fn
    params["iter"] = {}
    
    device = model.device
    model_name = model.get_model_name()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Train loader
    if not "train_loader" in params:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            drop_last=False,
            shuffle=True
        )
        params["train_loader"] = train_loader
    
    # Val loader
    val_loader = None
    if val_dataset:
        if not "val_loader" in params:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                collate_fn=collate_fn,
                drop_last=False,
                shuffle=False
            )
            params["val_loader"] = val_loader
    
    acc_fn = model.acc_fn
    device = model.device
    loss_fn = model.loss
    min_lr = model.min_lr
    module = model.module
    optimizer = model.optimizer
    scheduler = model.scheduler
    
    batch_transform = getattr(module, "batch_transform", None)
    step_forward = getattr(module, "step_forward", None)
    step_loss = getattr(module, "step_loss", None)
    step_scheduler = getattr(module, "step_scheduler", None)
    get_batch_size = getattr(module, "get_batch_size", None)
    
    if isinstance(loss_fn, nn.Module):
        loss_fn = loss_fn.to(model.device)
    
    def call_callback(name, params):
        if callbacks is not None:
            for callback in callbacks:
                if hasattr(callback, name):
                    f = getattr(callback, name, None)
                    if f is not None:
                        f(params)
    
    
    call_callback("on_start", params)
    
    print ("Start train " + str(model_name) + " on " + str(device))
    try:
        while model.do_training(epochs):
            
            time_start = time.time()
            
            params["status"] = model.get_epoch_train_status()
            
            params["status"]["total_count"] = len(train_dataset)
            if val_dataset is not None:
                params["status"]["total_count"] += len(val_dataset)
            
            call_callback("on_start_epoch", params)
            
            train_loader = params["train_loader"]
            val_loader = params["val_loader"]
            
            if do_train != False:
                
                # Train mode
                model.train()
                
                for batch in train_loader:
                    
                    batch_len = 1
                    if get_batch_size is not None:
                        batch_len = get_batch_size(batch)
                    else:
                        batch_len = len(batch["x"])
                    
                    # Set parameter gradients to zero
                    optimizer.zero_grad()
                    
                    acc_value = None
                    loss = None
                    
                    # Forward train
                    if step_forward is not None:
                        loss, acc_value, _ = step_forward(
                            batch, device,
                            loss_fn=loss_fn, acc_fn=acc_fn
                        )
                    
                    else:
                        
                        # Data to device
                        if batch_transform:
                            batch = batch_transform(batch, device)
                        
                        x_batch = batch_to(batch["x"], device)
                        y_batch = batch_to(batch["y"], device)
                        y_pred = module(x_batch)
                        
                        # Calc loss
                        if step_loss is not None:
                            loss = step_loss(y_pred, y_batch, loss_fn=loss_fn)
                        else:
                            loss = loss_fn(y_pred, y_batch, loss_fn=loss_fn)
                        
                        # Calc accuracy
                        if acc_fn is not None:
                            acc_value = acc_fn(y_pred, y_batch)
                        
                        params["iter"]["x_batch"] = x_batch
                        params["iter"]["y_batch"] = y_batch
                        params["iter"]["y_pred"] = y_pred
                        
                        del x_batch, y_batch, y_pred
                    
                    # Backward
                    loss.backward()
                    optimizer.step()
                    
                    # Add status
                    params["status"]["pos"] += batch_len
                    params["status"]["train_count"] += batch_len
                    params["status"]["train_loss_items"].append( loss.item() )
                    params["status"]["t"] = round(time.time() - time_start)
                    
                    if acc_value is not None:
                        params["status"]["train_acc_items"].append( acc_value )
                    
                    call_callback("on_train_iter", params)
                    
                    # Clear cache
                    if "x_batch" in params["iter"]:
                        del params["iter"]["x_batch"]
                    
                    if "y_batch" in params["iter"]:
                        del params["iter"]["y_batch"]
                    
                    if "y_pred" in params["iter"]:
                        del params["iter"]["y_pred"]
                    
                    del loss
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            call_callback("on_train", params)
            
            if val_loader is not None and do_val != False:
                
                # Testing mode
                model.eval()
                
                with torch.no_grad():

                    for batch in val_loader:
                        
                        batch_len = 1
                        if get_batch_size is not None:
                            batch_len = get_batch_size(batch)
                        else:
                            batch_len = len(batch["x"])
                        
                        acc_value = None
                        loss = None
                        
                        # Forward
                        if step_forward is not None:
                            loss, acc_value, _ = step_forward(
                                batch, device,
                                loss_fn=loss_fn, acc_fn=acc_fn
                            )
                            
                        else:
                            
                            # data to device
                            if batch_transform:
                                batch = batch_transform(batch, device)
                            
                            x_batch = batch_to(batch["x"], device)
                            y_batch = batch_to(batch["y"], device)
                            y_pred = module(x_batch)
                            
                            # Calc loss
                            if step_loss is not None:
                                loss = step_loss(y_pred, y_batch)
                            else:
                                loss = loss_fn(y_pred, y_batch)
                            
                            # Calc accuracy
                            if acc_fn is not None:
                                acc_value = acc_fn(y_pred, y_batch)
                            
                            params["iter"]["x_batch"] = x_batch
                            params["iter"]["y_batch"] = y_batch
                            params["iter"]["y_pred"] = y_pred
                            
                            del x_batch, y_batch, y_pred
                        
                        # Add status
                        params["status"]["pos"] += batch_len
                        params["status"]["val_count"] += batch_len
                        params["status"]["val_loss_items"].append( loss.item() )
                        params["status"]["t"] = round(time.time() - time_start)
                        
                        if acc_value is not None:
                            params["status"]["val_acc_items"].append( acc_value )
                        
                        call_callback("on_val_iter", params)
                        
                        # Clear cache
                        if "x_batch" in params["iter"]:
                            del params["iter"]["x_batch"]
                        
                        if "y_batch" in params["iter"]:
                            del params["iter"]["y_batch"]
                        
                        if "y_pred" in params["iter"]:
                            del params["iter"]["y_pred"]
                        
                        del loss
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
            
            call_callback("on_val", params)
            call_callback("on_next_epoch", params)
            
            # Add metricks
            time_end = time.time()
            params["status"]["t"] = round(time_end - time_start)
            params["status"]["time_end"] = time_end
            
            call_callback("on_end_epoch", params)
            
            if scheduler is not None:
                if step_scheduler is not None:
                    step_scheduler(params)
                else:
                    scheduler.step()
            
            model.add_epoch(params)
            call_callback("on_save", params)
            
            model.epoch = model.epoch + 1
            
        call_callback("on_end", params)
        
    except KeyboardInterrupt:
        
        print ("")
        print ("Stopped manually")
        print ("")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_embeddings(dataset, file_name, transform, emb_size, batch_size=8):
    
    import h5py, gc
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False
    )
    
    pos = 0
    next_pos = 0
    dataset_count = len(dataset)
    time_start = time.time()
    
    if os.path.exists(file_name):
        os.remove(file_name)
    
    with h5py.File(file_name, 'w') as file:
        
        file_dataset = file.create_dataset(
            'data', (dataset_count, emb_size), dtype='float32'
        )
        
        for batch in loader:
            
            batch = transform(batch)
            file_dataset[pos:pos+len(batch["x"])] = batch["x"].cpu().numpy()
            
            # Show progress
            pos = pos + len(batch["x"])
            if pos > next_pos:
                next_pos = pos + 16
                t = str(round(time.time() - time_start))
                print ("\r" + str(math.floor(pos / dataset_count * 100)) + "% " + t + "s", end='')
            
            del batch
            
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            gc.collect()
            file.flush()
        
        file.close()


def colab_upload_file_to_google_drive(src, dest):
    
    import shutil
    
    if not os.path.exists('/content/drive'):
        from google.colab import drive
        drive.mount('/content/drive')
    
    dest_dir_path = os.path.dirname(dest)
    if not os.path.exists(dest_dir_path):
        os.makedirs(dest_dir_path)
    
    shutil.copy(src, dest)
    

class ListDataset(Dataset):
    
    def __init__(self, items):
        self.items = items
    
    def __getitem__(self, index: int):
        return self.items[index]
    
    def __len__(self) -> int:
        return len(self.items)
