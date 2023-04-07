# -*- coding: utf-8 -*-

##
# Tiny ai helper
# Copyright (с) Ildar Bikmamatov 2022 - 2023 <support@bayrell.org>
# License: MIT
##

import torch, math, json, os, re
import numpy as np
from torch import nn
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


def make_index(arr, file_name=None):
    
    """
    Make index from arr. Returns dict of positions values in arr
    """
    
    res = {}
    for index in range(len(arr)):
        value = arr[index]
        if file_name is not None:
            value = value[file_name]
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
    
    if isinstance(x, list):
        for i in range(len(x)):
            x[i] = x[i].to(device)
    else:
        x = x.to(device)
    
    return x


def tensor_size(t):

    """
    Returns tensor size
    """

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


def get_acc_class(batch_predict, batch_y):
    """
    Returns class accuracy
    """
    
    batch_y = torch.argmax(batch_y, dim=1)
    batch_predict = torch.argmax(batch_predict, dim=1)
    acc = torch.sum( torch.eq(batch_y, batch_predict) ).item()
    
    return acc


def get_acc_binary(batch_predict, batch_y):
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
    image_resize = resize_image_canvas(image_new, new_size)
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


def show_image_in_plot(image, cmap=None, is_float=False, first_channel=False):
    
    """
    Plot show image
    """
    
    if isinstance(image, str):
        image = Image.open(image)
    
    if torch.is_tensor(image):
        if first_channel == True:
            image = torch.moveaxis(image, 0, 2)
        
        if is_float:
            image = image * 255
            image = image.to(torch.uint8)
    
    import matplotlib.pyplot as plt
    
    plt.imshow(image, cmap)
    plt.show()


def list_files(path="", recursive=True):
    
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
    
    json_str = json.dumps(obj, indent=indent, cls=JSONEncoder)
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


def summary(module, x, model_name=None, transform_x=None, device=None):
        
        """
        Show model summary
        """
        
        hooks = []
        layers = []
        res = {
            "layer_name_max": 10,
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
            
            if res["layer_name_max"] < len(module.__class__.__name__):
                res["layer_name_max"] = len(module.__class__.__name__)
            
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
        
        # Get batch from Dataset
        if isinstance(x, torch.utils.data.Dataset):
            loader = torch.utils.data.DataLoader(
                x,
                batch_size=2,
                drop_last=False,
                shuffle=False
            )
            it = loader._get_iterator()
            
            x, _ = next(it)
        
        # Trasform
        if transform_x is not None:
            x = transform_x(x)
        
        # Move to device
        if device is not None:
            x = batch_to(x, device)
        
        # Add input size
        if isinstance(x, list):
            shapes = []
            for i in range(len(x)):
                params, size = tensor_size(x[i])
                shapes.append(x[i].shape)
            res["total_size"] += size
            layers.append({
                "name": "Input",
                "shape": shapes,
                "params": 0
            })
        else:
            params, size = tensor_size(x)
            res["total_size"] += size
            layers.append({
                "name": "Input",
                "shape": x.shape,
                "params": 0
            })
        
        # Module predict
        module.apply(add_hooks)
        y = module(x)
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Remove hooks
        for item in hooks:
            item.remove()
        
        # Print info
        def format_row(res, args):
            layer_name_max = res["layer_name_max"] + 5
            s = "{:<5} {:>"+str(layer_name_max)+"} {:>20} {:>15}"
            return s.format(*args)
        
        res['total_size'] = round(res['total_size'] / 1024 / 1024 * 100) / 100
        
        width = 63
        print( "=" * width )
        print( format_row(res, ["", "Layer", "Output", "Params"]) )
        print( "-" * width )
        
        for i, layer in enumerate(layers):
            shape = layer["shape"]
            shape_str = ""
            if isinstance(shape, list):
                shape = [ "(" + ", ".join(map(str,s)) + ")" for s in shape ]
                shape_str = "[" + ", ".join(shape) + "]"
            else:
                shape_str = "(" + ", ".join(map(str,shape)) + ")"
            print( format_row(res, [i + 1, layer["name"], shape_str, layer["params"]]) )
        
        print( "-" * width )
        if model_name is not None:
            print( f"Model name: {model_name}" )
        print( f"Total params: {res['params_count']}" )
        print( f"Trainable params: {res['params_train_count']}" )
        print( f"Total size: {res['total_size']} MiB" )
        print( "=" * width )