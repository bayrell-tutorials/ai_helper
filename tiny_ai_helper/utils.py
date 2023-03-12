# -*- coding: utf-8 -*-

##
# Tiny ai helper
# Copyright (с) Ildar Bikmamatov 2022 - 2023 <support@bayrell.org>
# License: MIT
##

import torch, math, json, os
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


def split_dataset(dataset, k=0.2, index=None):
    
    """
    Split dataset for train and validation
    """
    
    sz = len(dataset)
    train_count = round(sz * (1 - k))
    val_count = sz - train_count
    
    if index is None:
        index = list(np.random.permutation(sz))
    
    from torch.utils.data import Subset
    return [Subset(dataset, index[0 : train_count]), Subset(dataset, index[train_count : ])]
    


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


def resize_image(image, size, contain=True, color=None):
   
    """
    Resize image canvas
    """
    
    if contain:
        image_new = image.copy()
        image_new.thumbnail(size, Image.LANCZOS)
        image_new = resize_image_canvas(image_new, size, color=color)
        return image_new
    
    width, height = image.size
    rect = (width, width)
    if width > height:
        rect = (height, height)
    
    image_new = resize_image_canvas(image, rect, color=color)
    image_new.thumbnail(size, Image.Resampling.LANCZOS)
    
    return image_new
    

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
    draw = ImageDraw.Draw(image_new)
    
    position = (
        math.ceil((width - image.size[0]) / 2),
        math.ceil((height - image.size[1]) / 2),
    )
    
    image_new.paste(image, position)
    
    del draw, image
    
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