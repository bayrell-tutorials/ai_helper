# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT
##

from .model import AbstractModel, ExtendModule
from .train import TrainStatus, TrainVerboseCallback, do_train

__version__ = "0.0.5"

__all__ = (
	
	"AbstractModel",
	"TrainStatus",
	"TrainVerboseCallback",
	"do_train",
	
)