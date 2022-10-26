# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT
##

from .model import Model, ExtendModule
from .train import TrainStatus, TrainVerboseCallback, do_train

__version__ = "0.0.5"

__all__ = (
	
	"Model",
	"TrainStatus",
	"TrainVerboseCallback",
	"do_train",
	
)