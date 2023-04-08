# -*- coding: utf-8 -*-

##
# Tiny ai helper
# Copyright (с) Ildar Bikmamatov 2022 - 2023 <support@bayrell.org>
# License: MIT
##

from .Model import Model
from .Trainer import Trainer
from .mp import save_features, save_features_mp

__version__ = "0.1.5"

__all__ = (
    "Model",
    "Trainer",
    "DatasetPredict",
    "save_features",
    "save_features_mp",
)
