# -*- coding: utf-8 -*-

##
# Tiny ai helper
# Copyright (с) Ildar Bikmamatov 2022 - 2023 <support@bayrell.org>
# License: MIT
##

from .Model import Model
from .Trainer import Trainer
from .utils import compile, fit

__version__ = "0.1.8"

__all__ = (
    "Model",
    "Trainer",
    "compile",
    "fit",
)
