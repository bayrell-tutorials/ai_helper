# -*- coding: utf-8 -*-

##
# Tiny ai helper
# Copyright (с) Ildar Bikmamatov 2022 - 2023 <support@bayrell.org>
# License: MIT
##

from .Model import Model, SaveCallback, ProgressCallback
from .utils import compile, fit
from .csv import CSVReader

__version__ = "0.1.12"

__all__ = (
    "Model",
    "SaveCallback",
    "ProgressCallback",
    "CSVReader",
    "compile",
    "fit",
)
