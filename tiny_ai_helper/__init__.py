# -*- coding: utf-8 -*-

##
# Tiny ai helper
# Copyright (с) Ildar Bikmamatov 2022 - 2023 <support@bayrell.org>
# License: MIT
##

from .Model import Model
from .utils import compile, fit

__version__ = "0.1.10"

__all__ = (
    "Model",
    "compile",
    "fit",
)
