#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

from .data_augment import TrainTransform, ValTransform
from .data_prefetcher import DataPrefetcher
from .dataloading import DataLoader, get_yolox_datadir
# Dataset adapters are optional in the slim portfolio package.  Browser use and
# ONNX export do not require them.  If the full upstream dataset package is
# present, keep the original public imports; otherwise allow lightweight tools
# to import yolox.data without failing at package-import time.
try:
    from .datasets import *
except ModuleNotFoundError as exc:
    if exc.name != f"{__name__}.datasets":
        raise
from .samplers import InfiniteSampler, YoloBatchSampler
