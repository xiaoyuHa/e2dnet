# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
# from .losses import (L1Loss, MSELoss, PSNRLoss, CharbonnierLoss, FFTLoss, HuberLoss, TVLoss)
#
# __all__ = [
#     'L1Loss', 'MSELoss', 'PSNRLoss', 'CharbonnierLoss', 'FFTLoss', 'HuberLoss', 'TVLoss'
# ]

from .losses import (CharbonnierLoss, SSIMLoss, FFTLoss, PSNRLoss)

__all__ = [
    'CharbonnierLoss', 'SSIMLoss', 'FFTLoss', 'PSNRLoss'
]