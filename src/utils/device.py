import os

import torch

from ..config import CUDA_DEVICES


def gpu_select():
    import pynvml
    pynvml.nvmlInit()  # 初始化

    deviceCount = pynvml.nvmlDeviceGetCount()
    selected_index = 0
    min_used_ratio = 1
    for idx in range(deviceCount):
        if idx in CUDA_DEVICES: continue
        handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used_ratio = info.used / info.total
        if used_ratio < min_used_ratio:
            min_used_ratio = used_ratio
            selected_index = idx

    os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_index)
