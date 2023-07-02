import torch

from .device import gpu_select
from .seed import set_seed


def init_env(seed=42):
    gpu_select()
    set_seed(seed)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")