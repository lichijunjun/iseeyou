import torch
from torchvision.models import resnet50


def load_advls_model(device="cuda"):
    # print('here')
    return resnet50(pretrained=True).eval().to(device)
