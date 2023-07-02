import torch
from torchvision.models import resnet50


def load_optical_model(device="cuda"):
    model = resnet50(pretrained=True)
    model.eval()
    model.to(device)
    return model