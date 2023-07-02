import torch
from torch import nn
from torchvision.models import vgg11_bn

vgg11_bn()



def load_classifier():
    model = vgg11_bn(pretrained=True)
    model.classifier =  nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 2),
        )
    return model