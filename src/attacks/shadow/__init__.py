import os

import torch
from torchvision import transforms

from ...config import Shadow_ROOT, class_n_gtsrb, class_n_lisa
from .gtsrb import GtsrbCNN
from .lisa import LisaCNN


def load_shadow_model(attack_db, model_type, device='cuda'):
    assert attack_db in ['LISA', 'GTSRB']
    if attack_db == "LISA":
        model = LisaCNN(n_class=class_n_lisa).to(device)
        model.load_state_dict(
            torch.load(os.path.join(Shadow_ROOT, f'./model/{"adv_" if model_type == "robust" else ""}model_lisa.pth'),
                    map_location=torch.device(device)))
        # pre_process = transforms.Compose([transforms.ToTensor()])
    else:
        model = GtsrbCNN(n_class=class_n_gtsrb).to(device)
        model.load_state_dict(
            torch.load(os.path.join(Shadow_ROOT, f'./model/{"adv_" if model_type == "robust" else ""}model_gtsrb.pth'),
                    map_location=torch.device(device)))
        # pre_process = transforms.Compose([
        #     pre_process_image, transforms.ToTensor()])
    model.eval()
    return model

