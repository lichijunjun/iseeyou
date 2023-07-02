'''
    Four attacks are considered in this project.
        [1] Shadow attack
            | datasets  | GTSRB, LISA
            | root_path | /home/kemove/ShadowAttack-master

        [2] AdvLB
            | datasets  | ImageNet
            | root_path | /home/kemove/Advlight-main/AdvLB
        [3] AdvLS
            | datasets  | ImageNet
            | root_path | /home/kemove/Advlight-main/AdvLS-main
        [4] AdvOptical
            | datasets  | 
            | root_path | /home/kemove/Advlight-main/optical_adversarial_attack-main
'''

import os

from torch.utils.data import ConcatDataset

from .. import config
from .adv_dataset import AdvImageDataset, NormalImageDataset
from .advlb import load_advlb_model
from .advls import load_advls_model
from .benchmark import BenchmarkDataset
from .optical import load_optical_model
from .shadow import load_shadow_model


def load_model(attack_type, attack_db, model_type, device):
    if attack_type == 'advlb':
        model = load_advlb_model(model_type=model_type, device=device)
    elif attack_type == 'advls':
        model = load_advls_model(device=device)
    elif attack_type == 'optical':
        model = load_optical_model(device=device)
    elif attack_type == 'shadow':
        model = load_shadow_model(attack_db, model_type, device)
    else:
        raise ValueError(f"Unkown Attack Type {attack_type}")
    
    return model


def analyzer1(path):
    '''
        analyzer for AdvLB, AdvLS
    '''
    return int(path.split('.')[0])

def analyzer2(path):
    '''
        analyzer for Shadow Attack
        cv2.imwrite(f"{save_dir}/{index}_{labels[index]}_{num_query}_{success}.jpg", adv_img)

    '''
    file_name = path.split('/')[-1]
    true_label = int(file_name.split('_')[1])
    return true_label


def load_adv_samples(attack_type, attack_db="", args=None):
    if attack_type == 'advlb':
        path = config.AdvLB_adv_samples_path
        target_analyzer = analyzer1
    elif attack_type == 'advls':
        path = config.AdvLS_adv_samples_path
        target_analyzer = analyzer1
    elif attack_type == 'optical':
        path = config.Optical_adv_samples_path
        target_analyzer = None
    elif attack_type == 'shadow':
        path = config.Shadow_adv_samples_path(attack_db)
        target_analyzer = analyzer2
        # format: .bmp files
    else:
        raise ValueError(f"Unkown Attack Type {attack_type}")
    
    return AdvImageDataset(attack_type, path, target_analyzer, loader=loaders[attack_type] if not args.std else loader_advls)


def load_normal_samples(attack_type, attack_db="", args=None):
    if attack_type == 'advlb':
        path = config.AdvLB_normal_samples_path
        target_analyzer = analyzer1
    elif attack_type == 'advls':
        path = config.AdvLS_normal_samples_path
        target_analyzer = analyzer1
    elif attack_type == 'optical':
        path = config.Optical_normal_samples_path
        target_analyzer = None
    elif attack_type == 'shadow':
        path = config.Shadow_normal_samples_path(attack_db)
        target_analyzer = analyzer2
        # format: .bmp files
    else:
        raise ValueError(f"Unkown Attack Type {attack_type}")
    
    return NormalImageDataset(attack_type, path, target_analyzer, loader=loaders[attack_type] if not args.std else loader_advls)


def load_dataset(attack_method, attack_db, args):
    data_list = []
    data_list.append(load_adv_samples(attack_method, attack_db, args))
    data_list.append(load_normal_samples(attack_method, attack_db, args))
    return ConcatDataset(data_list)


from PIL import Image
from torchvision.datasets.folder import default_loader


def loader_advls(path):

    return Image.open(path).convert("RGB")

def loader_advlb(path):
    return Image.open(path).convert('RGB').resize((224, 224), Image.BILINEAR)

def loader_optical(path):
    return Image.open(path).resize((224,224), Image.ANTIALIAS)

def loader_shadow(path):
    if isinstance(path, str):
        return default_loader(path)
    else:
        print(type(path))
        return path

loaders = {
    "advls":loader_advls,
    "advlb":loader_advlb,
    'optical':loader_optical,
    "shadow":loader_shadow
}
from torchvision.datasets.folder import default_loader


def load_benchmark(atk_type, atk_db, transform=None, split=False, loader=None):
    root_dir = config.benchmark_data_root
    if not split:
        return BenchmarkDataset(
            os.path.join(root_dir, atk_type, atk_db.upper()), transform=transform, loader=loader if loader is not None else default_loader
        )
    else:
        return (
            BenchmarkDataset(
                os.path.join(root_dir, atk_type, atk_db.upper()), transform=transform, split=True, train=True
            ),
            BenchmarkDataset(
                os.path.join(root_dir, atk_type, atk_db.upper()), transform=transform, split=True, train=False
            ),
        )