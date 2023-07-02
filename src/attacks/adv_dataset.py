import os
import pickle
from shutil import copyfile

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

from ..config import (AdvLS_data_dict_path, AdvLS_data_dict_std_path,
                      Optical_data_csv_path, attack_goals)

'''
    The same for `NormalImageDataset` and `AdvImageDataset`, 
        the label of `AdvImage` is ground-truth label instead of target label
'''


class NormalImageDataset(Dataset):
    def __init__(self, attack_method, root_dir, target_analyzer, loader=default_loader, transform=None) -> None:
        # super().__init__()
        self.attack_method = attack_method
        self.root_dir = root_dir
        self.target_analyzer = target_analyzer
        self.loader = loader
        self.transform = transform
        if 'adv' in attack_method:
            self.normal_samples_path_list = os.listdir(root_dir)
            meta_dict = torch.load(AdvLS_data_dict_path)
            self.target_list = [
                meta_dict[file_name] for file_name in self.normal_samples_path_list
            ]
            self.goal_list = [1] * len(self.normal_samples_path_list)
        elif attack_method == 'optical':
            self.normal_samples_path_list = os.listdir(root_dir)
            metadata = pd.read_csv(Optical_data_csv_path) 
            self.target_list = [
                metadata.loc[metadata['ImageId'] == img_path[:-4]].iloc[0,6]-1 for img_path in self.normal_samples_path_list
            ]
            self.goal_list = [1] * len(self.normal_samples_path_list)
        elif attack_method == 'shadow':
            with open(root_dir, 'rb') as dataset:
                test_data = pickle.load(dataset)
                self.normal_samples_path_list, self.target_list = \
                    test_data['data'], test_data['labels']
            self.goal_list = [1] * len(self.normal_samples_path_list)

        

    def __len__(self):
        return len(self.normal_samples_path_list)
    
    def __getitem__(self, idx):
        
        if self.attack_method != 'shadow':
            sample_path = os.path.join(
                self.root_dir,
                self.normal_samples_path_list[idx]
            )
            data = self.loader(sample_path)
        else:
            data = self.normal_samples_path_list[idx]

        if self.transform is not None:
            data = self.transform(data)
        
        return data, self.target_list[idx], self.goal_list[idx]

    def save_to(self, idx, target_path):

        if self.attack_method != 'shadow':
            sample_path = os.path.join(
                self.root_dir,
                self.normal_samples_path_list[idx]
            )
            copyfile(sample_path, target_path)
        else:
            print(target_path)
            cv2.imwrite(target_path, self.normal_samples_path_list[idx])

    
    

class AdvImageDataset(Dataset):

    def __init__(self, attack_method, root_dir, target_analyzer, loader=default_loader, transform=None) -> None:
        # super().__init__()
        self.attack_method = attack_method
        self.root_dir = root_dir
        self.target_analyzer = target_analyzer
        self.loader = loader
        self.transform = transform
        self.adv_samples_path_list = os.listdir(root_dir)
        if 'adv' in attack_method:
            meta_dict = torch.load(AdvLS_data_dict_path)
            # print([meta_dict[x] for x in meta_dict.keys()])
            self.target_list = [
                meta_dict[file_name] for file_name in self.adv_samples_path_list
            ]
            self.goal_list = [2] * len(self.adv_samples_path_list)
        elif attack_method == 'optical':
            metadata = pd.read_csv(Optical_data_csv_path) 
            self.target_list = [
                metadata.loc[metadata['ImageId'] == img_path[:-4]].iloc[0,6]-1 for img_path in self.adv_samples_path_list
            ]
            self.goal_list = [2] * len(self.adv_samples_path_list)

        elif attack_method == 'shadow':
            self.target_list = [
                target_analyzer(file_name) for file_name in self.adv_samples_path_list
            ]
            self.goal_list = [2] * len(self.adv_samples_path_list)
    
    def __len__(self):
        return len(self.adv_samples_path_list)
    
    def __getitem__(self, idx):
        
        sample_path = os.path.join(
            self.root_dir,
            self.adv_samples_path_list[idx]
        )
        data = self.loader(sample_path)

        if self.transform is not None:
            data = self.transform(data)
        
        return data, self.target_list[idx], self.goal_list[idx]
    
    def save_to(self, idx, target_path):
        sample_path = os.path.join(
            self.root_dir,
            self.adv_samples_path_list[idx]
        )
        if isinstance(sample_path, str):
            copyfile(sample_path, target_path)
    