'''                     
                        NORMAL         ADV
            LISA          500           500
    Shadow
            GTSRB         500           500
    AdvLS                 200           200
    AdvLB                 200           200
    Optical               200           200
'''
import os

from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader


def analyzer(path):
    return int(path.split('/')[-1][:-4].split('_')[-1])

class BenchmarkDataset(Dataset):

    def __init__(self, root_dir, loader=default_loader, transform=None, split=False, train=False) -> None:
        # super().__init__()
        ratio = 0.9
        self.root_dir = root_dir
        self.loader = loader
        self.transform = transform
        self.adv_samples_path_list = [
            os.path.join('adv', x) for x in os.listdir(os.path.join(root_dir, 'adv'))
        ]
        self.adv_target_list = [
            analyzer(x) for x in self.adv_samples_path_list
        ]
        self.normal_samples_path_list = [
            os.path.join('normal', x) for x in os.listdir(os.path.join(root_dir, 'normal'))
        ]
        self.normal_target_list = [
            analyzer(x) for x in self.normal_samples_path_list
        ]
        if split:
            train_length = int(len(self.normal_samples_path_list) * ratio)
            if train: start, end = 0, train_length
            else: start, end = train_length, len(self.normal_samples_path_list)
            self.normal_samples_path_list = self.normal_samples_path_list[start:end]
            self.adv_samples_path_list = self.adv_samples_path_list[start:end]
            self.normal_target_list = self.normal_target_list[start:end]
            self.adv_target_list = self.adv_target_list[start:end]
        self.samples_path_list = self.normal_samples_path_list + self.adv_samples_path_list
        self.target_list = self.normal_target_list + self.adv_target_list
        self.goal_list = [1] * len(self.normal_samples_path_list) + [0] * len(self.adv_samples_path_list)
        print(f"In this benchmark, there are totally {len(self.adv_samples_path_list)} adv samples ans {len(self.normal_samples_path_list)} normal samples")
    def __len__(self):
        return len(self.samples_path_list)
    
    def __getitem__(self, idx):
        
        sample_path = os.path.join(
            self.root_dir,
            self.samples_path_list[idx]
        )
        # print(sample_path)
        data = self.loader(sample_path)
        if self.transform is not None:
            data = self.transform(data)
        
        return data, self.target_list[idx], self.goal_list[idx]