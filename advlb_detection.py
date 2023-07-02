import argparse

from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from src.attacks.load import load_benchmark
from src.preprocess import MyFilter, custom_preprocess_v
from src.utils import init_env


def task_name(a, b): return f"{a}-{b}" if b!= '.' else "{a}"

parser = argparse.ArgumentParser(description="Test preprocess")
parser.add_argument("--seed", type=int, default=999, help="seed")

args = parser.parse_args()

benchmark_list = [
    ('advlb', '.'),
]

device = init_env(args.seed)

# Load transform
cur_transforms1 = custom_preprocess_v(["mysolarize",  "equalize", "adjustsharpness", "filter", "medianfilter", "grayscale"])

total_num = 0

normal_correct_nums = []
adv_correct_nums = []
total_nums = []
data_list = []
for atk_type, atk_db in benchmark_list:
    print(f"[Task Name] {atk_type}, {atk_db}")
    dataset = load_benchmark(atk_type, atk_db)
    cur_transform1 = cur_transforms1[0]
    
    demo_idx = 1
    max_diff_list = []
    max_normal_max = 0
    min_adv_max = 256
    for idx, (sample, target, adv_type) in tqdm(enumerate(dataset)):
        # if idx < 2000:continue
        total_num += 1

        c_sample = cur_transform1(sample)
        size_scale = transforms.Compose([
           transforms.Resize(256),
           transforms.CenterCrop(224),
        ])
        data = np.array(size_scale(sample))
        mask = np.array(c_sample, dtype=np.bool8)
        mask1 = 1 - mask; mask2 = mask
        normal_data_list = data[np.where(mask1)].reshape(-1, 3)
        mask_data_list = data[np.where(mask2)].reshape(-1, 3)
        data_list.append((normal_data_list, mask_data_list))
    torch.save(data_list, '/mnt/data/approx-killer/approx-killer/trash/advlb_cst_mean_std_5.pth')
'''
Q1: what about the figures after the transformation
Q2: different methods can compromise a part of adv samples, but are they the same part?
'''