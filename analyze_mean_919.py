import numpy as np
import torch

# 加载 .pth 文件中的数据
normal_mean_stds = torch.load("/mnt/data/approx-killer/approx-killer/trash/advlb_cst_mean_std_6.pth")
adv_mean_stds = torch.load("/mnt/data/approx-killer/approx-killer/trash/advlb_cst_mean_std_5.pth")

# 提取每个样本的特征（均值和标准差）
def extract_features(data):
    features = []
    for item in data:
        array1, array2 = item

        # 计算每个数组的均值和标准差
        mean1, std1 = array1.mean(axis=0), array1.std(axis=0)
        mean2, std2 = array2.mean(axis=0), array2.std(axis=0)

        # 将均值和标准差合并为一个特征向量
        feature_vector = np.concatenate((mean1, std1, mean2, std2))
        features.append(feature_vector)

    return np.array(features)

# 提取正常样本和对抗样本的特征
normal_mean_stds = extract_features(normal_mean_stds)
adv_mean_stds = extract_features(adv_mean_stds)

# 计算均值比率和标准差比率，并过滤 NaN 值
normal_mean_ratio_rgb = normal_mean_stds[:, 0] / normal_mean_stds[:, 2]
normal_std_ratio_rgb = normal_mean_stds[:, 3] / normal_mean_stds[:, 1]

# 过滤包含 NaN 值的样本
nan_mask1 = np.isnan(normal_mean_ratio_rgb)
nan_mask2 = np.isnan(normal_std_ratio_rgb)
valid_mask = ~(nan_mask1 | nan_mask2)

normal_mean_stds = normal_mean_stds[valid_mask]
normal_mean_ratio_rgb = normal_mean_ratio_rgb[valid_mask]
normal_std_ratio_rgb = normal_std_ratio_rgb[valid_mask]

# 获取每个样本在 RGB 通道上的最大值
normal_mean_max_values = normal_mean_ratio_rgb
normal_std_max_values = normal_std_ratio_rgb

# 对抗样本特征比率计算
adv_mean_ratio_rgb = adv_mean_stds[:, 0] / adv_mean_stds[:, 2]
adv_std_ratio_rgb = adv_mean_stds[:, 3] / adv_mean_stds[:, 1]
adv_mean_max_values = adv_mean_ratio_rgb
adv_std_max_values = adv_std_ratio_rgb

# 遍历不同的阈值组合，寻找最佳组合
max_one = 0
max_idx = -1
for y in range(0, 1000):
    std_threshold = y * 0.01
    for x in range(0, 1000):
        mean_threshold = x * 0.01
        adv_pos_num = np.sum((adv_mean_max_values > mean_threshold) | (adv_std_max_values > std_threshold))
        normal_pos_num = normal_mean_max_values.shape[0] - np.sum((normal_mean_max_values > mean_threshold) | (normal_std_max_values > std_threshold))
        total_num = normal_mean_max_values.shape[0] + adv_mean_max_values.shape[0]
        print(adv_pos_num, normal_pos_num)
        pos_ratio = (adv_pos_num + normal_pos_num) / total_num
        if pos_ratio > max_one:
            max_one = pos_ratio
            max_idx = (mean_threshold, std_threshold)

# 输出最佳的阈值组合和对应的最大成功率
print(max_idx, max_one)
