import numpy as np
import torch

#mydata = torch.load("/mnt/data/approx-killer/approx-killer/trash/advlb_cst_mean_std_5.pth")
#mask_size = np.array([len(x) for x, _ in mydata])

#normal_mask_size = mask_size[:470]
#adv_mask_size = mask_size[470:]
normal_mean_stds = torch.load("/mnt/data/approx-killer/approx-killer/trash/advlb_cst_mean_std_6.pth")
adv_mean_stds = torch.load("/mnt/data/approx-killer/approx-killer/trash/advlb_cst_mean_std_5.pth")

# 合并正常样本和对抗样本
mydata = normal_mean_stds + adv_mean_stds  # 合并两个列表

# 计算 mask size
mask_size = np.array([len(x) for x, _ in mydata])

# 分割正常样本和对抗样本
cut = 1300
normal_mask_size = mask_size[:cut]
adv_mask_size = mask_size[cut:]

#threshold = 2894
threshold = 4315
filtered_normal = normal_mask_size > threshold
filtered_adv = adv_mask_size > threshold
normal_pos_base = len(normal_mask_size) - np.sum(filtered_normal)


#cut = 470
#threshold = 2894
#filtered_normal = normal_mask_size > threshold
#normal_pos_base = cut - np.sum(filtered_normal)
#filtered_adv = adv_mask_size > threshold


# min_adv_mask_size = np.partition(adv_mask_size, 100-1)[100-1]
# print(min_adv_mask_size)
# print(normal_mask_size.max())

# max_mask_thre = 0
# max_pos_num = 0
# max_ratio = 0
# for x in range(224 * 224):
#     normal_pos_num = np.sum(normal_mask_size < x)
#     adv_pos_num = np.sum(adv_mask_size > x)
#     # if max_pos_num < adv_pos_num + normal_pos_num:
#         # max_pos_num = adv_pos_num + normal_pos_num
#     if max_ratio < normal_pos_num - (2000 - np.sum(adv_mask_size > x)):
#     # if max_pos_num <  normal_pos_num:
#         max_ratio = normal_pos_num - (2000 - np.sum(adv_mask_size > x))
#         max_pos_num = normal_pos_num
#         max_mask_thre = x
#         print(max_mask_thre, max_pos_num, (2000 - np.sum(adv_mask_size > x)))

'''
2891 1176 199
2894 1177 199
2982 1186 207
2986 1187 207

<-- 2894 is selected.
'''

# print(mydata)

mean_std = np.array(
    [(x.mean(axis=0), x.std(axis=0), y.mean(axis=0), y.std(axis=0)) for x, y in mydata]
)


print(mean_std.shape)
normal_mean_stds = mean_std[:cut][filtered_normal]
adv_mean_stds = mean_std[cut:][filtered_adv]
print(normal_mean_stds.shape, adv_mean_stds.shape)

normal_mean_ratio_rgb = normal_mean_stds[:, 0] / normal_mean_stds[:, 2]
nan_mask1 = np.isnan(normal_mean_ratio_rgb).any(axis=(1))
normal_std_ratio_rgb = normal_mean_stds[:, 3] / normal_mean_stds[:, 1]
nan_mask2 = np.isnan(normal_std_ratio_rgb).any(axis=(1))
normal_mean_stds = normal_mean_stds[~(nan_mask1 + nan_mask2)]
normal_mean_max_values = normal_mean_ratio_rgb.max(axis=1)
normal_std_max_values = normal_std_ratio_rgb.max(axis=1)

empty_num = np.sum((nan_mask1 + nan_mask2))

adv_mean_ratio_rgb = adv_mean_stds[:, 0]  / adv_mean_stds[:, 2]
adv_std_ratio_rgb = adv_mean_stds[:, 3] / adv_mean_stds[:, 1]
adv_std_max_values = adv_std_ratio_rgb.max(axis=1)
adv_mean_max_values = adv_mean_ratio_rgb.max(axis=1)

max_one = 0
max_idx = -1
adv_pos_num_t, normal_pos_num_t = -1, -1
for y in range(0, 500):
    std_threshold = y * 0.01
    for x in range(0, 500):
        mean_threshold = x * 0.01
        adv_pos_num = np.sum((adv_mean_max_values > mean_threshold) + (adv_std_max_values > std_threshold))
        normal_pos_num = normal_mean_max_values.shape[0] - np.sum((normal_mean_max_values > mean_threshold) + (normal_std_max_values > std_threshold)) + normal_pos_base
        total_num = 670
        pos_ratio = (adv_pos_num + normal_pos_num) / total_num
        if pos_ratio > max_one:
            max_one = pos_ratio
            max_idx = (mean_threshold, std_threshold)
            adv_pos_num_t = adv_pos_num
            normal_pos_num_t = normal_pos_num
    # print(threshold, pos_ratio)
print(max_idx, max_one, adv_pos_num_t, normal_pos_num_t)
# print(normal_mean_stds[:50])
# print('----')
# print(adv_mean_stds[:50])


'''
1.24394 0.7443548387096774 <-- max ratio in rgb
'''