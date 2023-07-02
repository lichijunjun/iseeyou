import os

Shadow_ROOT = '/mnt/data/approx-killer/ShadowAttack-master'
class_n_lisa, class_n_gtsrb = 16, 43
Shadow_adv_samples_path = \
    lambda x:f'/mnt/data/approx-killer/ShadowAttack-master/adv_img/{x.upper()}/43'
Shadow_normal_samples_path = \
    lambda x:os.path.join(Shadow_ROOT, f'./dataset/{x.upper()}/test.pkl') 
Shadow_config_path = '/home/kemove/approx-killer/src/attacks/shadow/params.json'

AdvLS_ROOT = '/mnt/data/approx-killer/AdvLS-main'
AdvLS_adv_samples_path = '/mnt/data/approx-killer/AdvLS-main/results_20_80' 
AdvLS_adv_samples_std_path = '/mnt/data/approx-killer/Advlight-main/AdvLS-main/results1_std'
AdvLS_adv_samples_neurips_std_path = '/mnt/data/approx-killer/Advlight-main/AdvLS-main/results2_1_50'
AdvLS_normal_samples_path = '/mnt/data/approx-killer/Advlight-main/query_imagenet' 
AdvLS_normal_samples_neurips_path = '/mnt/data/approx-killer/optical_adversarial_attack-main/archive/images'
AdvLS_data_dict_path = "/mnt/data/approx-killer/approx-killer/assets/advls_true_label_dict.pth"
AdvLS_data_dict_std_path = "/mnt/data/approx-killer/approx-killer/assets/advls_true_label_std_dict.pth"

AdvLB_ROOT = '/mnt/data/approx-killer/Advlight-main'
AdvLB_adv_samples_path = '/mnt/data/approx-killer/Advlight-main/results'
AdvLB_adv_samples_neurips_path = '/mnt/data/approx-killer/Advlight-main/results2'
AdvLB_adv_samples_std_path = '/mnt/data/approx-killer/Advlight-main/results'
AdvLB_adv_samples_neurips_std_path = '/mnt/data/approx-killer/Advlight-main/results2'
AdvLB_normal_samples_path = '/mnt/data/approx-killer/Advlight-main/query_imagenet'
AdvLB_normal_samples_neurips_path = '/mnt/data/approx-killer/optical_adversarial_attack-main/archive/images'

Optical_ROOT = '/mnt/data/approx-killer/optical_adversarial_attack-main'
Optical_adv_samples_path = '/mnt/data/approx-killer/Advlight-main/optical_adversarial_attack-main/result/simulation/Resnet50/image2'
Optical_adv_samples_imagenet_path = '/mnt/data/approx-killer/optical_adversarial_attack-main/result/simulation/Resnet50/image1'
Optical_normal_samples_path = '/mnt/data/approx-killer/optical_adversarial_attack-main/archive/images'
Optical_normal_samples_imagenet_path = '/mnt/data/approx-killer/Advlight-main/query_imagenet'
Optical_data_csv_path = "/mnt/data/approx-killer/Advlight-main/optical_adversarial_attack-main/experiment/simulation/images.csv"

CUDA_DEVICES = [2, 7]
  

attack_goals = {
    "advlb":"untargeted",
    "advls":"untargeted",
    "optical":"untargeted",
    "shadow":"untargeted"
}

benchmark_data_root = '/mnt/data/approx-killer/approx-killer/assets/benchmark_data'