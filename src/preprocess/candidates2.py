from functools import partial

import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter
from scipy import ndimage
from torch.nn import Module
from torchvision import transforms

from .defense import (defend_BitReduct, defend_FD_sig, defend_FD_sig_s,
                      defend_FixedJpeg, defend_gd, defend_gd_s,
                      defend_onlyrand, defend_onlyrand_s, defend_pd,
                      defend_rescale, defend_rescale_s, defend_shield,
                      defend_TotalVarience)


def _lut(image, lut):
    if image.mode == "P":
        # FIXME: apply to lookup table, not image data
        msg = "mode P support coming soon"
        raise NotImplementedError(msg)
    elif image.mode in ("L", "RGB"):
        if image.mode == "RGB" and len(lut) == 256:
            lut = lut + lut + lut
        return image.point(lut)
    else:
        msg = "not supported for this image mode"
        raise OSError(msg)

class MySolarize(torch.nn.Module):

    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def forward(self, img):
        
        lut = []
        for i in range(256):
            if i < self.threshold:
                lut.append(i)
            else:
                lut.append(0)
        return _lut(img, lut)


    def __repr__(self):
        return self.__class__.__name__ + '(threshold={})'.format(self.threshold)

class MyFilter(torch.nn.Module):

    def __init__(self, threshold=80):
        super().__init__()
        self.threshold = threshold
        
    def forward(self, img):
        img = np.array(img)
        avg = np.mean(img, axis=2)
        zoe = np.where(avg < 255 - avg, avg, 255 - avg)
        jar = np.where(zoe < self.threshold, 255, 0)
        img = np.repeat(jar[:, :, np.newaxis].astype(np.uint8), 3, axis=2)

        return Image.fromarray(img)

    def __repr__(self):
        return self.__class__.__name__ + '(threshold={})'.format(self.threshold)

def median_filter(image, win=3):
    return image.filter(ImageFilter.MedianFilter(win))

def smooth_filter(image):
    return image.filter((ImageFilter.ModeFilter(5)))


def save_to_trash(img):
    # print(img)
    if isinstance(img,  np.ndarray):
        x = Image.fromarray(img.astype(np.uint8))
    else: x = img
    # print(img.shape)
    # x = Image.fromarray(img.astype(np.uint8))
    x.save("/mnt/data/approx-killer/approx-killer/trash/look_after.jpg")
    # print(list(img.getdata())[224 * idx:448 * (idx + 1)])
    return img

def save_to_trashx(img):
    # print(img.shape)
    
    img.save("/mnt/data/approx-killer/approx-killer/trash/look_before.jpg")
    # print(list(img.getdata())[:224])
    return img

def save_to_trash_v2(img):
    # print(img)
    if isinstance(img,  np.ndarray):
        x = Image.fromarray(img.astype(np.uint8))
    else: x = img
    # print(img.shape)
    # x = Image.fromarray(img.astype(np.uint8))
    x.save("/mnt/data/approx-killer/approx-killer/trash/look_after.jpg")
    # print(list(img.getdata())[224 * idx:448 * (idx + 1)])
    return img

def save_to_trashx_v2(img):
    # print(img.shape)
    x = Image.fromarray(img.astype(np.uint8))
    x.save("/mnt/data/approx-killer/approx-killer/trash/look_before.jpg")
    # print(list(img.getdata())[:224])
    return img

def pre_process_image_gtsrb_part1(image):
    image = np.array(image)
    image[:, :, 0] = cv2.equalizeHist(image[:, :, 0])
    image[:, :, 1] = cv2.equalizeHist(image[:, :, 1])
    image[:, :, 2] = cv2.equalizeHist(image[:, :, 2])
    return image

def pre_process_image_gtsrb_part2(image):
    image = image / 255. - .5
    return image.astype(np.float32)

def Empty(image):
    return image

# defense-end transformation
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
def custom_preprocess(can_list):
    to_np = [np.array]
    size_scale = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    trans_list1 = [
        get_trans(trans_type, 0) for trans_type in can_list
    ]
    trans_list2 = [
        get_trans(trans_type, 1) for trans_type in can_list
    ]
    std_trans_list_imagenet = [
        transforms.ToTensor(),
        transforms.Normalize(mean = mean,std = std)
    ]
    std_trans_list_lowresolution = [
        transforms.ToTensor()
    ]
    std_trans_list_lowresolution2_part1 = [
        pre_process_image_gtsrb_part1,
    ]
    std_trans_list_lowresolution2_part2 = [
        pre_process_image_gtsrb_part2,
        transforms.ToTensor()
    ]
    return (
        transforms.Compose(size_scale + trans_list1 + std_trans_list_imagenet),
        # transforms.Compose(size_scale + [save_to_trashx] + trans_list1 + [save_to_trash] + std_trans_list_imagenet),
        transforms.Compose([save_to_trashx] + trans_list2 + [save_to_trash] + std_trans_list_lowresolution),        
        transforms.Compose(to_np + std_trans_list_lowresolution2_part1 + trans_list2 + std_trans_list_lowresolution2_part2),        
    )
   

    
def only_manual_preprocess(can_list):
    size_scale = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    trans_list1 = [
        get_trans(trans_type, 0) for trans_type in can_list
    ]
    trans_list2 = [
        get_trans(trans_type, 1) for trans_type in can_list
    ]
    return (
        transforms.Compose(size_scale + trans_list1),
        transforms.Compose(trans_list2),        
        transforms.Compose(trans_list2),        
    )
    
def standard_preprocess():
    return transforms.Compose([ 
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean = mean,std = std)
        ]
    )


class AdvLS_process(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.cur_process1 = custom_preprocess_v(["autocontrast", "rescale_s1"])[0]
        self.cur_process2 = custom_preprocess_v(["autocontrast", "rescale_s2"])[0]
        self.filter =  MyFilter(84)
        self.mean_filter = partial(median_filter, win=3)

    def forward(self, img):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img.astype(np.uint8))
        data = self.cur_process1(img) # ae 
        datap = self.cur_process2(img) # rescaled ae
        datax = self.mean_filter(data) # filter ae ()
        mt1 = np.array(data)
        mt2 = np.array(datax)
        mask = 255 - np.dstack([np.abs(np.average(np.array(self.filter(data)), axis=2).astype(np.uint8))] * 3)
        filtered_res = np.where(mask < 128, mt2, mt1)
        mask = Image.fromarray(filtered_res)

        return mask

def cv2_threshold(img):
    return cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)[1]


transforms_dict = {
    "colorjitter": transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0, hue=0.5),
    "rescale_s1": (partial(defend_rescale, size=224), defend_rescale_s, defend_rescale_s),
    "rescale_s2": (partial(defend_rescale, size=42), defend_rescale_s, defend_rescale_s),
    "posterize": transforms.RandomPosterize(8, p=1),
    "solarize": transforms.RandomSolarize(232, p=1),
    "mysolarize": MySolarize(250),
    "grayscale":transforms.Grayscale(1),
    "adjustsharpness": transforms.RandomAdjustSharpness(1000, p=1),
    "autocontrast":transforms.RandomAutocontrast(p=1),
    "equalize":transforms.RandomEqualize(p=1),
    "cv2equalize":cv2.equalizeHist,
    "filter":MyFilter(84),
    "medianfilter":partial(median_filter, win=7),
    "gaussianblur":partial(cv2.GaussianBlur, ksize=(9, 9), sigmaX=0),
    "cvtcolor": partial(cv2.cvtColor, code=cv2.COLOR_BGR2GRAY),
    "threshold": cv2_threshold,
    "smoothfilter":smooth_filter,
    "rand":(defend_onlyrand, defend_onlyrand_s),
    "fd":(defend_FD_sig, defend_FD_sig_s),
    "rdg":(defend_gd, defend_gd_s),
    "pd":defend_pd,
    "shield":defend_shield,
    "bitreduct":defend_BitReduct,
    "fixedjpeg":defend_FixedJpeg,
    "totalvarience":defend_TotalVarience
}

def get_trans(option, pos):
    if isinstance(option, Module):
        return option
    else:
        transes = transforms_dict[option]
        if isinstance(transes, tuple):
            return transes[pos]
        else:
            return transes
        
        
def custom_preprocess_v(can_list):
    to_np = [np.array]
    size_scale = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    trans_list1 = [
        get_trans(trans_type, 0) for trans_type in can_list
    ]
    trans_list2 = [
        get_trans(trans_type, 1) for trans_type in can_list
    ]
    std_trans_list_imagenet = [
        transforms.ToTensor(),
        transforms.Normalize(mean = mean,std = std)
    ]
    std_trans_list_lowresolution = [
        transforms.ToTensor()
    ]
    std_trans_list_lowresolution2_part1 = [
        pre_process_image_gtsrb_part1,
    ]
    std_trans_list_lowresolution2_part2 = [
        pre_process_image_gtsrb_part2,
        transforms.ToTensor()
    ]
    return (
        # transforms.Compose(size_scale + trans_list1 + std_trans_list_imagenet),
        transforms.Compose( trans_list1),
        transforms.Compose([save_to_trashx_v2] + trans_list2 + [save_to_trash_v2]),        
        transforms.Compose(to_np + std_trans_list_lowresolution2_part1 + trans_list2 + std_trans_list_lowresolution2_part2),        
    ) 
    
process_dict = {
    "advls":AdvLS_process(),
}

def get_process(atk_type):
    if atk_type == 'advls':
        return custom_preprocess([process_dict["advls"]])