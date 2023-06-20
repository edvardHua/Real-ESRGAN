# -*- coding: utf-8 -*-
# @Time : 2023/6/9 09:57
# @Author : zihua.zeng
# @File : degradation_pipeline.py
import os

import yaml
import cv2
import torch
import random
import numpy as np
from pprint import pprint
from torch.nn import functional as F
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_util import tensor2img
from basicsr.utils.img_process_util import filter2D
from basicsr.data.transforms import paired_random_crop
from realesrgan.data.realesrgan_dataset import RealESRGANDataset
from basicsr.utils.options import copy_opt_file, dict2str, ordered_yaml
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt

usm_sharpener = USMSharp()
jpeger = DiffJPEG(differentiable=False)


def degradation_in_dataset(opt):
    ds = RealESRGANDataset(opt)
    return ds


def degradation_in_feed_data(opt, data):
    gt = data['gt']
    # if opt['gt_sum'] is True:
    gt = usm_sharpener(gt)

    kernel1 = data['kernel1']
    kernel2 = data['kernel2']
    sinc_kernel = data['sinc_kernel']
    ori_h, ori_w = gt.size()[2:4]
    # ----------------------- The first degradation process ----------------------- #
    # blur, 这个卷积核来自 dataset
    out = filter2D(gt, kernel1)
    # random resize
    updown_type = random.choices(['up', 'down', 'keep'], opt['resize_prob'])[0]
    if updown_type == 'up':
        scale = np.random.uniform(1, opt['resize_range'][1])
    elif updown_type == 'down':
        scale = np.random.uniform(opt['resize_range'][0], 1)
    else:
        scale = 1

    mode = random.choice(['area', 'bilinear', 'bicubic'])
    out = F.interpolate(out, scale_factor=scale, mode=mode)
    # add noise
    gray_noise_prob = opt['gray_noise_prob']
    if np.random.uniform() < opt['gaussian_noise_prob']:
        out = random_add_gaussian_noise_pt(
            out, sigma_range=opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
    else:
        out = random_add_poisson_noise_pt(
            out,
            scale_range=opt['poisson_scale_range'],
            gray_prob=gray_noise_prob,
            clip=True,
            rounds=False)
    # JPEG compression
    jpeg_p = out.new_zeros(out.size(0)).uniform_(*opt['jpeg_range'])
    out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
    out = jpeger(out, quality=jpeg_p)

    # ----------------------- The second degradation process ----------------------- #
    # blur
    if np.random.uniform() < opt['second_blur_prob']:
        out = filter2D(out, kernel2)
    # random resize
    updown_type = random.choices(['up', 'down', 'keep'], opt['resize_prob2'])[0]
    if updown_type == 'up':
        scale = np.random.uniform(1, opt['resize_range2'][1])
    elif updown_type == 'down':
        scale = np.random.uniform(opt['resize_range2'][0], 1)
    else:
        scale = 1
    mode = random.choice(['area', 'bilinear', 'bicubic'])
    out = F.interpolate(
        out, size=(int(ori_h / opt['scale'] * scale), int(ori_w / opt['scale'] * scale)), mode=mode)
    # add noise
    gray_noise_prob = opt['gray_noise_prob2']
    if np.random.uniform() < opt['gaussian_noise_prob2']:
        out = random_add_gaussian_noise_pt(
            out, sigma_range=opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
    else:
        out = random_add_poisson_noise_pt(
            out,
            scale_range=opt['poisson_scale_range2'],
            gray_prob=gray_noise_prob,
            clip=True,
            rounds=False)

    # JPEG compression + the final sinc filter
    # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
    # as one operation.
    # We consider two orders:
    #   1. [resize back + sinc filter] + JPEG compression
    #   2. JPEG compression + [resize back + sinc filter]
    # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
    if np.random.uniform() < 0.5:
        # resize back + the final sinc filter
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, size=(ori_h // opt['scale'], ori_w // opt['scale']), mode=mode)
        out = filter2D(out, sinc_kernel)
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*opt['jpeg_range2'])
        out = torch.clamp(out, 0, 1)
        out = jpeger(out, quality=jpeg_p)
    else:
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*opt['jpeg_range2'])
        out = torch.clamp(out, 0, 1)
        out = jpeger(out, quality=jpeg_p)
        # resize back + the final sinc filter
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, size=(ori_h // opt['scale'], ori_w // opt['scale']), mode=mode)
        out = filter2D(out, sinc_kernel)

    # clamp and round
    lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

    lq = tensor2img(lq)
    gt = tensor2img(gt)
    vs_img = np.hstack([gt, lq])
    # cv2.imwrite("vs_test.jpg", vs_img)
    return vs_img

    # random crop
    # gt_size = opt['gt_size']
    # gt, lq = paired_random_crop(gt, lq, gt_size, opt['scale'])


def degradation_unit_test(opt, output_path):
    os.makedirs(output_path, exist_ok=True)
    ds = degradation_in_dataset(opt['datasets']['train'])

    for ind in range(len(ds)):
        item = next(iter(ds))
        item['gt'] = torch.unsqueeze(item['gt'], 0)
        out_img = degradation_in_feed_data(opt, item)
        cv2.imwrite(os.path.join(output_path, "%d.jpg" % ind), out_img)


def unit_test1():
    yfile = "options/edz.yml"
    opt = yaml.load(open(yfile, "r"), Loader=ordered_yaml()[0])
    opt['scale'] = 1
    data = degradation_unit_test(opt, "vs_degradation")
    pass


def test_blur_kernel():
    import math
    import random
    from basicsr.data.degradations import bivariate_Gaussian

    # [7, 9, 11, 13, 15, 17, 19, 21]
    kernel_range = [2 * v + 1 for v in range(3, 11)]
    kernel_size = random.choice(kernel_range)
    kernel_list = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
    kernel_prob = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
    blur_sigma = [0.2, 3]
    betag_range = [0.5, 4]
    betap_range = [1, 2]

    # iso
    # kernel = random_bivariate_Gaussian(
    #     kernel_size, blur_sigma, blur_sigma, (-math.pi, math.pi), noise_range=None, isotropic=True)

    path = "/Users/zihua.zeng/Dataset/商品修复/Product_HR/normal-1.jpg"
    image = cv2.imread(path)

    kernel = bivariate_Gaussian(11, 3, 3, )


if __name__ == '__main__':
    test_blur_kernel()
    pass
