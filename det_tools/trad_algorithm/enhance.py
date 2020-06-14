# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
"""
# This module provide
# Authors: suye02(suye02@baidu.com)
# Date: 2019/10/22 10:04 下午
"""
import os
from glob import glob
from tqdm import tqdm
import numpy as np
import cv2
from skimage.filters import unsharp_mask
import matplotlib.pyplot as plt


def unsharp(img_np, radius=5, amount=2):
    """unsharp"""
    # img_np_gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

    img_np_gb = cv2.GaussianBlur(img_np, (3, 3), 0)
    img_np_um = unsharp_mask(img_np_gb, radius=radius, amount=amount).astype(np.float32) * 255

    return img_np_um.astype(np.uint8)


def ds_um(img_np, radius, amount, ratio):
    height, width = img_np.shape[0], img_np.shape[1]
    img_np_ds = cv2.resize(img_np,
                           (int(width * ratio), int(height * ratio)),
                           interpolation=cv2.INTER_AREA)
    img_np_ds_um = unsharp(img_np_ds, radius=radius, amount=amount)
    return img_np_ds_um


def downsample_dir():
    """downsample_dir"""
    img_path = '/root/suye/classify/0322_data'
    out_path = img_path + '_dsum'
    ratio = 0.0625
    radius = 5
    amount = 10

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    imgs = glob(os.path.join(img_path, '*.jpg'))
    for img in tqdm(imgs):
        img_np = cv2.imread(img, 0)
        height, width = img_np.shape[0], img_np.shape[1]
        img_np_ds = cv2.resize(img_np,
                               (int(width * ratio), int(height * ratio)),
                               interpolation=cv2.INTER_AREA)
        img_np_ds_um = unsharp(img_np_ds, radius=radius, amount=amount)
        img_out = np.concatenate((img_np_ds, img_np_ds_um), axis=1)
        cv2.imwrite(os.path.join(out_path, os.path.basename(img)), img_np_ds_um)


def unsharp_fine_tune():
    """unsharp mask 精调实验"""
    img_path = '/Users/suye02/MRST/测试集/128/psbj/0021-0016-13.jpg'
    # img_path = '/Users/suye02/MRST/data/liangpin/not_psbj/1013-186-1.jpg'

    ratio = 0.0625
    img_np = cv2.imread(img_path, 0)
    plt.figure()

    height, width = img_np.shape[0], img_np.shape[1]
    img_np_ds = cv2.resize(img_np,
                           (int(width * ratio), int(height * ratio)),
                           interpolation=cv2.INTER_AREA)

    radius = (2, 5, 10, 20)
    amount = (2, 10, 25, 50)
    count = 1
    for r in tqdm(radius):
        for a in amount:
            img_np_ds_um = unsharp(img_np_ds, radius=r, amount=a)
            plt.subplot(4, 4, count)
            plt.imshow(img_np_ds_um, cmap='gray')
            plt.axis('off')
            plt.title('radius {} amount {}'.format(r, a))
            count += 1

    plt.show()


if __name__ == '__main__':
    downsample_dir()
