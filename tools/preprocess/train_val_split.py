# -*- coding: utf-8 -*-
"""
# This module provide utils for spliting train val
# Authors: suye
# Date: 2020/03/02 10:04 下午
"""
import os
from glob import glob
import random
import shutil


def make_val_imgs(all_path, val_ratio=8):
    """所有数据放入训练集后，复制x分之一制作测试集"""
    val_path = all_path + '_val'
    if not os.path.exists(val_path):
        os.mkdir(val_path)

    train_imgs = glob(all_path + '/*.jpg')

    test_num = int(len(train_imgs) / val_ratio)
    print('train num:', len(train_imgs) - test_num)
    print('test num', test_num)
    random.shuffle(train_imgs)
    test_imgs = train_imgs[:test_num]
    for test_img in test_imgs:
        shutil.move(test_img, os.path.join(val_path, os.path.basename(test_img)))

    generate_txt_list(all_path, 'train')
    generate_txt_list(val_path, 'val')


def generate_txt_list(images_path, split='train', ext='jpg'):
    """generate_txt_list"""
    images = glob(os.path.join(images_path, '*.{}'.format(ext)))
    for image in images:
        image_name = os.path.basename(image).replace(ext, '')
        txt_name = os.path.join(images_path, '{}.txt'.format(split))
        with open(txt_name, 'a') as f:
            f.write(image_name + '\n')


def coco_train_val_split(json_path):
    """coco_train_val_split"""
    # TODO
    pass


if __name__ == '__main__':
    imgs_path = '/Users/suye02/jingyan2/data/3月1号初版标注数据/dibu/imgs'
    make_val_imgs(imgs_path)
