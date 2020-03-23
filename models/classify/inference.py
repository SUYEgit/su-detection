# -*- coding: utf-8 -*-
"""

# Two stage classification.
# Authors: suye02
# Date: 2019/11/13 1:02 下午
"""
from __future__ import print_function
from __future__ import division
import os
from PIL import Image
from glob import glob
from math import exp

import torch
import numpy as np
import pretrainedmodels
import pretrainedmodels.utils
import cv2
import matplotlib.pyplot as plt
from skimage.filters import unsharp_mask

plt.switch_backend('agg')


class LoadImage(object):

    def __init__(self, space='RGB'):
        self.space = space

    def __call__(self, path_img):
        with open(path_img, 'rb') as f:
            with Image.open(f) as img:
                img = img.convert(self.space)
        return img


def unsharp(img_np, radius=5, amount=10):
    """unsharp"""
    # img_np_gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

    img_np_gb = cv2.GaussianBlur(img_np, (3, 3), 0)
    img_np_um = unsharp_mask(img_np_gb, radius=radius, amount=amount).astype(np.float32) * 255

    return img_np_um


def initialize_pretrained_model(model, settings):
    """initialize model with settings"""

    model.input_space = settings['input_space']
    model.input_size = settings['input_size']
    model.input_range = settings['input_range']
    model.mean = settings['mean']
    model.std = settings['std']
    model.load_state_dict(torch.load(settings['model_path']))


def load_model(model_path, num_classes, input_size):
    """Load model"""

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    settings = {
        'input_space': 'RGB',
        'input_size': [3, input_size, input_size],
        'input_range': [0, 1],
        'mean': [0.53337324, 0.53337324, 0.53337324],
        'std': [0.07127615, 0.07127615, 0.07127615],
        'num_classes': num_classes,
        'model_path': '{}'.format(model_path)
    }

    model = pretrainedmodels.__dict__[net](num_classes=num_classes, pretrained=None)
    model.last_linear = torch.nn.Linear(512, num_classes)
    initialize_pretrained_model(model, settings)
    model.to(device)
    model.eval()

    return model


def softmax(logits, ng_node_id):
    """softmax"""
    molecule = 0
    for i in range(len(logits)):
        molecule += exp(float(logits[i]))

    score = exp(float(logits[ng_node_id])) / molecule
    return score


def inference(image_path,
              model_path,
              input_size,
              num_classes):
    """inference"""

    scores = []

    model = load_model(model_path, num_classes=num_classes, input_size=input_size)

    load_img = pretrainedmodels.utils.LoadImage()
    tf_img = pretrainedmodels.utils.TransformImage(model)

    images = glob(os.path.join(image_path, '*.jpg'))
    for image in images:
        print(image)
        image_np = cv2.imread(image)
        input_img = Image.fromarray(image_np, mode='RGB')
        #         input_img = load_img(image)

        input_tensor = tf_img(input_img)
        input_tensor = input_tensor.unsqueeze(0)
        input_f = torch.autograd.Variable(input_tensor, requires_grad=False)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        input_f = input_f.to(device)

        output_logits = model(input_f)

        proba = True
        if proba:
            score = softmax(output_logits[0], ng_node_id=0)
        else:
            output_logits.detach_()
            _, predicted = torch.max(output_logits.data, 1)
            score = 1 - predicted[0]
        scores.append(score)
    return scores, images


def binary_pr_eval(model_path, ok_image_path, ng_image_path, input_size, num_classes=2, grains=20, draw_fp=False):
    """二分类evaluation"""

    ok_scores, ok_images = inference(image_path=ok_image_path,
                                     model_path=os.path.join(model_path, 'best_net_params.pkl'),
                                     input_size=input_size,
                                     num_classes=num_classes)
    ng_scores, ng_images = inference(image_path=ng_image_path,
                                     model_path=os.path.join(model_path, 'best_net_params.pkl'),
                                     input_size=input_size,
                                     num_classes=num_classes)

    ok_scores.sort()
    ng_scores.sort()

    recalls = []
    precisions = []
    fps = []
    end_thr = 1
    for thr in [x / grains for x in list(range(0, grains, 1))]:
        if (len([x for x in ng_scores if x > thr]) + len([x for x in ok_scores if x > thr])) == 0:
            end_thr = thr
            break

        recall = len([x for x in ng_scores if x > thr]) / len(ng_scores)
        fp = len([x for x in ok_scores if x > thr]) / len(ok_scores)
        precision = len([x for x in ng_scores if x > thr]) / (len([x for x in ng_scores if x > thr]) + len(
            [x for x in ok_scores if x > thr]))
        recalls.append(recall)
        precisions.append(precision)
        fps.append(fp)
    print(precisions, recalls, fps)

    plt.xlim((0, 1))
    plt.ylim((0, 1))

    if draw_fp:
        plt.plot(recalls, fps, c='red')
        plt.grid(True)
        plt.legend(loc='upper left')
        plt.xlabel("recall")
        plt.ylabel("false positive")
        for i, thr in enumerate([x / grains for x in list(range(0, grains, 1))]):
            if thr >= float(end_thr):
                break
            plt.annotate(round(thr, 2), (recalls[i], fps[i]))
    else:
        plt.plot(recalls, precisions, c='red')
        plt.grid(True)
        plt.legend(loc='upper left')
        plt.xlabel("recall")
        plt.ylabel("precision")
        for i, thr in enumerate([x / grains for x in list(range(0, grains, 1))]):
            if thr >= float(end_thr):
                break
            plt.annotate(round(thr, 2), (recalls[i], precisions[i]))

    plt.savefig("pr_curve.png")


if __name__ == '__main__':
    binary_pr_eval(model_path='/root/suye/classify/models/best_daowen',
                   ok_image_path='/root/suye/classify/data/daowen_um/val/ok',
                   ng_image_path='/root/suye/classify/data/daowen_um/val/ng')
