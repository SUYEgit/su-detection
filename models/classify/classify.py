# -*- coding: utf-8 -*-
"""
# Two stage classification.
# Authors: suye02
# Date: 2019/11/13 1:02 下午
"""
from math import exp

import cv2
import pretrainedmodels
import pretrainedmodels.utils
import torch
from PIL import Image


def softmax(logits, ng_node_id):
    """softmax score"""
    molecule = 0
    for i in range(len(logits)):
        molecule += exp(float(logits[i]))
    score = exp(float(logits[ng_node_id])) / molecule
    return score


def initialize_pretrained_model(model, settings):
    """initialize_pretrained_model"""
    model.load_state_dict(torch.load(settings['model_path']))
    model.input_space = settings['input_space']
    model.input_size = settings['input_size']
    model.input_range = settings['input_range']
    model.mean = settings['mean']
    model.std = settings['std']


def load_cls_model(num_classes=2,
                   cls_model_path='/home/adt/project-microsoft/deployment/model_store/'
                                  'damian/classify/best_net_params.pkl',
                   input_size=1000,
                   net='resnet18'):
    """加载分类模型"""
    model_path = cls_model_path
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


def correct_byio(image_np):
    """correct_byio"""
    tmp_path = '/home/adt/project-microsoft/cls_tmp.jpg'
    cv2.imwrite(tmp_path, image_np)
    return cv2.imread(tmp_path)


def run_classify_inference(model, image, thresh=0.8):
    """run_classify_inference"""
    if len(image.shape) < 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    tf_img = pretrainedmodels.utils.TransformImage(model)

    image = correct_byio(image)
    input_img = Image.fromarray(image, mode='RGB')

    input_tensor = tf_img(input_img)
    input_tensor = input_tensor.unsqueeze(0)
    input_f = torch.autograd.Variable(input_tensor, requires_grad=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_f = input_f.to(device)
    output_logits = model(input_f)
    score = softmax(output_logits[0], ng_node_id=0)
    return score > thresh


def get_crop(results, image_np, target_label):
    """
    :return:
    """
    crop_nps, indexes = [], []
    for i in range(len(results.bboxes)):
        if results.labels[i] != target_label:
            continue
        else:
            xmin, ymin, xmax, ymax = results.bboxes[i][0], results.bboxes[i][1], \
                                     results.bboxes[i][2], results.bboxes[i][3]
            crop_np = image_np[int(ymin):int(ymax), int(xmin):int(xmax)]
            crop_nps.append(crop_np)
            indexes.append(i)
    return crop_nps, indexes
