# -*- coding: utf-8 -*-
"""
# This module provide feature parameterization over segmentation results.
# Authors: suye
# Date: 2019/11/05 1:02 下午
"""

import os
import argparse
import copy

import cv2
from tqdm import tqdm
import json
from pycocotools.coco import COCO
import numpy as np


def visualize(images_path, json_path, out_path, with_mask=False):
    """visualize annotations on image"""
    coco = COCO(json_path)

    with open(json_path, 'r') as f:
        categories = json.load(f)['categories']
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    cat_ids = coco.getCatIds()
    image_ids = coco.getImgIds()
    print('Visualizing {} images'.format(len(image_ids)))
    for image_id in tqdm(image_ids):
        image = coco.loadImgs(image_id)[0]
        image_name = image['file_name']
        image_path = os.path.join(images_path, image_name)
        image_np = cv2.imread(image_path)
        image_np_ori = copy.deepcopy(image_np)
        # print('visualizing {}'.format(image_path))

        # get anns
        ann_ids = coco.getAnnIds(imgIds=image['id'], catIds=cat_ids, iscrowd=None)
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            bbox = ann['bbox']
            category_id = ann['category_id']
            class_name = categories[category_id - 1]['name']
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = bbox[0] + bbox[2]
            ymax = bbox[1] + bbox[3]

            cv2.rectangle(image_np, (int(xmin), int(ymin)), (int(xmax), int(ymax)),
                          [255, 255, 255], 2)

            if with_mask:
                mask = coco.annToMask(ann)
                mask_bin = mask.astype(np.bool)
                color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                image_np[mask_bin] = image_np[mask_bin] * 0.5 + color_mask * 0.5
            cv2.putText(image_np, class_name, (int(xmin), int(ymin)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

        image_out = np.concatenate((image_np_ori, image_np), axis=1)
        cv2.imwrite(os.path.join(out_path, image_name), image_out)


if __name__ == '__main__':
    images_path_df = '/Users/suye02/copy/youji_data/验收训练/mmnet/ceguai/JPEGImages'
    json_path_df = '/Users/suye02/copy/youji_data/验收训练/mmnet/ceguai/train.json'
    out_path_df = '/Users/suye02/copy/youji_data/验收训练/mmnet/ceguai/JPEGImages_vis'

    parser = argparse.ArgumentParser(description='This script support visualizing coco annotations')
    parser.add_argument('--images_path', type=str, default=images_path_df, help='path to images directory')
    parser.add_argument('--json_path', type=str, default=json_path_df, help='path to annotation files directory')
    parser.add_argument('--out_path', type=str, default=out_path_df, help='path to visualization output directory')

    args = parser.parse_args()
    visualize(images_path=args.images_path,
              json_path=args.json_path,
              out_path=args.out_path)