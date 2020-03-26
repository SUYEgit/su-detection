# -*- coding: utf-8 -*-
"""
# This module provide coco data analysis tools
# Authors: suye
# Date: 2020/03/25
"""
import os
import argparse
import copy
import shutil

import cv2
from tqdm import tqdm
import json
from pycocotools.coco import COCO
import numpy as np
from icv.data import Coco


class CoCoDataset:
    def __init__(self, json_path, images_path):
        self.json_path = json_path
        self.images_path = images_path
        self.coco = COCO(json_path)

    def _target_exists(self, anns, target_names):
        all_ann_names = [self.coco.cats[ann['category_id']]['name'] for ann in anns]
        for target_name in target_names:
            if target_name in all_ann_names:
                return True
        return False

    def ann_analysis(self, plot_out_path):
        cc = Coco(image_dir=self.images_path,
                  anno_file=self.json_path)

        cc.statistic(print_log=True,
                     is_plot_show=False,
                     plot_save_path=plot_out_path)

    def visualize(self, vis_out_path, with_mask=False, target_names=None):
        """visualize annotations on image"""
        if not os.path.exists(vis_out_path):
            os.mkdir(vis_out_path)

        cat_ids = self.coco.getCatIds()
        image_ids = self.coco.getImgIds()
        print('Visualizing {} images'.format(len(image_ids)))
        for image_id in tqdm(image_ids):
            image = self.coco.loadImgs(image_id)[0]
            image_name = image['file_name']
            image_path = os.path.join(self.images_path, image_name)
            image_np = cv2.imread(image_path)
            image_np_ori = copy.deepcopy(image_np)

            # get anns
            ann_ids = self.coco.getAnnIds(imgIds=image['id'], catIds=cat_ids, iscrowd=None)
            anns = self.coco.loadAnns(ann_ids)
            if not self._target_exists(anns, target_names):
                continue

            for ann in anns:
                bbox = ann['bbox']
                category_id = ann['category_id']
                class_name = self.coco.cats[category_id]['name']
                xmin = bbox[0]
                ymin = bbox[1]
                xmax = bbox[0] + bbox[2]
                ymax = bbox[1] + bbox[3]

                cv2.rectangle(image_np, (int(xmin), int(ymin)), (int(xmax), int(ymax)),
                              [255, 255, 255], 2)

                if with_mask:
                    mask = self.coco.annToMask(ann)
                    mask_bin = mask.astype(np.bool)
                    color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                    image_np[mask_bin] = image_np[mask_bin] * 0.5 + color_mask * 0.5
                cv2.putText(image_np, class_name, (int(xmin), int(ymin)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

            image_out = np.concatenate((image_np_ori, image_np), axis=1)
            cv2.imwrite(os.path.join(vis_out_path, image_name), image_out)

    def remake(self, json_out_path, target_names):
        """remake"""
        if not os.path.exists(json_out_path):
            os.makedirs(json_out_path)
        else:
            shutil.rmtree(json_out_path)
            os.makedirs(json_out_path)

        with open(self.json_path, 'r') as f:
            coco_data = json.load(f)
        anns = coco_data['annotations']
        imgs = coco_data['images']

        class_names = [self.coco.cats[cat_id]['name'] for cat_id in self.coco.cats]
        new_anns = []
        cur_ann_id = 0
        for ann in anns:
            category_id = ann['category_id']
            class_name = class_names[category_id - 1]
            if class_name not in target_names:
                continue
            else:
                ann['id'] = cur_ann_id
                ann['category_id'] = target_names.index(class_name) + 1
                new_anns.append(ann)
                cur_ann_id += 1

        new_categories = []
        for i, name in enumerate(target_names):
            new_categories.append({'id': i + 1, 'name': name, "supercategory": "none"})

        new_json = {'images': imgs,
                    'annotations': new_anns,
                    'categories': new_categories}

        with open(os.path.join(json_out_path, 'instances.json'), 'w') as f:
            f.write(json.dumps(new_json))


if __name__ == '__main__':
    out_path = '/Users/suye02/jingyan2/data/3月1号初版标注数据/cemian/cemian_coco/'
    coco_dataset = CoCoDataset(json_path='/Users/suye02/jingyan2/data/3月1号初版标注数据/cemian/cemian_coco/0307_annotations/instances_val.json',
                               images_path='/Users/suye02/jingyan2/data/3月1号初版标注数据/cemian/cemian_coco/val')
    coco_dataset.ann_analysis(os.path.join(out_path, 'statistics'))
    # coco_dataset.visualize(out_path)
    # coco_dataset.remake(out_path, target_names=['quepenghuashang'])