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
from glob import glob

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
        if target_names is None:
            return True
        all_ann_names = [self.coco.cats[ann['category_id']]['name'] for ann in anns]
        for target_name in target_names:
            if target_name in all_ann_names:
                return True
        return False

    def ann_analysis(self, plot_out_path):
        print('-----analyzing coco data and plotting-----')
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
        print('-----Visualizing {} images-----'.format(len(image_ids)))
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
        print('-----remaking coco dataset with target names {} -----'.format(target_names))
        if not os.path.exists(json_out_path):
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

        with open(os.path.join(json_out_path, 'instances_remake.json'), 'w') as f:
            f.write(json.dumps(new_json))

    def merge_background_img(self, bg_imgs_path, json_out_path):
        print('-----merging background images into coco dataset-----')
        bg_imgs = glob(os.path.join(bg_imgs_path, '*.jpg'))
        print('-----get {} background images-----'.format(len(bg_imgs)))
        with open(self.json_path, 'r') as f:
            coco_data = json.load(f)
        imgs = coco_data['images']
        print('-----old coco {} images-----'.format(len(imgs)))
        for i, bg_img in enumerate(bg_imgs):
            bg_img_np = cv2.imread(bg_img)
            new_image = {"file_name": os.path.basename(bg_img),
                         "height": bg_img_np.shape[0],
                         "width": bg_img_np.shape[1],
                         "id": len(imgs) + i}
            imgs.append(new_image)
        print('-----new coco {} images-----'.format(len(imgs)))
        new_json = {'images': imgs,
                    'annotations': coco_data['annotations'],
                    'categories': coco_data['categories']}

        with open(os.path.join(json_out_path, 'train_withbg.json'), 'w') as f:
            f.write(json.dumps(new_json))


if __name__ == '__main__':
    json_path_df = '/Users/suye02/jingyan2/data/3月1号初版标注数据/cemian/cemian_coco/0306_annotations/instances_val.json'
    images_path_df = '/Users/suye02/jingyan2/data/3月1号初版标注数据/cemian/cemian_coco/val'
    out_path_df = '/Users/suye02/jingyan2/data/3月1号初版标注数据/cemian/cemian_coco/'

    parser = argparse.ArgumentParser(description='COCO Data Analysis Tool')
    parser.add_argument('--json_path', type=str, default=json_path_df)
    parser.add_argument('--images_path', type=str, default=images_path_df)
    parser.add_argument('--out_path', type=str, default=out_path_df)

    args = parser.parse_args()

    coco_dataset = CoCoDataset(json_path=args.json_path,
                               images_path=args.images_path)

    analysis_out_path = os.path.join(args.out_path, 'analysis')
    if not os.path.exists(analysis_out_path):
        os.mkdir(analysis_out_path)

    coco_dataset.ann_analysis(plot_out_path=os.path.join(analysis_out_path, 'statistic'))
    # coco_dataset.visualize(os.path.join(analysis_out_path, 'visualize'))
    # coco_dataset.remake(json_out_path='/Users/suye02/jingyan2/data/3月1号初版标注数据/cemian/cemian_coco/0306_annotations/',
    #                     target_names=['quepenghuashang'])
    # coco_dataset.merge_background_img(bg_imgs_path='/Users/suye02/jingyan2/data/3月1号初版标注数据/cemian/cemian_coco/train',
    #                                   json_out_path='/Users/suye02/jingyan2/data/3月1号初版标注数据/cemian/cemian_coco/0307_annotations/')
