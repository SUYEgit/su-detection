# -*- coding: utf-8 -*-
"""
# This module provide coco data analysis tools
# Authors: suye
# Date: 2020/03/25
"""
import os
import argparse
import copy
from glob import glob
import random

import cv2
from tqdm import tqdm
import json
from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt

from icv.data import Coco
# from ..trad_algorithm.enhance import ds_um
plt.switch_backend('agg')


class CoCoDataset:
    def __init__(self, json_path, images_path):
        self.json_path = json_path
        self.images_path = images_path
        self.coco = COCO(json_path)
        self.class_names = [cat['name'] for cat in self.coco.cats.values()]

    def _target_exists(self, anns, target_names):
        if target_names is None:
            return True
        all_ann_names = [self.coco.cats[ann['category_id']]['name'] for ann in anns]
        for target_name in target_names:
            if target_name in all_ann_names:
                return True
        return False

    def ann_counts(self):
        ann_ids = self.coco.getAnnIds()
        print('TOTALLY {} ANNS'.format(len(ann_ids)))
        anns = self.coco.loadAnns(ann_ids)

        counts_dict = {}
        for ann in anns:
            gt_label = ann['category_id']
            gt_name = self.class_names[gt_label - 1]
            if gt_name in counts_dict:
                counts_dict[gt_name] += 1
            else:
                counts_dict[gt_name] = 1

        print('ANN COUNTS: {}'.format(counts_dict))

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

    def remake_by_num(self, json_out_path, ann_nums, assured_ids=(4, 5, 6)):
        """remake dataset by set numbers"""
        assert len(self.class_names) == len(ann_nums)

        new_ann_ids = []
        # 针对精研正的特殊逻辑
        # 最笨的办法：json load  然后shuffle + 遍历ann，累加新ann的同时计数，够了就不加了
        for i, ann_num in enumerate(ann_nums):
            cat_id = i + 1
            ann_ids = self.coco.getAnnIds(catIds=cat_id)
            print('cat_id', cat_id, 'ann_num', ann_num)
            print('before filter', len(ann_ids))
            new_ann_id = []
            for j, ann_id in enumerate(ann_ids):
                ann = self.coco.loadAnns(ids=ann_id)[0]
                image_id = int(ann['image_id'])
                oth_ann_ids = self.coco.getAnnIds(imgIds=[image_id])
                oth_anns = self.coco.loadAnns(ids=oth_ann_ids)
                oth_cats = [int(x['category_id']) for x in oth_anns]
                assert ann['category_id'] in oth_cats
                if cat_id not in assured_ids and list(set(assured_ids).intersection(set(oth_cats))):
                    #                     ann_ids.remove(ann_id)
                    continue
                elif cat_id in assured_ids:
                    three_count = 0
                    for oth_cat in oth_cats:
                        if oth_cat in assured_ids:
                            three_count += 1
                    if three_count > 1:
                        continue
                new_ann_id.append(ann_id)

            print('after filter', len(new_ann_id))
            new_ann_ids.extend(random.sample(new_ann_id, ann_num))

        random.shuffle(new_ann_ids)

        new_json = {'images': [],
                    'annotations': [],
                    'categories': [x for x in self.coco.cats.values()]}

        finished_imgs = []
        new_img_id = 0
        new_id = 0
        coco_copy = COCO(self.json_path)
        for i, new_ann_id in enumerate(new_ann_ids):
            ann = coco_copy.loadAnns(ids=[new_ann_id])[0]
            first_id = ann['category_id']
            image_id = int(ann['image_id'])
            if image_id in finished_imgs:
                continue
            else:
                finished_imgs.append(image_id)

            try:
                image = coco_copy.loadImgs(ids=[image_id])[0]
            except Exception:
                print('image id {} not found in json.'.format(image_id))
                continue

            image['id'] = new_img_id
            new_json['images'].append(image)
            # 一个标注被选中了 整个图片的标注都应该被制作进来 否则就等于漏标了
            new_anns = coco_copy.loadAnns(ids=coco_copy.getAnnIds(imgIds=image_id))
            new_anns_cp = copy.deepcopy(new_anns)
            all_ids = [int(x['category_id']) for x in new_anns]

            if first_id not in all_ids:
                print('WRONG image_id:{} anno_ids:{} ann id {} first_id {} all_ids {}'.format(image_id,
                                                                                              coco_copy.getAnnIds(
                                                                                                  imgIds=image_id),
                                                                                              fuck_ann_id, first_id,
                                                                                              all_ids))

            for new_ann in new_anns_cp:
                new_ann['image_id'] = new_img_id
                new_ann['id'] = new_id
                new_json['annotations'].append(new_ann)
                new_id += 1

            new_img_id += 1

        with open(json_out_path, 'w') as f:
            f.write(json.dumps(new_json))
        print('Remaked Dataset {} {}'.format(self.class_names, ann_nums))
        new_coco_dataset = CoCoDataset(json_out_path,
                                       images_path=args.images_path)
        new_coco_dataset.ann_counts()
        new_coco_dataset.ann_analysis(plot_out_path=os.path.join(analysis_out_path, 'statistic'))

    def remake_by_name(self, json_out_path, target_names):
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

        with open(os.path.join(json_out_path, 'val_withbg.json'), 'w') as f:
            f.write(json.dumps(new_json))

    def extract_cat_imgs(self, target_id, img_out_path):
        if not os.path.exists(img_out_path):
            os.mkdir(img_out_path)
        image_ids = self.coco.getImgIds()
        print('-----Extracting cat:{} images-----'.format(target_id))
        target_imgs = []
        untarget_imgs = []
        for image_id in tqdm(image_ids):
            image = self.coco.loadImgs(image_id)[0]
            image_name = image['file_name']
            image_path = os.path.join(self.images_path, image_name)
            print(image_path)
            # get anns
            ann_ids = self.coco.getAnnIds(imgIds=image['id'], iscrowd=None)
            anns = self.coco.loadAnns(ann_ids)
            ann_cats = [ann['category_id'] for ann in anns]
            print('cats', ann_cats)
            if target_id in ann_cats:
                target_imgs.append(image_path)
            else:
                untarget_imgs.append(image_path)
        random.shuffle(untarget_imgs)
        untarget_imgs = random.sample(untarget_imgs, len(target_imgs))
        print('GET {} target imgs and bg imgs'.format(len(untarget_imgs)))

        for target_img in target_imgs:
            target_path = os.path.join(img_out_path, self.class_names[target_id - 1])
            if not os.path.exists(target_path):
                os.mkdir(target_path)

            img_np = cv2.imread(target_img, 0)
            img_np = ds_um(img_np, radius=5, amount=10, ratio=0.0625)
            cv2.imwrite(os.path.join(target_path, os.path.basename(target_img)), img_np)

        for untarget_img in untarget_imgs:
            target_path = os.path.join(img_out_path, 'ok')
            if not os.path.exists(target_path):
                os.mkdir(target_path)

            img_np = cv2.imread(untarget_img, 0)
            img_np = ds_um(img_np, radius=5, amount=10, ratio=0.0625)
            cv2.imwrite(os.path.join(target_path, os.path.basename(untarget_img)), img_np)


if __name__ == '__main__':
    json_path_df = '/Users/suye02/LIANFA/coco/annotations/train.json'
    images_path_df = '/Users/suye02/LIANFA/coco/train'
    out_path_df = '/Users/suye02/LIANFA/coco'

    parser = argparse.ArgumentParser(description='COCO Data Analysis Tool')
    parser.add_argument('--json_path', type=str, default=json_path_df)
    parser.add_argument('--images_path', type=str, default=images_path_df)
    parser.add_argument('--out_path', type=str, default=out_path_df)

    args = parser.parse_args()

    coco_dataset = CoCoDataset(json_path=args.json_path,
                               images_path=args.images_path)

    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)
    analysis_out_path = os.path.join(args.out_path, 'analysis')
    if not os.path.exists(analysis_out_path):
        os.mkdir(analysis_out_path)

    # coco_dataset.ann_counts()
    #     coco_dataset.remake_by_num(json_out_path=os.path.join(out_path_df, 'train.json'),
    #                                ann_nums=(200, 3, 190))
    coco_dataset.ann_analysis(plot_out_path=os.path.join(analysis_out_path, 'statistic'))
    # coco_dataset.visualize(os.path.join(analysis_out_path, 'visualize'))
    # coco_dataset.remake(json_out_path='/Users/suye02/jingyan2/data/3月1号初版标注数据/cemian/cemian_coco/0306_annotations/',
    #                     target_names=['quepenghuashang'])
    # coco_dataset.merge_background_img(bg_imgs_path='/root/suye/jingyan2_data/0420_bx_merge/pure_lp_imgs',
    #                                   json_out_path='/root/suye/jingyan2_data/0420_bx_merge/annotations/')



