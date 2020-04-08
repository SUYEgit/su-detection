# -*- coding: utf-8 -*-
"""
# This module provide cutting utils for coco dataset.
# Authors: suye
# Date: 2020/03/20 3:14 pm
"""
import json
import os
import shutil
import time

from tqdm import tqdm
import cv2
import numpy as np
from pycocotools.coco import COCO


def refresh_dir(path):
    """refresh dir"""
    if not os.path.exists(path):
        os.makedirs(path)
    # else:
    #     shutil.rmtree(path)
    #     os.makedirs(path)


class CocoCutter:
    """cut coco annotations"""

    def __init__(self, imgs_path, json_path, out_path, data_type='train'):
        self.imgs_path = imgs_path
        self.json_path = json_path
        self.out_path = out_path
        self.imgs_out_path = os.path.join(self.out_path, data_type)
        self.data_type = data_type

        refresh_dir(self.out_path)
        os.mkdir(self.imgs_out_path)

        self.coco = COCO(json_path)
        with open(self.json_path, 'r') as f:
            self.categories = json.load(f)['categories']

        self.new_image_idx = 0
        self.new_annotation_idx = 0
        self.new_json = {'images': [],
                         'annotations': [],
                         'categories': self.categories}

    def add_new_annotation(self, polygon, cat_id):
        """

        :param polygon:
        :param cat_id:
        :return:
        """
        if len(polygon) != 1:
            polygon = [polygon]
        polygon_np = np.array(polygon).reshape([-1, 2])
        x, y, w, h = cv2.boundingRect(polygon_np)
        new_annotation = {
            'segmentation': polygon,
            'bbox': [x, y, w, h],
            'area': cv2.contourArea(polygon_np),
            'iscrowd': 0,
            'image_id': self.new_image_idx,
            'id': self.new_annotation_idx,
            'category_id': cat_id,
        }
        self.new_json['annotations'].append(new_annotation)

    def crop_coco_annotation(self, crop_mask_np, cat_id):
        """

        :param crop_mask_np:
        :param cat_id:
        :return:
        """
        if np.count_nonzero(crop_mask_np) == 0:
            return False

        flag = False
        contours, hierarchy = cv2.findContours(crop_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for index, cnt in enumerate(contours):
            segmentation = cnt.flatten()
            segmentation = [[int(_) for _ in list(segmentation)]]
            if len(segmentation[0]) <= 4:
                continue
            self.add_new_annotation(segmentation, cat_id)
            self.new_annotation_idx += 1
            flag = True
        return flag

    def _cut_by_region(self, image_name, anns, overlap, ignore_empty):
        """

        :param image_name:
        :param anns:
        :param overlap:


        :return:
        """
        image_path = os.path.join(self.imgs_path, image_name)
        if not os.path.exists(image_path):
            return
        img_np = cv2.imread(image_path, 0)
        height, width = img_np.shape

        x_offset, y_offset = 0, 0

        width_crop_num = (width - overlap) // (self.crop_width - overlap) + 1
        height_crop_num = (height - overlap) // (self.crop_height - overlap) + 1

        for i in range(width_crop_num):
            y_offset = 0
            for j in range(height_crop_num):
                if len(anns) == 0:
                    continue
                new_image_name = image_name.replace('.jpg', '_{}_{}.jpg'.format(i, j))
                flag = False
                for k in range(len(anns)):
                    mask_np = np.zeros([height, width], dtype=np.uint8)
                    cat_id = anns[k]['category_id']
                    segment = anns[k]['segmentation'][0]
                    segment_np = np.array(segment).reshape([-1, 2])
                    cv2.fillPoly(mask_np, pts=[segment_np], color=(255, 255, 255))
                    crop_mask_np = mask_np[y_offset: y_offset + self.crop_height, x_offset: x_offset + self.crop_width]
                    if np.count_nonzero(crop_mask_np) == 0:
                        continue
                    flag = self.crop_coco_annotation(crop_mask_np, cat_id)  # flag表示是否切到了有效标注

                if flag or not ignore_empty:
                    crop_img_np = img_np[y_offset: y_offset + self.crop_height, x_offset: x_offset + self.crop_width]

                    self.new_json['images'].append({
                        'file_name': new_image_name,
                        'height': crop_img_np.shape[0],
                        'width': crop_img_np.shape[1],
                        'id': self.new_image_idx,
                    })
                    self.new_image_idx += 1
                    cv2.imwrite(os.path.join(self.imgs_out_path, new_image_name), crop_img_np)
                y_offset += self.crop_height - overlap
            x_offset += self.crop_width - overlap

    def cut_by_region(self, crop_width, crop_height, overlap, ignore_empty=True):
        """cut by region"""
        image_ids = self.coco.getImgIds()
        catIds = self.coco.getCatIds()
        self.crop_width = crop_width
        self.crop_height = crop_height

        for i in range(len(image_ids)):
            image = self.coco.loadImgs(image_ids[i])[0]
            image_name = image['file_name']
            print(image_name)

            ann_ids = self.coco.getAnnIds(imgIds=image['id'], catIds=catIds, iscrowd=None)
            anns = self.coco.loadAnns(ann_ids)
            self._cut_by_region(image_name, anns, overlap, ignore_empty)

        with open(os.path.join(self.out_path, 'instances_{}.json'.format(self.data_type)), 'w') as f:
            f.write(json.dumps(self.new_json))

    def cut_by_gt(self, crop_width, crop_height):
        """以每个gt为中心，固定宽高切图"""
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        annotations = data['annotations']
        images = data['images']

        ann_lookup = {}
        for annotation in annotations:
            image_id = annotation['image_id']
            if image_id not in ann_lookup.keys():
                ann_lookup[image_id] = [annotation]
            else:
                ann_lookup[image_id].append(annotation)
        #         print('ann_lookup', [len(ann_lookup[key]) for key in ann_lookup.keys()])

        cat_ids = self.coco.getCatIds()
        print('cat_ids', cat_ids)
        for i, image in tqdm(enumerate(images)):
            start = time.time()
            img_name = image['file_name']
            img_id = image['id']
            img_np = cv2.imread(os.path.join(self.imgs_path, img_name))
            img_height, img_width = img_np.shape[0], img_np.shape[1]

            # 从coco获取图片对应的ann
            ann_ids = self.coco.getAnnIds(imgIds=image['id'], catIds=cat_ids, iscrowd=None)
            anns = self.coco.loadAnns(ann_ids)

            # 初始化mask 及bbox segm
            masks = []
            cut_centers = []
            cat_ids = []
            for j, annotation in enumerate(anns):
                image_id = annotation['image_id']
                if image_id != img_id:
                    continue
                else:
                    single_mask = self.coco.annToMask(annotation)
                    poly_points = annotation['segmentation'][0]
                    xs = [poly_points[i] for i in range(len(poly_points)) if i % 2 == 0]
                    ys = [poly_points[i] for i in range(len(poly_points)) if i % 2 != 0]
                    xmin, xmax = min(xs), max(xs)
                    ymin, ymax = min(ys), max(ys)

                    # 记录切图中心点及标注的mask
                    cut_centers.append(((xmin + xmax) // 2, (ymax + ymin) // 2))
                    masks.append(single_mask)
                    cat_ids.append(annotation['category_id'])

            # 切图并根据mask提取segmentation
            for cut_center in cut_centers:
                xmin = max(0, cut_center[0] - (crop_width // 2))
                xmax = xmin + crop_width
                if xmax >= img_width:
                    xmin = img_width - crop_width
                    xmax = xmin + crop_width

                ymin = max(0, cut_center[1] - (crop_height // 2))
                ymax = ymin + crop_height
                if ymax >= img_height:
                    ymin = img_height - crop_height
                    ymax = ymin + crop_height

                # 切图并保存
                img_crop = img_np[ymin:ymax, xmin:xmax]
                img_new_name = img_name.replace('.jpg', '{}_{}_{}_{}.jpg'.format(xmin, ymin, xmax, ymax))

                cv2.imwrite(os.path.join(self.imgs_out_path, img_new_name), img_crop)
                new_image = {
                    'file_name': img_new_name,
                    'height': crop_height,
                    'width': crop_width,
                    'id': self.new_image_idx,
                }
                self.new_json['images'].append(new_image)

                # 添加新annotations
                for j, mask in enumerate(masks):
                    mask_crop = mask[ymin:ymax, xmin:xmax]
                    contours, hierarchy = cv2.findContours(mask_crop, cv2.RETR_EXTERNAL,
                                                           cv2.CHAIN_APPROX_SIMPLE)
                    for index, cnt in enumerate(contours):
                        segmentation = cnt.flatten()
                        segmentation = [[int(_) for _ in list(segmentation)]]
                        if len(segmentation[0]) <= 4:
                            continue
                        self.add_new_annotation(segmentation, cat_ids[j])
                        self.new_annotation_idx += 1

                self.new_image_idx += 1
            print(img_name, 'seg time {}s cuts{}'.format(time.time() - start, len(cut_centers)))

        with open(os.path.join(self.out_path, 'instances_{}.json'.format(self.data_type)), 'w') as f:
            f.write(json.dumps(self.new_json))


if __name__ == '__main__':
    coco_cutter = CocoCutter(imgs_path='/Users/suye02/su-detection/data/HUAWEI/data/训练数据/侧面_sorted/train',
                             json_path='/Users/suye02/su-detection/data/HUAWEI/data/训练数据/侧面_sorted/annotations/instances_train.json',
                             out_path='/Users/suye02/su-detection/data/HUAWEI/data/训练数据/侧面_sorted/cbr',
                             data_type='train')
    coco_cutter.cut_by_region(crop_width=400,
                              crop_height=1000,
                              overlap=10,
                              ignore_empty=True)
    # coco_cutter.cut_by_gt(crop_width=512,
    #                       crop_height=512)
