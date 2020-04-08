# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
"""
# This module provide
# Authors: suye02(suye02@baidu.com)
# Date: 2019/10/22 10:04 下午
"""
import json
import os
import argparse

import cv2
import numpy as np
import xmltodict
import pypinyin
from tqdm import tqdm


def hanzi_to_pinyin(word):
    """hanzi_to_pinyin"""
    s = ''
    for i in pypinyin.pinyin(word, style=pypinyin.NORMAL):
        s += ''.join(i)
    return s


def load_json(xml_path):
    """
    获取xml文件
    :param xml_path:
    :return:
    """
    xml_file = open(xml_path, 'r', encoding='utf-8')
    xml_str = xml_file.read()
    xmlparse = xmltodict.parse(xml_str)
    jsonstr = json.dumps(xmlparse, indent=1)
    jsonstr = json.loads(jsonstr)
    return jsonstr


def get_id_from_categories(name, categories):
    """

    :param name:
    :param categories:
    :return:
    """
    for cat in categories:
        if cat['name'] == name:
            return cat['id']
    category = {
        'id': len(categories) + 1,
        'name': name,
        'supercategory': 'none'
    }
    categories.append(category)
    return len(categories)


def add_new_annotation(new_annotations, polygon, new_images_idx, new_anno_idx, id):
    """

    :param new_annotations:
    :param polygon:
    :param new_images_idx:
    :param new_anno_idx:
    :param id:
    :return:
    """
    polygon_np = np.array(polygon).reshape([-1, 2])
    x, y, w, h = cv2.boundingRect(polygon_np)

    new_annotation = {
        'segmentation': polygon,
        'bbox': [x, y, w, h],
        'area': cv2.contourArea(polygon_np),
        'iscrowd': 0,
        'image_id': new_images_idx,
        'id': new_anno_idx,
        'category_id': id,
    }
    new_annotations.append(new_annotation)


def weiyi_xml_to_coco(xml_dir, img_dir, coco_path):
    """

    :param xml_dir:
    :param img_dir:
    :param coco_path:
    :return:
    """
    new_images, new_annotations = [], []
    new_images_idx, new_anno_idx = 0, 0

    cats = {}
    for xml_name in tqdm(os.listdir(xml_dir)):
        if not xml_name.endswith('.xml'):
            continue
        xml_path = os.path.join(xml_dir, xml_name)
        img_path = os.path.join(img_dir, xml_name.replace('.xml', '.jpg'))
        json_str = load_json(xml_path)

        if not os.path.exists(img_path):
            # print('NO IMAGE')
            continue
        if json_str is None or json_str['doc']['outputs']['object'] is None:
            # print('NONE')
            continue

        img_np = cv2.imread(img_path, 0)
        new_image = {
            'file_name': xml_name.replace('.xml', '.jpg'),
            'height': img_np.shape[0],
            'width': img_np.shape[1],
            'id': new_images_idx,
        }
        new_images.append(new_image)
        items = json_str['doc']['outputs']['object']['item']

        if type(items) == dict:
            items = [items]

        for item in items:
            name = item['name']
            if name not in cats.keys():
                cats[name] = 1
            else:
                cats[name] += 1

            if name in filtered_names:
                continue
            id = name_id_map[name]

            name = hanzi_to_pinyin(name)

            if 'polygon' in item.keys() and item['polygon']:
                polygon = [[int(_) for _ in item['polygon'].values()]]

                if len(polygon[0]) <= 4:
                    continue

                # polygon_np = np.array(polygon).reshape([-1, 2])
                add_new_annotation(new_annotations, polygon, new_images_idx, new_anno_idx, id)
                new_anno_idx += 1

            elif 'line' in item.keys() and item['line']:

                mask_np = np.zeros(img_np.shape, dtype=np.uint8)
                d = int(item['width'])

                points = [[int(_) for _ in item['line'].values()]]

                points_np = np.array(points).reshape([-1, 2])
                for i in range(len(points_np) - 1):
                    cv2.line(mask_np, tuple(points_np[i]), tuple(points_np[i + 1]),
                             color=(255, 255, 255), thickness=max(3, d))

                contours, hierarchy = cv2.findContours(mask_np, cv2.RETR_EXTERNAL,
                                                       cv2.CHAIN_APPROX_SIMPLE)
                # if len(contours) == 0:
                #     print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>{}  no contours'.format(name))
                for index, cnt in enumerate(contours):
                    segmentation = cnt.flatten()
                    segmentation = [[int(_) for _ in list(segmentation)]]
                    add_new_annotation(new_annotations, segmentation, new_images_idx, new_anno_idx,
                                       id)
                    new_anno_idx += 1

            elif 'point' in item.keys() and item['point']:

                mask_np = np.zeros(img_np.shape, dtype=np.uint8)
                d = int(item['width'])

                points = [int(_) for _ in item['point'].values()]
                points_np = np.array(points).reshape([-1, 2])
                cv2.circle(mask_np, tuple(points_np[0]), max(3, d // 2), color=(255, 255, 255),
                           thickness=-1)
                contours, hierarchy = cv2.findContours(mask_np, cv2.RETR_EXTERNAL,
                                                       cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) == 0:
                    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>{}  no contours'.format(name))
                for index, cnt in enumerate(contours):
                    segmentation = cnt.flatten()
                    segmentation = [[int(_) for _ in list(segmentation)]]

                    add_new_annotation(new_annotations, segmentation, new_images_idx, new_anno_idx,
                                       id)
                    new_anno_idx += 1

            elif 'bndbox' in item.keys() and item['bndbox']:
                bndbox = [int(_) for _ in item['bndbox'].values()]
                xmin = bndbox[0]
                ymin = bndbox[1]
                xmax = bndbox[2]
                ymax = bndbox[3]

                assert xmin < xmax and ymin < ymax

                bndbox_np = [[xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]]

                add_new_annotation(new_annotations, bndbox_np, new_images_idx, new_anno_idx, id)
                new_anno_idx += 1
            else:
                print('ELSE')
                print(item)
        new_images_idx += 1

    new_json = {
        'images': new_images,
        'annotations': new_annotations,
        'categories': categories,
    }
    with open(coco_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(new_json))


def get_cats():
    cats_dict = {}
    for xml_name in os.listdir(xml_dir):
        if not xml_name.endswith('.xml'):
            continue
        xml_path = os.path.join(xml_dir, xml_name)
        img_path = os.path.join(img_dir, xml_name.replace('.xml', '.jpg'))
        json_str = load_json(xml_path)

        if not os.path.exists(img_path):
            continue
        if json_str is None or json_str['doc']['outputs']['object'] is None:
            continue

        items = json_str['doc']['outputs']['object']['item']

        if type(items) == dict:
            items = [items]

        for item in items:
            name = item['name']
            if name not in cats_dict.keys():
                cats_dict[name] = 1
            else:
                cats_dict[name] += 1
    print(cats_dict)

    return list(cats_dict.keys())


if __name__ == '__main__':
    data_types = ['train', 'val']
    base_path_df = '/Users/suye02/MRST/data/新机台/0326三臂机数据'
    out_path_df = os.path.join(base_path_df, 'annotations')

    parser = argparse.ArgumentParser(description='weiyi xml to coco')
    parser.add_argument('--base_path', type=str, default=base_path_df)
    parser.add_argument('--out_path', type=str, default=out_path_df)

    args = parser.parse_args()

    for data_type in data_types:
        print('____________MAKING {} DATA TO COCO_________________'.format(data_type))
        xml_dir = os.path.join(args.base_path, 'xmls')
        img_dir = os.path.join(args.base_path, data_type)
        coco_path = os.path.join(args.out_path, 'instances_{}.json'.format(data_type))

        filtered_names = []
        if data_type == 'train':
            cats = get_cats()
        name_id_map = {x: cats.index(x) + 1 for x in cats}

        categories = []
        for i, cat in enumerate(cats):
            alp_name = hanzi_to_pinyin(cat)
            category = {"id": i + 1, "name": alp_name, "supercategory": "none"}
            categories.append(category)

        if not os.path.exists(args.out_path):
            os.mkdir(args.out_path)

        weiyi_xml_to_coco(xml_dir, img_dir, coco_path)
        print(name_id_map)
        print(categories)
