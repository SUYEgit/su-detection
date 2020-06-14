#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
合并COCO结果文件
Date:    2020/3/3 下午7:04
"""
import json
import os


def merge_res_json(path1, path2, out_path):
    with open(path1, 'r') as f:
        coco_data1 = json.load(f)
    with open(path2, 'r') as f:
        coco_data2 = json.load(f)

    coco_data_out = coco_data1 + coco_data2

    with open(os.path.join(out_path, 'merged_results.pkl.bbox.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(coco_data_out))


if __name__ == '__main__':
    merge_res_json(
        path1='/root/suye/mmdetection-1.0rc0/train_outputs/0108_regioncut_bd_faster_rcnn_dconv_c3-c5_r50_fpn_1x/0326data_bd_eval_analysis/results.pkl.bbox.json',
        path2='/root/suye/mmdetection-1.0rc0/train_outputs/0220_sd5_512regioncut_cascade_mask_rcnn_r18_fpn_1x/0326data_sd_eval_analysis/results.pkl.bbox.json',
        out_path='/root/suye/mmdetection-1.0rc0/train_outputs/0220_sd5_512regioncut_cascade_mask_rcnn_r18_fpn_1x/0326data_sd_eval_analysis')
