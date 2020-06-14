# -*- coding: utf-8 -*-
"""
Inference tools for mmdetection and paddle
Authors: suye
Date:    2020/03/24 19:26:26
"""
from glob import glob
import os
import time
import copy
import shutil

import cv2
import mmcv
import numpy as np
from mmdet.apis import inference_detector, init_detector
import pycocotools.mask as maskUtils


def compute_iou(bbox1, bbox2):
    """
    compute iou
    :param bbox1:
    :param bbox2:
    :return: iou
    """
    # TODO 重写
    bbox1xmin = bbox1[0]
    bbox1ymin = bbox1[1]
    bbox1xmax = bbox1[2]
    bbox1ymax = bbox1[3]
    bbox2xmin = bbox2[0]
    bbox2ymin = bbox2[1]
    bbox2xmax = bbox2[2]
    bbox2ymax = bbox2[3]
    area1 = (bbox1ymax - bbox1ymin) * (bbox1xmax - bbox1xmin)
    area2 = (bbox2ymax - bbox2ymin) * (bbox2xmax - bbox2xmin)
    bboxxmin = max(bbox1xmin, bbox2xmin)
    bboxxmax = min(bbox1xmax, bbox2xmax)
    bboxymin = max(bbox1ymin, bbox2ymin)
    bboxymax = min(bbox1ymax, bbox2ymax)
    if bboxxmin >= bboxxmax:
        return 0
    if bboxymin >= bboxymax:
        return 0
    area = (bboxymax - bboxymin) * (bboxxmax - bboxxmin)
    iou = area / (area1 + area2 - area)
    return iou


def NMSBbox(bboxes, labels):
    """

    :param bboxes:
    :param labels:
    :return:
    """
    vis = np.zeros(len(bboxes))
    rmpos = []
    for p in range(len(bboxes)):
        if vis[p]:
            continue
        vis[p] = 1

        for q in range(len(bboxes) - p - 1):
            if vis[q + p + 1]:
                continue
            bbox1 = bboxes[p]
            bbox2 = bboxes[q + p + 1]
            if compute_iou(bbox1, bbox2) > 0.2:
                if bboxes[p + q + 1][4] > bboxes[p][4]:
                    rmpos.append(p)
                    break
                else:
                    rmpos.append(q + p + 1)
                    vis[q + p + 1] = 1
    rmpos.sort(reverse=True)
    for p in rmpos:
        bboxes.pop(p)
        labels.pop(p)
    return bboxes, labels


def mmdet_inference(model_path, config_path, images_path, vis_path=None, thresh=0.5,
                    cut=False, cut_width=0, cut_height=0, overlap=0):
    if vis_path is not None:
        if not os.path.exists(vis_path):
            os.makedirs(vis_path)
        else:
            shutil.rmtree(vis_path)
            os.makedirs(vis_path)
    print('INITIALIZING DETECTORS...')
    model = init_detector(config_path, model_path)

    images = glob(os.path.join(images_path, '*.jpg'))
    total_times = 0
    for i, image in enumerate(images):
        print('cut: {} inferencing {}'.format(cut, image))
        img = cv2.imread(image)
        height, width = img.shape[0], img.shape[1]
        cut_width = width // 2 + 50
        cut_height = height // 2 + 50
        if cut:
            assert cut_width != 0 and cut_height != 0
            height_cut_num = (height - overlap) // (cut_height - overlap) + 1
            width_cut_num = (width - overlap) // (cut_width - overlap) + 1
            print('height_cut_num {} width_cut_num {}'.format(height_cut_num, width_cut_num))
        else:
            cut_width, cut_height, overlap = width, height, 0
            height_cut_num = 1
            width_cut_num = 1

        start_x = 0
        start_y = 0
        total_boxes, total_labels = [], []
        masks = []
        for y_index in range(height_cut_num):
            end_y = min(start_y + cut_height, height)
            if end_y - start_y < cut_height:
                start_y = end_y - cut_height

            for x_index in range(width_cut_num):
                cut_masks = []
                end_x = min(start_x + cut_width, width)
                if end_x - start_x < cut_width:
                    start_x = end_x - cut_width

                sub_img = img[start_y: end_y + 1, start_x: end_x + 1]
                start_time = time.time()
                result = inference_detector(model, sub_img)
                end_time = time.time()
                inf_time = end_time - start_time
                print('pure inference time: {}'.format(inf_time))
                total_times += inf_time

                if isinstance(result, tuple):
                    bbox_result, segm_result = result
                else:
                    bbox_result, segm_result = result, None
                # draw bounding boxes
                labels = [
                    np.full(bbox.shape[0], i, dtype=np.int32)
                    for i, bbox in enumerate(bbox_result)
                ]
                labels = np.concatenate(labels)
                # get bbox and score
                bboxes = np.vstack(bbox_result)

                for bbox in bboxes:
                    bbox[0] += start_x
                    bbox[1] += start_y
                    bbox[2] += start_x
                    bbox[3] += start_y
                    total_boxes.append(bbox)
                for label in labels:
                    total_labels.append(label)

                if segm_result is not None:
                    segms = mmcv.concat_list(segm_result)
                    for k in range(len(segms)):
                        mask = maskUtils.decode(segms[k])
                        masks.append(mask)
                    assert len(total_boxes) == len(masks)
                    assert len(total_labels) == len(masks)
                start_x = start_x + cut_width - overlap

            start_x = 0
            start_y = start_y + cut_height - overlap

        #####

        # NMS
        total_boxes, total_labels = NMSBbox(total_boxes, total_labels)

        total_boxes = np.array(total_boxes)
        total_labels = np.array(total_labels)

        write_inference_results(image, i, total_boxes, total_labels)

        if vis_path is not None:
            out_path = os.path.join(vis_path, os.path.basename(image))
            visualize(total_labels, total_boxes, img, out_path, thresh, masks)
    print('>>>>>>>>AVERAGE TIME {}'.format(total_times / len(images)))


def visualize(labels, bboxes, img_np, out_path, thresh, masks=None, concat=False):
    """visualize gt and det"""
    if concat:
        img_np_ori = copy.deepcopy(img_np)

    # 画模型检测框
    print('VISUALIZING DETECTIONS...')
    for i, label in enumerate(labels):
        bbox = bboxes[i]
        if bbox[-1] < thresh:
            continue
        left_top = (int(bbox[0]), int(bbox[1]))
        right_bottom = (int(bbox[2]), int(bbox[3]))
        cv2.rectangle(img_np, left_top, right_bottom, (0, 255, 0), 2)
        # put text
        label_text = 'DET {} | {:.02f}'.format(class_names[label], bbox[-1])
        cv2.putText(img_np, label_text, (int(bbox[0]), int(bbox[1]) - 2),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0))
        if masks:
            mask = masks[i]
            mask_bin = mask.astype(np.bool)
            color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            img_np[mask_bin] = img_np[mask_bin] * 0.5 + color_mask * 0.5

    if concat:
        img_out = np.concatenate((img_np_ori, img_np), axis=1)
        cv2.imwrite(out_path, img_out)

    else:
        cv2.imwrite(out_path, img_np)


def write_inference_results(image, image_id, bboxes, labels):
    # TODO 有需要时再写（转成coco标准json结果文件）
    return


if __name__ == '__main__':
    class_names = ['heidian', 'aokeng', 'guashang', 'daowen', 'penshabujun', 'yise', 'cashang', 'shuiyin', 'shahenyin',
                   'aotuhen']
    mmdet_inference(
        model_path='/root/suye/mmdetection-1.0rc0/train_outputs/0401_jy2_dazheng_ga_faster_r50_caffe_fpn_1x/latest.pth',
        config_path='/root/suye/mmdetection-1.0rc0/train_outputs/0401_jy2_dazheng_ga_faster_r50_caffe_fpn_1x/ga_faster_r50_caffe_fpn_1x.py',
        images_path='/root/suye/jingyan2_data/dazheng/images/val',
        vis_path=None,
        thresh=0.5,
        cut=False, cut_width=0, cut_height=0, overlap=0)
