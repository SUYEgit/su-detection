# -*- coding: utf-8 -*-
"""
Utils for drawing confusion matrix by results json and coco annotations.
Authors: suye
Date:    2020/03/24 19:26:26
"""
import os
import random
import copy
import argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

plt.switch_backend('agg')


def plot_confmat(confusion_mat, classes, set_name='', out_path='./'):
    """plot confusion matrix"""
    # 归一化
    plt.clf()
    confusion_mat_N = confusion_mat.copy()
    for i in range(len(classes)):
        confusion_mat_N[i, :] = confusion_mat[i, :] / confusion_mat[i, :].sum()

    # 获取颜色
    cmap = plt.cm.get_cmap('Greys')  # 更多颜色: http://matplotlib.org/examples/color/colormaps_reference.html
    plt.imshow(confusion_mat_N, cmap=cmap)
    plt.colorbar()

    # 设置文字
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, list(classes), rotation=60)
    plt.yticks(xlocations, list(classes))
    plt.xlabel('Predict label')
    plt.ylabel('True label')
    plt.title('Confusion_Matrix_' + set_name)

    # 打印数字
    for i in range(confusion_mat_N.shape[0]):
        for j in range(confusion_mat_N.shape[1]):
            plt.text(x=j, y=i, s=int(confusion_mat[i, j]), va='center', ha='center', color='red', fontsize=10)
    # 保存
    plt.savefig(os.path.join(out_path, 'Confusion_Matrix_' + set_name + '.png'))
    plt.close()


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
    Non-maximum-Compression
    """
    # TODO 重写
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


class ConfMat:
    def __init__(self, ann_json, res_json, out_path, thr, iou_thr):
        self.ann_coco = COCO(ann_json)
        self.res_coco = self.ann_coco.loadRes(res_json)
        self.cat_ids = self.ann_coco.getCatIds()
        self.class_names = [cat['name'] for cat in self.ann_coco.cats.values()]
        self.class_names.append('bg')
        self.confmat = np.zeros((len(self.class_names), len(self.class_names)))
        self.confmat_ids = {'{}{}'.format(i, j): [] for i in range(len(self.class_names)) for j in
                            range(len(self.class_names))}
        self.thr = thr
        self.iou_thr = iou_thr
        self.out_path = out_path
        if not os.path.exists(out_path):
            os.mkdir(out_path)

    def _load_ann_by_id(self, image_id):
        image = self.ann_coco.loadImgs(image_id)[0]
        ann_ids = self.ann_coco.getAnnIds(imgIds=image['id'], catIds=self.cat_ids, iscrowd=None)
        annotations = self.ann_coco.loadAnns(ann_ids)
        return annotations

    def _load_results_by_id(self, image_id):
        """load results and thresh"""
        res_ids = self.res_coco.getAnnIds(imgIds=image_id, catIds=self.cat_ids, iscrowd=None)
        if len(res_ids) == 0:
            # 完全无检出的情况处理
            return [], []
        else:
            results = self.res_coco.loadAnns(res_ids)

            res_labels, res_bboxes = [], []
            for result in results:
                if float(result['score']) < self.thr:
                    continue
                else:
                    res_labels.append(result['category_id'])
                    res_bboxes.append((result['bbox'][0],
                                       result['bbox'][1],
                                       result['bbox'][0] + result['bbox'][2],
                                       result['bbox'][1] + result['bbox'][3],
                                       result['score']))
            return res_labels, res_bboxes

    def compute_confmat(self, annotations, res_labels, res_bboxes, image_id):
        """累计混淆矩阵"""
        for i, ann in enumerate(annotations):
            gt_label = ann['category_id']
            #             print('GT +1')
            gt_bbox = ann['bbox']
            xmin = int(gt_bbox[0])
            ymin = int(gt_bbox[1])
            xmax = int(gt_bbox[0]) + int(gt_bbox[2])
            ymax = int(gt_bbox[1]) + int(gt_bbox[3])

            matched_labels = []
            for j in range(len(res_labels)):
                if compute_iou((xmin, ymin, xmax, ymax), res_bboxes[j]) > self.iou_thr:
                    matched_labels.append(res_labels[j])
            if len(matched_labels) > 0:
                #                 print("DET IS TP +1")
                if gt_label in matched_labels:
                    # 框对且类别正确
                    self.confmat[gt_label - 1, gt_label - 1] += 1
                    self.confmat_ids['{}{}'.format(gt_label - 1, gt_label - 1)].append(image_id)

                else:
                    # 框对但类别错误，随机选一个label
                    det_label = matched_labels[random.randint(0, len(matched_labels) - 1)]
                    self.confmat[gt_label - 1, det_label - 1] += 1
                    self.confmat_ids['{}{}'.format(gt_label - 1, det_label - 1)].append(image_id)
            else:
                # 漏检
                self.confmat[gt_label - 1, -1] += 1
                self.confmat_ids['{}{}'.format(gt_label - 1, len(self.class_names) - 1)].append(image_id)

        for i in range(len(res_bboxes)):
            for j in range(len(annotations)):
                ann = annotations[j]
                gt_bbox = ann['bbox']
                xmin = int(gt_bbox[0])
                ymin = int(gt_bbox[1])
                xmax = int(gt_bbox[0]) + int(gt_bbox[2])
                ymax = int(gt_bbox[1]) + int(gt_bbox[3])
                if compute_iou((xmin, ymin, xmax, ymax), res_bboxes[i]) > self.iou_thr:
                    break
                if j == len(annotations) - 1:
                    # 所有ann都未与bbox匹配，确定此bbox为过杀
                    #                     print("DET IS FP +1")
                    self.confmat[-1, res_labels[i] - 1] += 1
                    self.confmat_ids['{}{}'.format(len(self.class_names) - 1, res_labels[i] - 1)].append(image_id)

        return self.confmat

    def _visualize(self, annotations, labels, bboxes, img_np, out_name, concat=True):
        """visualize gt and det"""
        if concat:
            img_np_ori = copy.deepcopy(img_np)

        # 画模型检测框
        for i, label in enumerate(labels):
            bbox = bboxes[i]
            left_top = (int(bbox[0]), int(bbox[1]))
            right_bottom = (int(bbox[2]), int(bbox[3]))
            cv2.rectangle(img_np, left_top, right_bottom, (0, 255, 0), 2)
            # put text
            label_text = 'DET {} | {:.02f}'.format(self.class_names[label - 1], bbox[-1])
            cv2.putText(img_np, label_text, (int(bbox[0]), int(bbox[1]) - 2),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0))

        # 画标注框
        for ann in annotations:
            bbox = ann['bbox']
            xmin = int(bbox[0])
            ymin = int(bbox[1])
            xmax = int(bbox[0]) + int(bbox[2])
            ymax = int(bbox[1]) + int(bbox[3])
            cv2.rectangle(img_np, (xmin, ymin), (xmax, ymax), (255, 255, 255), 2)
            # put text
            label_text = 'GT {}'.format(self.class_names[ann['category_id'] - 1])
            cv2.putText(img_np, label_text, (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255))
        if concat:
            img_out = np.concatenate((img_np_ori, img_np), axis=1)
            cv2.imwrite(out_name, img_out)
        else:
            cv2.imwrite(out_name, img_np)

    def visualize_by_index(self, image_path, gt_id=None, res_id=None):
        """
        index_in_confmat
        note: ids start from 1
        """

        if gt_id is None or res_id is None:
            target_ids = self.ann_coco.getImgIds()  # 可视化所有检测结果
            vis_path = os.path.join(self.out_path, 'visualize')
        else:
            target_ids = self.confmat_ids['{}{}'.format(gt_id - 1, res_id - 1)]
            vis_path = os.path.join(self.out_path, '{}_clsed_to_{}'.format(self.class_names[gt_id - 1],
                                                                           self.class_names[res_id - 1]))
        if not os.path.exists(vis_path):
            os.mkdir(vis_path)

        for target_id in target_ids:
            image = self.ann_coco.loadImgs(target_id)[0]
            img_np = cv2.imread(os.path.join(image_path, image['file_name']))

            annotations = self._load_ann_by_id(target_id)
            labels, bboxes = self._load_results_by_id(target_id)
            self._visualize(annotations, labels, bboxes, img_np, os.path.join(vis_path, image['file_name']))

    def run(self):
        image_ids = self.ann_coco.getImgIds()
        for image_id in image_ids:
            annotations = self._load_ann_by_id(image_id)
            labels, bboxes = self._load_results_by_id(image_id)
            self.compute_confmat(annotations, labels, bboxes, image_id)

        plot_confmat(self.confmat, self.class_names, out_path=self.out_path)


if __name__ == '__main__':
    ann_json_df = '/root/suye/PaddleDetection/jingyan_data/cemian/annotations/instances_val.json'
    res_json_df = '/root/suye/PaddleDetection/paddle_outputs/0305_cemian_mask_rcnn_hrnetv2p_w18_1x/bbox.json'
    out_path_df = './'
    thr_df = 0.5
    iou_thr_df = 0.0001
    image_path_df = '/root/suye/PaddleDetection/jingyan_data/cemian/val'

    parser = argparse.ArgumentParser(description='This script support confusion matrix utils.')
    parser.add_argument('--ann_json', type=str, default=ann_json_df)
    parser.add_argument('--res_json', type=str, default=res_json_df)
    parser.add_argument('--out_path', type=str, default=out_path_df)
    parser.add_argument('--thresh', type=str, default=thr_df)
    parser.add_argument('--iou_thr', type=str, default=iou_thr_df)
    parser.add_argument('--image_path', type=str, default=None)

    args = parser.parse_args()
    print(args.ann_json)
    confusion = ConfMat(ann_json=args.ann_json,
                        res_json=args.res_json,
                        out_path=args.out_path,
                        thr=float(args.thresh),
                        iou_thr=float(args.iou_thr))
    confusion.run()
    if args.image_path is not None:
        confusion.visualize_by_index(args.image_path, gt_id=None, res_id=None)
