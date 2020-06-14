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

from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from pycocotools.coco import COCO, maskUtils

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


class Evaluator:
    def __init__(self, ann_json, res_json, image_path, out_path, thr, iou_thr, mask_json=None):
        self.image_path = image_path
        self.ann_coco = COCO(ann_json)
        self.res_coco = self.ann_coco.loadRes(res_json)
        if mask_json and os.path.exists(mask_json):
            self.mask_res_coco = self.ann_coco.loadRes(mask_json)

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

    def _load_bbox_results_by_id(self, image_id, res_json=None):
        """load results and thresh"""
        if res_json is None:
            res_ids = self.res_coco.getAnnIds(imgIds=image_id, catIds=self.cat_ids, iscrowd=None)
        else:
            res_coco = self.ann_coco.loadRes(res_json)
            res_ids = res_coco.getAnnIds(imgIds=image_id, catIds=self.cat_ids, iscrowd=None)
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

    def _load_mask_results_by_id(self, image_id, res_json=None):
        """load mask from result json"""
        if res_json is None:
            res_ids = self.mask_res_coco.getAnnIds(imgIds=image_id, catIds=self.cat_ids, iscrowd=None)
        else:
            res_coco = self.ann_coco.loadRes(res_json)
            res_ids = res_coco.getAnnIds(imgIds=image_id, catIds=self.cat_ids, iscrowd=None)
        results = self.res_coco.loadAnns(res_ids)

        res_masks = []
        for result in results:
            if float(result['score']) < self.thr:
                continue
            else:
                res_masks.append(result)

        return res_masks

    def compute_confmat(self, annotations, res_labels, res_bboxes, image_id):
        """累计混淆矩阵"""
        if len(annotations) == 0 and len(res_labels) > 0:
            for i in range(len(res_labels)):
                self.confmat[-1, res_labels[i] - 1] += 1
                self.confmat_ids['{}{}'.format(len(self.class_names) - 1, res_labels[i] - 1)].append(image_id)

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
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 255))

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
            labels, bboxes = self._load_bbox_results_by_id(target_id)
            self._visualize(annotations, labels, bboxes, img_np, os.path.join(vis_path, image['file_name']))

    def visualize_fp_fn(self, image_path):
        if not os.path.exists(os.path.join(self.out_path, 'badcase_vis')):
            os.mkdir(os.path.join(self.out_path, 'badcase_vis'))

        # 可视化漏检
        print('>>>>>>VISUALIZING FN BADCASES...')
        for i, class_name in enumerate(self.class_names[:-1]):
            # 该类别漏检id
            target_ids = self.confmat_ids['{}{}'.format(i, len(self.class_names) - 1)]
            if len(target_ids) == 0:
                continue
            vis_path = os.path.join(self.out_path,
                                    'badcase_vis',
                                    '{}_clsed_to_{}'.format(self.class_names[i],
                                                            self.class_names[len(self.class_names) - 1]))
            if not os.path.exists(vis_path):
                os.mkdir(vis_path)

            for target_id in target_ids:
                image = self.ann_coco.loadImgs(target_id)[0]
                img_np = cv2.imread(os.path.join(image_path, image['file_name']))

                annotations = self._load_ann_by_id(target_id)
                labels, bboxes = self._load_bbox_results_by_id(target_id)
                self._visualize(annotations, labels, bboxes, img_np, os.path.join(vis_path, image['file_name']))

        # 可视化虚警
        print('>>>>>>VISUALIZING FP BADCASES...')
        for i, class_name in enumerate(self.class_names[:-1]):
            # 该类别漏检id
            target_ids = self.confmat_ids['{}{}'.format(len(self.class_names) - 1, i)]
            if len(target_ids) == 0:
                continue
            vis_path = os.path.join(self.out_path,
                                    'badcase_vis',
                                    '{}_clsed_to_{}'.format(self.class_names[len(self.class_names) - 1],
                                                            self.class_names[i]))
            if not os.path.exists(vis_path):
                os.mkdir(vis_path)

            for target_id in target_ids:
                image = self.ann_coco.loadImgs(target_id)[0]
                img_np = cv2.imread(os.path.join(image_path, image['file_name']))

                annotations = self._load_ann_by_id(target_id)
                labels, bboxes = self._load_bbox_results_by_id(target_id)
                self._visualize(annotations, labels, bboxes, img_np, os.path.join(vis_path, image['file_name']))

    def binary_pr(self, fp_scores, tp_scores, out_path, grains=100):
        fp_scores.sort()
        tp_scores.sort()

        recalls = []
        precisions = []
        fps = []
        end_thr = 1
        for thr in [x / grains for x in list(range(0, grains, 1))]:
            if (len([x for x in tp_scores if x > thr]) + len([x for x in fp_scores if x > thr])) == 0:
                end_thr = thr
                break

            recall = len([x for x in tp_scores if x > thr]) / len(tp_scores)
            fp = len([x for x in fp_scores if x > thr]) / len(fp_scores)
            precision = len([x for x in tp_scores if x > thr]) / (len([x for x in tp_scores if x > thr]) + len(
                [x for x in fp_scores if x > thr]))
            recalls.append(recall)
            precisions.append(precision)
            fps.append(fp)

        plt.xlim((0, 1))
        plt.ylim((0, 1))

        plt.plot(recalls, fps, c='red')
        plt.grid(True)
        plt.legend(loc='upper left')
        plt.xlabel("recall")
        plt.ylabel("false positive rate")
        for i, thr in enumerate([x / grains for x in list(range(0, grains, 1))]):
            if thr >= float(end_thr):
                break
            plt.annotate(round(thr, 2), (recalls[i], fps[i]))

        plt.savefig(os.path.join(out_path, "fp-r_curve.png"))

    def img_level_pr(self, out_path):
        self.thr = 1e-10
        image_ids = self.ann_coco.getImgIds()

        fp_scores = []
        tp_scores = []
        for image_id in image_ids:
            annotations = self._load_ann_by_id(image_id)
            labels, bboxes = self._load_bbox_results_by_id(image_id)
            scores = [x[-1] for x in bboxes]
            if len(scores) == 0:
                scores.append(0)
            if len(annotations) > 0:
                tp_scores.append(max(scores))
            else:
                fp_scores.append(max(scores))
        self.binary_pr(fp_scores, tp_scores, out_path)

    def product_level_pr(self, out_path):
        self.thr = 1e-10
        image_ids = self.ann_coco.getImgIds()

        product_id_map = {}  # {'0001-0003':[1,3,34], ...}
        for image_id in image_ids:
            image = self.ann_coco.loadImgs(image_id)[0]
            image_name = image['file_name']
            if len(image_name) > 11:
                product_num = image_name[:9]
            else:
                product_num = image_name[:4]

            if product_num in product_id_map.keys():
                product_id_map[product_num].append(image_id)
            else:
                product_id_map[product_num] = [image_id]
        fp_scores = []
        tp_scores = []
        print('running product_level_pr for {} products'.format(len(product_id_map)))

        for product_num in product_id_map:
            img_ids = product_id_map[product_num]
            product_anns = []
            product_scores = []
            for img_id in img_ids:
                annotations = self._load_ann_by_id(img_id)
                labels, bboxes = self._load_bbox_results_by_id(img_id)
                scores = [x[-1] for x in bboxes]
                product_anns += annotations
                product_scores += scores

            if len(product_scores) == 0:
                # 整个product都没有检出
                product_scores.append(0)

            if len(product_anns) > 0:
                tp_scores.append(max(product_scores))
            else:
                fp_scores.append(max(product_scores))
        self.binary_pr(fp_scores, tp_scores, out_path)

    def weiyi_fuhe(self, out_path, image_path):
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        print('visualizing fuhe images')
        self.visualize_by_index(image_path=image_path)

        # 生成结果csv
        excel_dict = {'文件名': [], '缺陷': [], 'x': [], 'y': [], 'width': [], 'height': [], '分数': []}

        target_ids = self.ann_coco.getImgIds()
        for target_id in target_ids:
            image = self.ann_coco.loadImgs(target_id)[0]
            image_name = image['file_name']
            h
            labels, bboxes = self._load_bbox_results_by_id(target_id)
            for i, label in enumerate(labels):
                bbox = bboxes[i]
                excel_dict['文件名'].append(image_name)
                excel_dict['缺陷'].append(self.class_names[label - 1])
                excel_dict['x'].append(float(bbox[0]))
                excel_dict['y'].append(float(bbox[1]))
                excel_dict['width'].append(float(bbox[2]) - float(bbox[0]))
                excel_dict['height'].append(float(bbox[3]) - float(bbox[1]))
                excel_dict['分数'].append(float(bbox[4]))

        data_df = pd.DataFrame(excel_dict)

        writer = pd.ExcelWriter(os.path.join(out_path, 'results.xlsx'))
        data_df.to_excel(writer, float_format='%.5f')
        writer.save()

        print('fuhe file saved to: {}'.format(os.path.join(out_path, 'results.xlsx')))

    def _compute_fp_errors(self, bbox, mask, ann):
        """

        :param bbox:
        :param anns:
        :return:
        """
        file_name = self.ann_coco.loadImgs(ann['image_id'])[0]['file_name']
        img_path = os.path.join(self.image_path, file_name)
        assert os.path.exists(img_path)
        img_np = cv2.imread(img_path, 0)
        mask_np = self.mask_res_coco.annToMask(mask)
        # print(mask)
        # mask_np = maskUtils.decode(mask)
        assert mask_np.shape == img_np.shape

        xmin = int(bbox[0])
        ymin = int(bbox[1])
        xmax = int(bbox[2])
        ymax = int(bbox[3])
        assert xmin >= 0 and xmax <= img_np.shape[1] and ymin >= 0 and ymax <= img_np.shape[0]
        bbox_np = img_np[ymin: ymax, xmin: xmax]
        mask_np = mask_np[ymin: ymax, xmin: xmax]
        cv2.imwrite('./tmp/{}'.format(file_name), np.concatenate((bbox_np, mask_np * 255), axis=1))

        def _extract_feat(image, mask):
            """

            :param image:
            :param mask:
            :return:
            """
            import feature_param
            length, width = feature_param.extract_length_width_bydt(mask)
            pixel_area = feature_param.extract_pixel_area(mask)
            brightness, _, _ = feature_param.extract_brightness(mask, image)
            gradients = feature_param.extract_gradients(mask, image)
            contrast = feature_param.extract_contrast(mask, image)

            return np.array([length, width, pixel_area, brightness, gradients, contrast]).astype(np.int64)

        res_feats = _extract_feat(bbox_np, mask_np)
        ann_feats = np.array([ann['length'], ann['width'], ann['pixel_area'],
                              ann['brightness'], ann['gradients'], ann['contrast']])
        print('res fps', res_feats)
        print('ann fps', ann_feats)
        feats_error = np.subtract(ann_feats, res_feats)

        return list(np.absolute(feats_error))

    def _eval_fp(self, annotations, res_labels, res_bboxes, res_masks, fp_errors):
        """

        :param annotations:
        :param res_labels:
        :param res_bboxes:
        :param fp_errors:
        :return:
        """
        if len(annotations) == 0 or len(res_labels) == 0:
            return fp_errors

        for i, ann in enumerate(annotations):
            gt_bbox = ann['bbox']
            xmin = int(gt_bbox[0])
            ymin = int(gt_bbox[1])
            xmax = int(gt_bbox[0]) + int(gt_bbox[2])
            ymax = int(gt_bbox[1]) + int(gt_bbox[3])

            matched_ious = {}
            for j in range(len(res_labels)):
                iou = compute_iou((xmin, ymin, xmax, ymax), res_bboxes[j])
                if iou > self.iou_thr:
                    matched_ious[j] = iou
            print('{} matched bboxs'.format(len(matched_ious)))
            if len(matched_ious) > 0:
                iou_values = np.array(list(matched_ious.values()))
                max_iou_index = list(matched_ious.keys())[np.argmax(iou_values)]
                bbox_wmaxiou = res_bboxes[max_iou_index]
                bbox_errors = self._compute_fp_errors(bbox_wmaxiou, res_masks[max_iou_index], ann)
                if bbox_errors[1] > 1000:
                    continue
                assert len(bbox_errors) == len(fp_errors)
                for k, key in enumerate(fp_errors.keys()):
                    fp_errors[key].append(bbox_errors[k])
            else:
                # 未匹配框不做计算
                continue

        return fp_errors

    def eval_fp(self):
        print('EVLUATING FP...')
        image_ids = self.ann_coco.getImgIds()

        fp_errors = {'length': [],
                     'width': [],
                     'pixel_area': [],
                     'brightness': [],
                     'gradients': [],
                     'contrast': []
                     }
        for image_id in tqdm(image_ids):
            annotations = self._load_ann_by_id(image_id)
            labels, bboxes = self._load_bbox_results_by_id(image_id)
            masks = self._load_mask_results_by_id(image_id)

            assert (len(masks) == len(labels))
            print('get {} bboxs'.format(len(masks)))
            fp_errors = self._eval_fp(annotations, labels, bboxes, masks, fp_errors)
        for p in fp_errors:
            fp_errors[p] = sum(fp_errors[p]) / len(fp_errors[p])
            print(p, fp_errors[p])

    def vis_fp(self, out_path, bbox_json1, bbox_json2, mask_json1, mask_json2):
        if not os.path.exists(out_path):
            os.mkdir(out_path)

        image_ids = self.ann_coco.getImgIds()
        for image_id in tqdm(image_ids):
            file_name = self.ann_coco.loadImgs(image_id)[0]['file_name']
            print(file_name)
            img_path = os.path.join(self.image_path, file_name)
            assert os.path.exists(img_path)
            img_np = cv2.imread(img_path, 0)
            img_np_vis = copy.deepcopy(img_np)

            annotations = self._load_ann_by_id(image_id)
            labels1, bboxes1 = self._load_bbox_results_by_id(image_id, res_json=bbox_json1)

            labels2, bboxes2 = self._load_bbox_results_by_id(image_id, res_json=bbox_json2)
#             print('model 1 {} bboxes'.format(len(labels1)))
#             print('model 2 {} bboxes'.format(len(labels2)))

#             return
            masks1 = self._load_mask_results_by_id(image_id, res_json=mask_json1)
            masks2 = self._load_mask_results_by_id(image_id, res_json=mask_json2)

            assert len(labels2) == len(bboxes2) == len(masks2)

            labels_group = [labels1, labels2]
            bboxes_group = [bboxes1, bboxes2]
            masks_group = [masks1, masks2]

            for i, ann in enumerate(annotations):
                gt_bbox = ann['bbox']
                xmin = int(gt_bbox[0])
                ymin = int(gt_bbox[1])
                xmax = int(gt_bbox[0]) + int(gt_bbox[2])
                ymax = int(gt_bbox[1]) + int(gt_bbox[3])

                matched = 0
                print('<<<<<<<<< annotations')
                for j, res_labels in enumerate(labels_group):
                    matched_ious = {}
                    res_bboxes = bboxes_group[j]
                    print('group {} bbox {}'.format(j, len(res_bboxes)))
                    res_masks = masks_group[j]
                    for k in range(len(res_labels)):
                        iou = compute_iou((xmin, ymin, xmax, ymax), res_bboxes[k])
                        if iou > self.iou_thr:
                            matched_ious[k] = iou
                    print('{} matched bboxs'.format(len(matched_ious)))
                    if len(matched_ious) > 0:
                        matched += 1
                        iou_values = np.array(list(matched_ious.values()))
                        max_iou_index = list(matched_ious.keys())[np.argmax(iou_values)]
                        bbox_wmaxiou = res_bboxes[max_iou_index]
                        mask_wmaxiou = res_masks[max_iou_index]
                        b_xmin = int(bbox_wmaxiou[0])
                        b_ymin = int(bbox_wmaxiou[1])
                        b_xmax = int(bbox_wmaxiou[2])
                        b_ymax = int(bbox_wmaxiou[3])

                        # 提取fp并计算error
                        fp_errors = self._compute_fp_errors(bbox_wmaxiou, mask_wmaxiou, ann)
                        mask_np = self.mask_res_coco.annToMask(mask_wmaxiou)
                        assert mask_np.shape == img_np.shape

                        # 可视化mask bbox fp
                        colors = [(255, 255, 0), (255, 0, 255)]
                        color = colors[j]
                        cv2.rectangle(img_np_vis, (b_xmin, b_ymin), (b_xmax, b_ymax), color, 2)
                        # put text
                        label_text = 'DET {} | {:.02f} | {}'.format(self.class_names[res_labels[max_iou_index]],
                                                                    bbox_wmaxiou[-1],
                                                                    fp_errors)
                        if j == 0:
                            cv2.putText(img_np_vis, label_text, (b_xmin, b_ymin - 2), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                                        color)
                        elif j == 1:
                            cv2.putText(img_np_vis, label_text, (b_xmin, b_ymin - 2), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                                        color)

                        # mask_np = mask_np.astype(np.float32)
                        # color_mask = np.array(color).astype(np.int8)
                        # img_np_vis = cv2.cvtColor(img_np_vis, cv2.COLOR_GRAY2RGB)
                        # mask_np = cv2.cvtColor(mask_np, cv2.COLOR_GRAY2RGB)
                        # print(img_np_vis.shape, mask_np.shape)
                        # mask_np = mask_np.astype(np.int8)
                        # img_np_vis[mask_np] = img_np_vis[mask_np] * 0.5 + color_mask * 0.5
            if matched == 2:
                img_out = np.concatenate((img_np, img_np_vis), axis=1)
                cv2.imwrite(os.path.join(out_path, file_name), img_out)

    def confusion(self):
        image_ids = self.ann_coco.getImgIds()
        for image_id in image_ids:
            annotations = self._load_ann_by_id(image_id)
            labels, bboxes = self._load_bbox_results_by_id(image_id)
            self.compute_confmat(annotations, labels, bboxes, image_id)

        plot_confmat(self.confmat, self.class_names, out_path=self.out_path)


if __name__ == '__main__':
    image_path_df = '/root/suye/huawei_data/0410_cemian/val'
    ann_json_df = '/root/suye/huawei_data/0410_cemian/annotations/instances_val_fp.json'
    res_json_df = '/root/suye/mmdetection/train_outputs/0421_HUAWEI_nrc_cemian_htc_without_semantic_r50_fpn_1x/eval_analysis/results.pkl.bbox.json'
    mask_res_json_df = '/root/suye/mmdetection/train_outputs/0421_HUAWEI_nrc_cemian_htc_without_semantic_r50_fpn_1x/eval_analysis/results.pkl.segm.json'
    out_path_df = './'
    thr_df = 0.001
    iou_thr_df = 0.1

    parser = argparse.ArgumentParser(description='This script support confusion matrix utils.')
    parser.add_argument('--image_path', type=str, default=image_path_df)
    parser.add_argument('--ann_json', type=str, default=ann_json_df)
    parser.add_argument('--res_json', type=str, default=res_json_df)
    parser.add_argument('--mask_res_json', type=str, default=mask_res_json_df)
    parser.add_argument('--out_path', type=str, default=out_path_df)
    parser.add_argument('--thresh', type=str, default=thr_df)
    parser.add_argument('--iou_thr', type=str, default=iou_thr_df)

    args = parser.parse_args()
    print(args.ann_json)
    assert args.thresh is not None
    evaluator = Evaluator(image_path=args.image_path,
                          ann_json=args.ann_json,
                          res_json=args.res_json,
                          mask_json=args.mask_res_json,
                          out_path=args.out_path,
                          thr=float(args.thresh),
                          iou_thr=float(args.iou_thr))
    evaluator.vis_fp(out_path='/root/suye/mmdetection/fp_compare',
                     bbox_json1='/root/suye/mmdetection/train_outputs/0413_HUAWEI_nrc_cemian_mask_rcnn_r50_fpn_1x/eval_analysis/results.pkl.bbox.json',
                     bbox_json2='/root/suye/mmdetection/train_outputs/0421_HUAWEI_nrc_cemian_htc_without_semantic_r50_fpn_1x/eval_analysis/results.pkl.bbox.json',
                     mask_json1='/root/suye/mmdetection/train_outputs/0413_HUAWEI_nrc_cemian_mask_rcnn_r50_fpn_1x/eval_analysis/results.pkl.segm.json',
                     mask_json2='/root/suye/mmdetection/train_outputs/0421_HUAWEI_nrc_cemian_htc_without_semantic_r50_fpn_1x/eval_analysis/results.pkl.segm.json')
#     evaluator.confusion()
# evaluator.weiyi_fuhe(args.out_path, args.image_path)
# confusion.product_level_pr(out_path=args.out_path)
# if args.image_path is not None:
#     # confusion.visualize_by_index(args.image_path, gt_id=None, res_id=None)
#     confusion.visualize_fp_fn(args.image_path)

