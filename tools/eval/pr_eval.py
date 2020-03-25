#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
对COCO数据的模型结果进行AP计算，PR曲线绘制等分析
Date:    2020/3/3 下午7:04
"""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import os
from argparse import ArgumentParser
from multiprocessing import Pool

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from terminaltables import AsciiTable
import itertools


def makeplot(rs, ps, out_dir, class_name, iou_type):
    cs = np.vstack([
        np.ones((2, 3)),
        np.array([.31, .51, .74]),
        np.array([.75, .31, .30]),
        np.array([.36, .90, .38]),
        np.array([.50, .39, .64]),
        np.array([1, .6, 0])
    ])
    areaNames = ['allarea', 'small', 'medium', 'large']
    # areaNames = ['allarea']
    types = ['C75', 'C50', 'Loc', 'Sim', 'Oth', 'BG', 'FN']
    for i in range(len(areaNames)):
        area_ps = ps[..., i, 0]
        figure_tile = iou_type + '-' + class_name + '-' + areaNames[i]
        aps = [ps_.mean() for ps_ in area_ps]
        ps_curve = [
            ps_.mean(axis=1) if ps_.ndim > 1 else ps_ for ps_ in area_ps
        ]
        ps_curve.insert(0, np.zeros(ps_curve[0].shape))
        fig = plt.figure()
        plt.grid(True)
        plt.plot([0, 1], [0, 1], 'r--')
        ax = plt.subplot(111)
        for k in range(len(types)):
            ax.plot(rs, ps_curve[k + 1], color=[0, 0, 0], linewidth=0.5)
            ax.fill_between(
                rs,
                ps_curve[k],
                ps_curve[k + 1],
                color=cs[k],
                label=str('[{:.3f}'.format(aps[k]) + ']' + types[k]))
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.xlim(0, 1.)
        plt.ylim(0, 1.)
        plt.title(figure_tile)
        plt.legend()
        # plt.show()
        fig.savefig(out_dir + '/{}.png'.format(figure_tile))
        plt.close(fig)


def analyze_individual_category(k, cocoDt, cocoGt, catId, iou_type):
    nm = cocoGt.loadCats(catId)[0]
    print('--------------analyzing {}-{}---------------'.format(
        k + 1, nm['name']))
    ps_ = {}
    dt = copy.deepcopy(cocoDt)
    nm = cocoGt.loadCats(catId)[0]
    imgIds = cocoGt.getImgIds()
    dt_anns = dt.dataset['annotations']
    select_dt_anns = []
    for ann in dt_anns:
        if ann['category_id'] == catId:
            select_dt_anns.append(ann)
    dt.dataset['annotations'] = select_dt_anns
    dt.createIndex()
    # compute precision but ignore superclass confusion
    gt = copy.deepcopy(cocoGt)
    child_catIds = gt.getCatIds(supNms=[nm['supercategory']])
    for idx, ann in enumerate(gt.dataset['annotations']):
        if (ann['category_id'] in child_catIds
                and ann['category_id'] != catId):
            gt.dataset['annotations'][idx]['ignore'] = 1
            gt.dataset['annotations'][idx]['iscrowd'] = 1
            gt.dataset['annotations'][idx]['category_id'] = catId
    cocoEval = COCOeval(gt, copy.deepcopy(dt), iou_type)
    cocoEval.params.imgIds = imgIds
    cocoEval.params.maxDets = [100]
    cocoEval.params.iouThrs = [.1]
    cocoEval.params.useCats = 1
    cocoEval.evaluate()
    cocoEval.accumulate()
    ps_supercategory = cocoEval.eval['precision'][0, :, k, :, :]
    ps_['ps_supercategory'] = ps_supercategory
    # compute precision but ignore any class confusion
    gt = copy.deepcopy(cocoGt)
    for idx, ann in enumerate(gt.dataset['annotations']):
        if ann['category_id'] != catId:
            gt.dataset['annotations'][idx]['ignore'] = 1
            gt.dataset['annotations'][idx]['iscrowd'] = 1
            gt.dataset['annotations'][idx]['category_id'] = catId
    cocoEval = COCOeval(gt, copy.deepcopy(dt), iou_type)
    cocoEval.params.imgIds = imgIds
    cocoEval.params.maxDets = [100]
    cocoEval.params.iouThrs = [.1]
    cocoEval.params.useCats = 1
    cocoEval.evaluate()
    cocoEval.accumulate()
    ps_allcategory = cocoEval.eval['precision'][0, :, k, :, :]
    ps_['ps_allcategory'] = ps_allcategory
    return k, ps_


def category_ap(precisions, coco_gt):
    """compute per category ap"""
    catIds = coco_gt.getCatIds()
    # precision has dims (iou, recall, cls, area range, max dets)
    assert len(catIds) == precisions.shape[2]

    for iou_thr in [0.75, 0.5, 0.1]:
        results_per_category = []
        for idx, catId in enumerate(catIds):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            nm = coco_gt.loadCats(catId)[0]
            if iou_thr == 0.75:
                precision = precisions[0, :, idx, 0, -1]
            elif iou_thr == 0.5:
                precision = precisions[1, :, idx, 0, -1]
            else:
                precision = precisions[2, :, idx, 0, -1]
            #             precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float('nan')
            results_per_category.append(('{}'.format(nm['name']), '{:0.3f}'.format(float(ap * 100))))

        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        headers = ['category', 'AP({})'.format(iou_thr)] * (N_COLS // 2)
        results_2d = itertools.zip_longest(
            *[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table_data = [headers]
        table_data += [result for result in results_2d]
        table = AsciiTable(table_data)
        print(table.table)


def analyze_results(res_file, ann_file, res_types, out_dir):
    """analyze_results"""
    # coco_eval.eval['precision']是一个5维的数组
    #  precision  - [TxRxKxAxM] precision for every evaluation setting
    #  catIds     - [all] K cat ids to use for evaluation
    #  iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
    #  recThrs    - [0:.01:1] R=101 recall thresholds for evaluation
    #  areaRng    - [...] A=4 object area ranges for evaluation
    #  maxDets    - [1 10 100] M=3 thresholds on max detections per image

    # 第一维T：IoU的10个阈值，从0.5到0.95间隔0.05
    # 第二维R：101个recall 阈值，从0到101
    # 第三维K：类别，如果是想展示第一类的结果就设为0
    # 第四维A：area 目标的大小范围 （all，small, medium, large）（全部，小，中，大）
    # 第五维M：maxDets 单张图像中最多检测框的数量 三种 1,10,100

    # coco_eval.eval['precision'][0, :, 0, 0, 2] 所表示的就是当IoU=0.5时
    # 从0到100的101个recall对应的101个precision的值
    print(">>>>>>>>>>>>>>>>>start analyze results")
    for res_type in res_types:
        assert res_type in ['bbox', 'segm']
    directory = os.path.dirname(out_dir + '/')
    if not os.path.exists(directory):
        print('-------------create {}-----------------'.format(out_dir))
        os.makedirs(directory)
    cocoGt = COCO(ann_file)
    cocoDt = cocoGt.loadRes(res_file)
    imgIds = cocoGt.getImgIds()
    for res_type in res_types:
        res_out_dir = out_dir + '/' + res_type + '/'
        res_directory = os.path.dirname(res_out_dir)
        if not os.path.exists(res_directory):
            print(
                '-------------create {}-----------------'.format(res_out_dir))
            os.makedirs(res_directory)
        iou_type = res_type
        coco_eval = COCOeval(copy.deepcopy(cocoGt), copy.deepcopy(cocoDt), iou_type)
        coco_eval.params.imgIds = imgIds
        coco_eval.params.iouThrs = [.75, .5, .1]
        coco_eval.params.maxDets = [100]
        coco_eval.evaluate()
        coco_eval.accumulate()
        ps = coco_eval.eval['precision']
        category_ap(ps, copy.deepcopy(cocoGt))

        ps = np.vstack([ps, np.zeros((4, *ps.shape[1:]))])
        catIds = cocoGt.getCatIds()
        recThrs = coco_eval.params.recThrs
        with Pool(processes=48) as pool:
            args = [(k, cocoDt, cocoGt, catId, iou_type)
                    for k, catId in enumerate(catIds)]
            analyze_results = pool.starmap(analyze_individual_category, args)
        for k, catId in enumerate(catIds):
            nm = cocoGt.loadCats(catId)[0]
            print('--------------saving {}-{}---------------'.format(
                k + 1, nm['name']))
            analyze_result = analyze_results[k]
            assert k == analyze_result[0]
            ps_supercategory = analyze_result[1]['ps_supercategory']
            ps_allcategory = analyze_result[1]['ps_allcategory']
            # 计算消除超类混淆的precision
            ps[3, :, k, :, :] = ps_supercategory
            # 计算消除类间混淆的precision
            ps[4, :, k, :, :] = ps_allcategory
            # 计算消除背景过杀和漏检的precision
            ps[ps == -1] = 0
            ps[5, :, k, :, :] = (ps[4, :, k, :, :] > 0)
            ps[6, :, k, :, :] = 1.0
            makeplot(recThrs, ps[:, :, k], res_out_dir, nm['name'], iou_type)
        makeplot(recThrs, ps, res_out_dir, 'allclass', iou_type)


def main():
    parser = ArgumentParser(description='COCO Error Analysis Tool')
    parser.add_argument('result', help='result file (json format) path')
    parser.add_argument('out_dir', help='dir to save analyze result images')
    parser.add_argument(
        '--ann',
        default='/root/suye/PaddleDetection/jingyan_data/dingbu/annotations/instances_val.json',
        help='annotation file path')
    parser.add_argument(
        '--types', type=str, nargs='+', default=['bbox'], help='result types')
    args = parser.parse_args()
    analyze_results(args.result, args.ann, args.types, out_dir=args.out_dir)


if __name__ == '__main__':
    main()
