# -*- coding: utf-8 -*-
"""
# NMS
# Authors: suye
# Date: 2019/03/06 1:02 下午
"""
import numpy as np
from compute_iou import compute_iou


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
