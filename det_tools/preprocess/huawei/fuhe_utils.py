# -*- coding: utf-8 -*-
"""
# This module provide utils for fuhe data.
# Authors: suye
# Date: 2020/03/20 3:14 pm
"""
import os

import cv2


def visualize(csv_path, image_path, out_path):
    class_names = ['huashang', 'pengya', 'yise', 'cashang', 'madian', 'zhendaowen']

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    with open(csv_path, 'r') as f:
        lines = f.readlines()
    cur_name = 0
    for line in lines:
        name, label, xmin, ymin, width, height, score, res = line.split('\t')
        name = name.replace(' ', '')
        print(name, label, xmin, ymin, width, height, score, res)
        if name != cur_name:
            print(os.path.join(image_path, name + '.jpg'))
            image_np = cv2.imread(os.path.join(image_path, name + '.jpg'))
            cur_name = name
        if '检对' in res:
            continue
        elif '漏检' in res:
            cv2.rectangle(image_np, (int(xmin), int(ymin)), (int(xmin) + int(width), int(ymin) + int(height)),
                          [255, 255, 255], 2)

            cv2.putText(image_np, 'FN', (int(xmin), int(ymin)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
            cv2.imwrite(os.path.join(out_path, 'FN' + name + '.jpg'), image_np)

        elif '过检' in res:
            cv2.rectangle(image_np, (int(xmin), int(ymin)), (int(xmin) + int(width), int(ymin) + int(height)),
                          [255, 0, 255], 2)

            cv2.putText(image_np, 'FP', (int(xmin), int(ymin)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 2)
            cv2.imwrite(os.path.join(out_path, 'FP' + name + '.jpg'), image_np)

    return


if __name__ == '__main__':
    csv_path = '/Users/suye02/HUAWEI/data/第一次数据复核/fuhe1.txt'
    image_path = '/Users/suye02/HUAWEI/data/训练数据/侧面_sorted/val'
    out_path = '/Users/suye02/HUAWEI/data/第一次数据复核/visualize'
    visualize(csv_path, image_path, out_path)
