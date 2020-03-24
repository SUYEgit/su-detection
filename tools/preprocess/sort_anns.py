# -*- coding: utf-8 -*-
"""
# Sort annotations and imgs.
# Authors: suye
# Date: 2019/03/06 1:02 下午
"""
import os
import shutil
import xmltodict
import json


def sort_anns(base_path, out_path):
    """sort anns"""
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    img_path = os.path.join(out_path, 'imgs')
    xml_path = os.path.join(out_path, 'xmls')
    if not os.path.exists(img_path):
        os.mkdir(img_path)
    if not os.path.exists(xml_path):
        os.mkdir(xml_path)

    for root, dirs, files in os.walk(base_path, topdown=False):
        for name in files:
            file_path = os.path.join(root, name)
            print(file_path)
            filename = os.path.basename(file_path)

            if filename.endswith('.jpg'):
                shutil.copy(file_path, os.path.join(img_path, filename))
            elif filename.endswith('.xml'):
                shutil.copy(file_path, os.path.join(xml_path, filename))


def load_json(xml_path):
    """
    获取xml文件
    :param xml_path:
    :return:
    """
    with open(xml_path, 'r', encoding='utf-8') as f:
        xml_str = f.read()
    xmlparse = xmltodict.parse(xml_str)
    jsonstr = json.dumps(xmlparse, indent=1)
    jsonstr = json.loads(jsonstr)
    return jsonstr


if __name__ == '__main__':
    path = '/Users/suye02/jingyan2/data/3月1号初版标注数据/dibu'
    out_path = path + '_sorted'
    sort_anns(path, out_path)
