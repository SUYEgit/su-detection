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
import argparse


def sort_anns(base_path, out_path):
    """sort anns"""
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    img_path = os.path.join(out_path, 'train')
    xml_path = os.path.join(out_path, 'xmls')
    if not os.path.exists(img_path):
        os.mkdir(img_path)
    if not os.path.exists(xml_path):
        os.mkdir(xml_path)

    for root, dirs, files in os.walk(base_path, topdown=False):
        for name in files:
            file_path = os.path.join(root, name)
            # print(file_path)
            filename = os.path.basename(file_path)
            if 'defect' in filename or 'setting' in filename:
                continue
            if os.path.exists(os.path.join(img_path, filename)):
                print('{} OVER WRITING {}'.format(file_path, os.path.join(img_path, filename)))
                tmp_path = os.path.join(out_path, 'tmp')
                if not os.path.exists(tmp_path):
                    os.mkdir(tmp_path)
                shutil.copy(file_path, os.path.join(tmp_path, filename))
                continue
            # assert not os.path.exists(os.path.join(img_path, filename))

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


def main():
    print('hello')


if __name__ == '__main__':
    base_path_df = '/Users/suye02/su-detection/data/HUAWEI/data/train/侧面'

    parser = argparse.ArgumentParser(description='Sort anns')
    parser.add_argument('--base_path', type=str, default=base_path_df)
    args = parser.parse_args()

    out_path = args.base_path + '_sorted'
    print('____________SORTINIG IMGS AND XMLS_________________')
    sort_anns(args.base_path, out_path)
    print('____________IMGS AND XMLS SORTED IN {}_________________'.format(out_path))

