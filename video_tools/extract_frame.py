# -*- coding:utf8 -*-
import cv2
import os
import shutil


def extract_frame(video_path, out_name, strides):
    if not os.path.exists(out_name):
        os.makedirs(out_name)
        print('path of %s is build' % out_name)
    else:
        shutil.rmtree(out_name)
        os.makedirs(out_name)
        print('path of %s already exist and rebuild' % out_name)

    # 开始读视频
    video_capture = cv2.VideoCapture(video_path)
    i = 0
    j = 0

    while True:
        success, frame = video_capture.read()
        if not success:
            print('video is all read')
            break
        i += 1
        if i % strides == 0:
            # 保存图片
            j += 1
            out_name = os.path.basename(video_path.split('.')[0] + '_' + str(j) + '_' + str(i) + '.jpg')
            cv2.imwrite(os.path.join(out_name, out_name), frame)
            print(os.path.join(out_name, out_name))


if __name__ == '__main__':
    extract_frame(video_path='/Users/suye02/泉州水务-训练视频/水箱锈蚀.mp4',
                  out_name='/Users/suye02/泉州水务-训练视频/水箱锈蚀/',
                  strides=5)
