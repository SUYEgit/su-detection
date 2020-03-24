#!/bin/bash

model_path='/root/suye/mmdetection-1.0rc0/train_outputs/0220_sd5_512regioncut_cascade_mask_rcnn_r18_fpn_1x'

config_path=${model_path}/cascade_mask_rcnn_r18_fpn_1x.py
pth_path=${model_path}/epoch_60.pth
json_out=${model_path}/results

python tools/test.py \
    ${config_path} \
    ${pth_path} \
    --json_out ${json_out} \
    --eval bbox segm

