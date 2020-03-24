#!/bin/bash

model_path='/root/suye/PaddleDetection/paddle_outputs/0305_cemian_mask_rcnn_hrnetv2p_w18_1x'

config_path=${model_path}/mask_rcnn_hrnetv2p_w18_1x.yml
ckpt_path=${model_path}/mask_rcnn_hrnetv2p_w18_1x/model_final
json_out=${model_path}/

python tools/eval.py \
    -c ${config_path} \
    -o weights=${ckpt_path} \
    --output_eval ${json_out}
