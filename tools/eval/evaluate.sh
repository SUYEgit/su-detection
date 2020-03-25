#!/bin/bash

ann_json='/root/suye/PaddleDetection/jingyan_data/cemian/annotations/instances_val.json'
res_json='/root/suye/PaddleDetection/paddle_outputs/0305_cemian_mask_rcnn_hrnetv2p_w18_1x/bbox.json'
out_path='/root/suye/PaddleDetection/paddle_outputs/0305_cemian_mask_rcnn_hrnetv2p_w18_1x/eval_analysis'
thresh=0.3

python plot_confmat.py \
    --ann_json ${ann_json} \
    --res_json ${res_json} \
    --out_path ${out_path} \
    --thr ${thresh}

python pr_eval.py ${res_json} ${out_path} --ann ${ann_json} --types bbox segm



