# 基于coco格式数据的标准化evaluation工具

##1. mmdetection

`python tools/test.py {CONFIG_PATH}/mask_rcnn.py {CKPT_PATH}/epoch_60.pth --json_out {JSON_OUT_PATH} --eval bbox segm`

##2. Paddle

`python tools/eval.py -c {CONFIG_PATH}/mask_rcnn_hrnetv2p_w18_1x.yml --output_eval {JSON_OUT_PATH}`

##3. 进行详细PR分析

bbox:

`python metric_analysis.py {JSON_OUT_PATH}/0323_test.bbox.json {METRIC_OUT_PATH} --ann {GT_PATH}/instances_test.json --types bbox`

segm:

`python metric_analysis.py {JSON_OUT_PATH}/0323_test.segm.json {METRIC_OUT_PATH} --ann {GT_PATH}/instances_test.json --types segm`