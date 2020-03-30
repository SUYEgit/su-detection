#!/bin/bash
base_path='/Users/suye02/su-detection/data/HUAWEI/data/3.28+29/拐角组装后'
ann_name='annotations'

# 1. Sort xmls and imgs
sorted_path=${base_path}_sorted
rm -r ${sorted_path}
python sort_anns.py --base_path ${base_path}

# 2. Split train val imgs
python train_val_split.py --images_path ${sorted_path}/train

# 3. Make COCO dataset
python weiyi_xml_to_coco.py \
    --base_path ${sorted_path} \
    --out_path ${sorted_path}/${ann_name}

# 4. Analyze COCO dataset
python coco_analysis.py \
    --json_path ${sorted_path}/annotations/instances_train.json \
    --images_path ${sorted_path}/train \
    --out_path ${sorted_path}

# 5. Upload DATA
# ./upload_coco_bos.sh