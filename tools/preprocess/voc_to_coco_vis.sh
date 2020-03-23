#!/bin/bash

# Usage: sh voc_to_coco_vis.sh

base_dir='/Users/suye02/copy/youji_data/验收训练/mmnet/ceguai'
for split in train val
do
    python voc_to_coco.py \
        --ann_dir ${base_dir}/Annotations \
        --ann_ids ${base_dir}/ImageSets/Main/${split}.txt \
        --labels ${base_dir}/ImageSets/labels.txt \
        --output ${base_dir}/${split}.json \
        --ext xml
    python visualize_coco.py \
        --images_path ${base_dir}/JPEGImages \
        --json_path ${base_dir}/${split}.json \
        --out_path ${base_dir}/${split}_vis
done


