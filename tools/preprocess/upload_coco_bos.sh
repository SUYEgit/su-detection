#!/bin/bash
coco_path='/Users/suye02/HUAWEI/data/train_data/0328_cemian_sorted'
data_name='0328_cemian_allclass_coco'
bucket_name='project-huawei'

zip -r ${data_name}.zip \
    ${coco_path}/train \
    ${coco_path}/val \
    ${coco_path}/annotations \
    ${coco_path}/xmls

~/bos/bcecmd bos cp ${data_name}.zip bos:/${bucket_name}/${data_name}.zip

echo DATA UPLOADED AT:
~/bos/bcecmd bos gen_signed_url bos:/${bucket_name}/{data_name}.zip
rm ${data_name}.zip