#!/bin/bash
coco_path='/Users/suye02/su-detection/data/HUAWEI/data/训练数据/0410_cemian'
data_name='0410_cemian'
bucket_name='project-huawei'

cd ${coco_path}
zip -r ${data_name}.zip \
    ${coco_path}/train \
    ${coco_path}/val \
    ${coco_path}/annotations \
    ${coco_path}/xmls

~/bos/bcecmd bos cp ${data_name}.zip bos:/${bucket_name}/${data_name}.zip

echo DATA UPLOADED AT:
~/bos/bcecmd bos gen_signed_url bos:/${bucket_name}/{data_name}.zip
rm ${data_name}.zip