model_name='0423_BX_merge_his_withbg_faster_rcnn_r50_caffe_c4_1x'
config_name='faster_rcnn_r50_caffe_c4_1x.py'
ann_json='/root/suye/jingyan2_data/BX/0422_bx_merge/annotations/instances_val.json'
image_path='/root/suye/jingyan2_data/BX/0422_bx_merge/val_his'

analysis_name='eval_analysis'
conf_thresh=0.5

#1. Start training
# ./tools/dist_train.sh configs/suye/${config_name} 1 --validate
# cp configs/suye/${config_name} train_outputs/${model_name}/

#2. Run Evaluation
model_path=/root/suye/mmdetection-1.0rc0/train_outputs/${model_name}
echo evaluating model ${model_path} ......
config_path=${model_path}/${config_name}
pth_path=${model_path}/latest.pth
out_path=${model_path}/${analysis_name}
mkdir ${out_path}
json_out=${out_path}/results.pkl

echo INFERENCING TEST IMAGES...
python tools/eval.py \
    -c ${config_path} \
    -o weights=${ckpt_path} \
    --output_eval ${json_out}
#     > ${out_path}/EVAL.log

#3. Analyze model performance
echo EVALUATING MODDEL PERFORMANCE...
bbox_res_json=${json_out}.bbox.json
python eval.py \
    --ann_json ${ann_json} \
    --res_json ${bbox_res_json} \
    --out_path ${out_path} \
    --thresh ${conf_thresh} \
    --image_path ${image_path}

for type in bbox
do
    res_json=${json_out}.${type}.json
    mkdir ${out_path}/${type}
    python pr_eval.py ${res_json} ${out_path} --ann ${ann_json} --types ${type} > ${out_path}/${type}/${type}.log
done

echo DONE
