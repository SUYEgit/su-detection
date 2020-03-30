model_name='0328_HUAWEI_cemianbs_mask_rcnn_hrnetv2p_w32_1x'
config_name='mask_rcnn_hrnetv2p_w32_1x.py'
ann_json='/root/suye/huawei_data/0328_cemian/annotations/instances_val.json'

analysis_name='0328_eval_analysis'
conf_thresh=0.1


#1. Start training
./tools/dist_train.sh configs/suye/${config_name} 1 --validate
cp configs/suye/${config_name} train_outputs/${model_name}/


#2. Run Evaluation
model_path=/root/suye/mmdetection-1.0rc0/train_outputs/${model_name}
echo evaluating model ${model_path} ......
config_path=${model_path}/${config_name}
pth_path=${model_path}/latest.pth
out_path=${model_path}/${analysis_name}
mkdir ${out_path}
json_out=${out_path}/results.pkl

echo INFERENCING TEST IMAGES...
python tools/test.py \
    ${config_path} \
    ${pth_path} \
    --out ${json_out} \
    --eval bbox segm \
    > ${out_path}/EVAL.log


#3. Analyze model performance
echo EVALUATING MODDEL PERFORMANCE...
for type in bbox segm
do
    res_json=${json_out}.${type}.json
    image_path=''

    python plot_confmat.py \
        --ann_json ${ann_json} \
        --res_json ${res_json} \
        --out_path ${out_path} \
        --thr ${conf_thresh}
        #--image_path ${image_path} \
    mkdir ${out_path}/${type}
    python pr_eval.py ${res_json} ${out_path} --ann ${ann_json} --types ${type} > ${out_path}/${type}/${type}.log
done

echo DONE


#4. Visualize Results


