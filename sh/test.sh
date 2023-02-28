language_list=('fr')
test_list=('fr') 
export CUDA_VISIBLE_DEVICES='3,2'
for lg in ${language_list[*]}; do
for test_lg in ${test_list[*]}; do
    python predict.py \
    --lg $lg \
    --test_lg $test_lg \
    --layer_id 12\
    --dataset_path ./data/ \
    --queue_length 0 \
    --load_model_path ./mask_unsup_result/30%_4-${test_lg}-32-true-0-0.06-1-100-0.999-0-dev_qq-layer_12/best.pt \
    --unsupervised 0 \
    > log/test/30%mask_unsup_test-${lg}-${test_lg}-32.log 2>&1

done
done
# test_list=('de' 'es' 'fr' 'ru' 'ko' 'ja' 'zh' 'ar')