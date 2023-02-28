language_list=('all')
test_list=('fr')  
export CUDA_VISIBLE_DEVICES='6,7'
for lg in ${language_list[*]}; do
for test_lg in ${test_list[*]}; do
    python predict.py \
    --lg $lg \
    --test_lg $test_lg \
    --layer_id 12\
    --dataset_path ./data/ \
    --queue_length 0 \
    --load_model_path ./adv_result/4-${test_lg}-32-true-0-0.06-1-100-0.999-0-dev_qq-layer_12/best.pt \
    --unsupervised 0 \
    > log/test/test-${lg}-${test_lg}-32-adv.log 2>&1

done
done
# test_list=('de' 'es' 'fr' 'ru' 'ko' 'ja' 'zh' 'ar')
