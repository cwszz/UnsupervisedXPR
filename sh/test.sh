language_list=('ja')
test_list=('ja')
export CUDA_VISIBLE_DEVICES='7,6'
for lg in ${language_list[*]}; do
for test_lg in ${test_list[*]}; do
    python predict.py \
    --lg $lg \
    --test_lg $test_lg \
    --layer_id 12\
    --queue_length 0 \
    --dataset_path ./data/ \
    --eval_batch_size 2\
    --load_model_path ./cpt/QUEUE0_LG-ja_trainSample4_avail-32_seed-1_T-0.06_epoch-100_m-0.999_layer-12_/best.pt \
    --unsupervised 0 \
    > log/test/en_nomask_unsup_test-${lg}-${test_lg}-32.log 2>&1

done
done
# test_list=('de' 'es' 'fr' 'ru' 'ko' 'ja' 'zh' 'ar')