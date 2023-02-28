for lg in 'fr' 'de' 'es' 'ar' 'ko' 'ru' 'zh' 'ja' ;
do
export CUDA_VISIBLE_DEVICES='6,7'
python -m torch.distributed.launch --nproc_per_node=2 --master_port 29504 \
    adv_train.py \
    --lg $lg \
    --sn '32' \
    --simclr 0 \
    --queue_length 0 \
    --T_para 0.06 \
    --seed 1 \
    --output_log_dir 'adv_result' \
    --dev_only_q_encoder 1 \
    > log/adv_${lg}-32-1-layer_${layer}.log 2>&1

done