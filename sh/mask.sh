for lg in 'fr' 'de' 'es' ;
do
export CUDA_VISIBLE_DEVICES='4,5'
python -m torch.distributed.launch --nproc_per_node=2 --master_port 29503 \
    unsup_train.py \
    --lg $lg \
    --sn '32' \
    --simclr 0 \
    --queue_length 512 \
    --T_para 0.06 \
    --seed 1 \
    --output_log_dir 'mask_unsup_result' \
    --mask_percent 2 \
    --ismasked 1 \
    --dev_only_q_encoder 1 \
    > log/unsup/queue512_30%mask_unsup_${lg}-32-1-layer_${layer}.log 2>&1
done