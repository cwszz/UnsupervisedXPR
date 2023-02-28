for lg in 'zh' 'ar' 'ko' 'ru' 'fr' 'de' 'es' 'ja' ;
do
export CUDA_VISIBLE_DEVICES='2,3'
python -m torch.distributed.launch --nproc_per_node=2 --master_port 29505 \
    trainMoCo.py \
    --lg $lg \
    --sn '32' \
    --simclr 0 \
    --queue_length 0 \
    --T_para 0.06 \
    --seed 1 \
    --output_log_dir 'result' \
    --dev_only_q_encoder 1 \
    > log/${lg}-32-1-layer_${layer}.log 2>&1
done