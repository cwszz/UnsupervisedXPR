for lg in 'ja' 'ru' 'zh' 'ko' 'ar' ;
do
export CUDA_VISIBLE_DEVICES='0,1'

python -m torch.distributed.launch --nproc_per_node=2 --master_port 29502 \
    unsup_train.py \
    --lg $lg \
    --sn '32' \
    --simclr 0 \
    --T_para 0.06 \
    --seed 1 \
    --ismasked 1 \
    --mask_percent 8 \
    --train_epoch 50\
    --dev_sample_num 16\
    --queue_length 0\
    --output_log_dir 'cpt' \
    --dev_only_q_encoder 1 \
    > log/new_unsup/mask8_${lg}-1.log 2>&1

python -m torch.distributed.launch --nproc_per_node=2 --master_port 29502 \
    unsup_train.py \
    --lg $lg \
    --sn '32' \
    --simclr 0 \
    --T_para 0.06 \
    --seed 1 \
    --ismasked 0 \
    --train_epoch 50\
    --dev_sample_num 16\
    --mask_percent 0 \
    --queue_length 0\
    --output_log_dir 'cpt' \
    --dev_only_q_encoder 1 \
    > log/new_unsup/nomask_${lg}-1.log 2>&1



done

# python -m torch.distributed.launch --nproc_per_node=2 --master_port 29502 \
#     unsup_train.py \
#     --lg $lg \
#     --sn '32' \
#     --simclr 0 \
#     --T_para 0.06 \
#     --seed 1 \
#     --ismasked 1 \
#     --mask_percent 7 \
#     --queue_length 0\
#     --output_log_dir 'cpt' \
#     --dev_only_q_encoder 1 \
#     > log/unsup/mask7_${lg}-1.log 2>&1