for lg in 'all_unsupervised' ;
do
export CUDA_VISIBLE_DEVICES='6,7'

python -m torch.distributed.launch --nproc_per_node=2 --master_port 29502 \
    unsup_train.py \
    --lg $lg \
    --sn '32' \
    --simclr 0 \
    --T_para 0.06 \
    --seed 1 \
    --onlylg 1\
    --ismasked 0 \
    --mask_percent 0\
    --queue_length 0\
    --output_log_dir 'cpt' \
    --dev_only_q_encoder 1 \
    > log/new_unsup/lg_no_mask_${lg}-1.log 2>&1

python -m torch.distributed.launch --nproc_per_node=2 --master_port 29502 \
    unsup_train.py \
    --lg $lg \
    --sn '32' \
    --simclr 0 \
    --T_para 0.06 \
    --seed 1 \
    --ismasked 0 \
    --mask_percent 8 \
    --queue_length 0\
    --output_log_dir 'cpt' \
    --dev_only_q_encoder 1 \
    > log/new_unsup/no_mask_${lg}-1.log 2>&1


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