export CUDA_VISIBLE_DEVICES='0,7'

for lg in 'fr' 'de' 'es' 'all_unsupervised';
do

python -m torch.distributed.launch --nproc_per_node=2 --master_port 29504 \
    unsup_train.py \
    --lg $lg \
    --sn '32' \
    --simclr 0 \
    --T_para 0.06 \
    --seed 1 \
    --ismasked 1 \
    --mask_percent 9 \
    --queue_length 0\
    --output_log_dir 'cpt' \
    --dev_only_q_encoder 1 \
    > log/unsup/mask9_${lg}-1.log 2>&1

# python -m torch.distributed.launch --nproc_per_node=2 --master_port 29504 \
#     unsup_train.py \
#     --lg $lg \
#     --sn '32' \
#     --simclr 0 \
#     --T_para 0.06 \
#     --seed 1 \
#     --queue_length 0\
#     --output_log_dir 'cpt' \
#     --dev_only_q_encoder 1 \
#     > log/unsup/unsup_${lg}-1.log 2>&1

# python -m torch.distributed.launch --nproc_per_node=2 --master_port 29504 \
#     unsup_train.py \
#     --lg $lg \
#     --sn '32' \
#     --simclr 0 \
#     --T_para 0.06 \
#     --seed 1 \
#     --queue_length 512\
#     --output_log_dir 'cpt' \
#     --dev_only_q_encoder 1 \
#     > log/unsup/queue512_unsup_${lg}-1.log 2>&1

python -m torch.distributed.launch --nproc_per_node=2 --master_port 29504 \
    unsup_train.py \
    --lg $lg \
    --sn '32' \
    --simclr 0 \
    --T_para 0.06 \
    --seed 1 \
    --ismasked 1 \
    --mask_percent 2 \
    --queue_length 512\
    --output_log_dir 'cpt' \
    --dev_only_q_encoder 1 \
    > log/unsup/mask2_queue_${lg}-1.log 2>&1

python -m torch.distributed.launch --nproc_per_node=2 --master_port 29504 \
    unsup_train.py \
    --lg $lg \
    --sn '32' \
    --simclr 0 \
    --T_para 0.06 \
    --seed 1 \
    --BN 1 \
    --ismasked 1 \
    --mask_percent 2 \
    --queue_length 0\
    --output_log_dir 'cpt' \
    --dev_only_q_encoder 1 \
    > log/unsup/mask2_BN_${lg}-1.log 2>&1

python -m torch.distributed.launch --nproc_per_node=2 --master_port 29504 \
    unsup_train.py \
    --lg $lg \
    --sn '32' \
    --simclr 0 \
    --T_para 0.06 \
    --seed 1 \
    --dual_lg 1 \
    --queue_length 0\
    --output_log_dir 'cpt' \
    --dev_only_q_encoder 1 \
    > log/unsup/Dual_${lg}-1.log 2>&1

python -m torch.distributed.launch --nproc_per_node=2 --master_port 29504 \
    unsup_train.py \
    --lg $lg \
    --sn '32' \
    --simclr 0 \
    --T_para 0.06 \
    --seed 1 \
    --BN 1 \
    --ismasked 1 \
    --mask_percent 2 \
    --queue_length 512\
    --output_log_dir 'cpt' \
    --dev_only_q_encoder 1 \
    > log/unsup/mask2_bn_queue_${lg}-1.log 2>&1

python -m torch.distributed.launch --nproc_per_node=2 --master_port 29504 \
    unsup_train.py \
    --lg $lg \
    --sn '32' \
    --simclr 0 \
    --T_para 0.06 \
    --seed 1 \
    --ismasked 1 \
    --mask_percent 3 \
    --queue_length 0\
    --output_log_dir 'cpt' \
    --dev_only_q_encoder 1 \
    > log/unsup/mask3_${lg}-1.log 2>&1

done