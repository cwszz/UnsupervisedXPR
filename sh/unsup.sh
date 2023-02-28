for lg in 'fr' 'de' 'es' 'ar' 'all_unsupervised' ;
do
export CUDA_VISIBLE_DEVICES='6,7'
python -m torch.distributed.launch --nproc_per_node=2 --master_port 29502 \
    unsup_train.py \
    --lg $lg \
    --sn '32' \
    --simclr 0 \
    --T_para 0.06 \
    --seed 1 \
    --queue_length 0\
    --output_log_dir 'unsup_result' \
    --dev_only_q_encoder 1 \
    > log/unsup/unsup_${lg}-1.log 2>&1

python -m torch.distributed.launch --nproc_per_node=2 --master_port 29502 \
    unsup_train.py \
    --lg $lg \
    --sn '32' \
    --simclr 0 \
    --T_para 0.06 \
    --seed 1 \
    --queue_length 512\
    --output_log_dir 'unsup_result' \
    --dev_only_q_encoder 1 \
    > log/unsup/queue512_unsup_${lg}-1.log 2>&1

python -m torch.distributed.launch --nproc_per_node=2 --master_port 29502 \
    unsup_train.py \
    --lg $lg \
    --sn '32' \
    --simclr 0 \
    --T_para 0.06 \
    --seed 1 \
    --dual_lg 1\
    --queue_length 0\
    --output_log_dir 'unsup_result' \
    --dev_only_q_encoder 1 \
    > log/unsup/dual_unsup_${lg}-1.log 2>&1

python -m torch.distributed.launch --nproc_per_node=2 --master_port 29502 \
    unsup_train.py \
    --lg $lg \
    --sn '32' \
    --simclr 0 \
    --T_para 0.06 \
    --seed 1 \
    --ismasked 1 \
    --mask_percent 2 \
    --queue_length 0\
    --output_log_dir 'mask_result' \
    --dev_only_q_encoder 1 \
    > log/unsup/mask2_${lg}-1.log 2>&1

python -m torch.distributed.launch --nproc_per_node=2 --master_port 29502 \
    unsup_train.py \
    --lg $lg \
    --sn '32' \
    --simclr 0 \
    --T_para 0.06 \
    --seed 1 \
    --ismasked 1 \
    --mask_percent 3 \
    --queue_length 0\
    --output_log_dir 'mask_result' \
    --dev_only_q_encoder 1 \
    > log/unsup/mask3_${lg}-1.log 2>&1

python -m torch.distributed.launch --nproc_per_node=2 --master_port 29502 \
    unsup_train.py \
    --lg $lg \
    --sn '32' \
    --simclr 0 \
    --T_para 0.06 \
    --seed 1 \
    --ismasked 1 \
    --mask_percent 2 \
    --queue_length 512\
    --output_log_dir 'mask_result' \
    --dev_only_q_encoder 1 \
    > log/unsup/queue512_mask2_${lg}-1.log 2>&1

done