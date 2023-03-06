for lg in 'fr' 'de' 'es' ;
do
export CUDA_VISIBLE_DEVICES='5,6'
python -m torch.distributed.launch --nproc_per_node=2 --master_port 29502 \
    unsup_train.py \
    --lg $lg \
    --sn '32' \
    --simclr 0 \
    --T_para 0.06 \
    --seed 1 \
    --ismasked 1 \
    --mask_percent 4 \
    --queue_length 0\
    --output_log_dir 'cpt' \
    --dev_only_q_encoder 1 \
    > log/unsup/mask4_${lg}-1.log 2>&1

python -m torch.distributed.launch --nproc_per_node=2 --master_port 29502 \
    unsup_train.py \
    --lg $lg \
    --sn '32' \
    --simclr 0 \
    --T_para 0.06 \
    --seed 1 \
    --ismasked 1 \
    --mask_percent 1 \
    --queue_length 0\
    --output_log_dir 'cpt' \
    --dev_only_q_encoder 1 \
    > log/unsup/mask1_${lg}-1.log 2>&1

python -m torch.distributed.launch --nproc_per_node=2 --master_port 29502 \
    unsup_train.py \
    --lg $lg \
    --sn '32' \
    --simclr 0 \
    --T_para 0.06 \
    --seed 1 \
    --ismasked 1 \
    --mask_percent 0 \
    --queue_length 0\
    --output_log_dir 'cpt' \
    --dev_only_q_encoder 1 \
    > log/unsup/mask0_${lg}-1.log 2>&1

python -m torch.distributed.launch --nproc_per_node=2 --master_port 29502 \
    unsup_train.py \
    --lg $lg \
    --sn '32' \
    --simclr 0 \
    --T_para 0.06 \
    --seed 1 \
    --ismasked 1 \
    --mask_percent 5 \
    --queue_length 0\
    --output_log_dir 'cpt' \
    --dev_only_q_encoder 1 \
    > log/unsup/mask5_${lg}-1.log 2>&1

python -m torch.distributed.launch --nproc_per_node=2 --master_port 29502 \
    unsup_train.py \
    --lg $lg \
    --sn '32' \
    --simclr 0 \
    --T_para 0.06 \
    --seed 1 \
    --ismasked 1 \
    --mask_percent 6 \
    --queue_length 0\
    --output_log_dir 'cpt' \
    --dev_only_q_encoder 1 \
    > log/unsup/mask6_${lg}-1.log 2>&1

python -m torch.distributed.launch --nproc_per_node=2 --master_port 29502 \
    unsup_train.py \
    --lg $lg \
    --sn '32' \
    --simclr 0 \
    --T_para 0.06 \
    --seed 1 \
    --ismasked 1 \
    --mask_percent 7 \
    --queue_length 0\
    --output_log_dir 'cpt' \
    --dev_only_q_encoder 1 \
    > log/unsup/mask7_${lg}-1.log 2>&1

python -m torch.distributed.launch --nproc_per_node=2 --master_port 29502 \
    unsup_train.py \
    --lg $lg \
    --sn '32' \
    --simclr 0 \
    --T_para 0.06 \
    --seed 1 \
    --ismasked 1 \
    --mask_percent 8 \
    --queue_length 0\
    --output_log_dir 'cpt' \
    --dev_only_q_encoder 1 \
    > log/unsup/mask8_${lg}-1.log 2>&1

done