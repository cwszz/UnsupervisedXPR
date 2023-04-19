for lg in 'ko' 'zh' 'ru' ;
do
export CUDA_VISIBLE_DEVICES='3,4'
python -m torch.distributed.launch --nproc_per_node=2 --master_port 29503 \
    unsup_train.py \
    --lg $lg \
    --sn '32' \
    --simclr 0 \
    --T_para 0.06 \
    --seed 1 \
    --onlylg 1\
    --ismasked 0 \
    --mask_percent 0 \
    --queue_length 0\
    --output_log_dir 'cpt' \
    --dev_only_q_encoder 1 \
    > log/new_unsup/lg_nomask_${lg}-1.log 2>&1

done

for lg in 'de' 'es' ;
do
export CUDA_VISIBLE_DEVICES='3,4'
python -m torch.distributed.launch --nproc_per_node=2 --master_port 29503 \
    unsup_train.py \
    --lg $lg \
    --sn '32' \
    --simclr 0 \
    --T_para 0.06 \
    --seed 1 \
    --onlylg 1\
    --ismasked 1 \
    --mask_percent 0 \
    --queue_length 0\
    --output_log_dir 'cpt' \
    --dev_only_q_encoder 1 \
    > log/new_unsup/lg_mask0${lg}-1.log 2>&1

python -m torch.distributed.launch --nproc_per_node=2 --master_port 29503 \
    unsup_train.py \
    --lg $lg \
    --sn '32' \
    --simclr 0 \
    --T_para 0.06 \
    --seed 1 \
    --onlylg 1\
    --ismasked 1 \
    --mask_percent 1 \
    --queue_length 0\
    --output_log_dir 'cpt' \
    --dev_only_q_encoder 1 \
    > log/new_unsup/lg_mask1${lg}-1.log 2>&1

python -m torch.distributed.launch --nproc_per_node=2 --master_port 29503 \
    unsup_train.py \
    --lg $lg \
    --sn '32' \
    --simclr 0 \
    --T_para 0.06 \
    --seed 1 \
    --onlylg 1\
    --ismasked 1 \
    --mask_percent 2 \
    --queue_length 0\
    --output_log_dir 'cpt' \
    --dev_only_q_encoder 1 \
    > log/new_unsup/lg_mask2${lg}-1.log 2>&1

python -m torch.distributed.launch --nproc_per_node=2 --master_port 29503 \
    unsup_train.py \
    --lg $lg \
    --sn '32' \
    --simclr 0 \
    --T_para 0.06 \
    --seed 1 \
    --onlylg 1\
    --ismasked 1 \
    --mask_percent 3 \
    --queue_length 0\
    --output_log_dir 'cpt' \
    --dev_only_q_encoder 1 \
    > log/new_unsup/lg_mask3${lg}-1.log 2>&1

python -m torch.distributed.launch --nproc_per_node=2 --master_port 29503 \
    unsup_train.py \
    --lg $lg \
    --sn '32' \
    --simclr 0 \
    --T_para 0.06 \
    --seed 1 \
    --onlylg 1\
    --ismasked 1 \
    --mask_percent 4 \
    --queue_length 0\
    --output_log_dir 'cpt' \
    --dev_only_q_encoder 1 \
    > log/new_unsup/lg_mask4${lg}-1.log 2>&1


python -m torch.distributed.launch --nproc_per_node=2 --master_port 29503 \
    unsup_train.py \
    --lg $lg \
    --sn '32' \
    --simclr 0 \
    --T_para 0.06 \
    --seed 1 \
    --onlylg 1\
    --ismasked 1 \
    --mask_percent 5 \
    --queue_length 0\
    --output_log_dir 'cpt' \
    --dev_only_q_encoder 1 \
    > log/new_unsup/lg_mask5${lg}-1.log 2>&1


python -m torch.distributed.launch --nproc_per_node=2 --master_port 29503 \
    unsup_train.py \
    --lg $lg \
    --sn '32' \
    --simclr 0 \
    --T_para 0.06 \
    --seed 1 \
    --onlylg 1\
    --ismasked 1 \
    --mask_percent 6 \
    --queue_length 0\
    --output_log_dir 'cpt' \
    --dev_only_q_encoder 1 \
    > log/new_unsup/lg_mask6${lg}-1.log 2>&1


python -m torch.distributed.launch --nproc_per_node=2 --master_port 29503 \
    unsup_train.py \
    --lg $lg \
    --sn '32' \
    --simclr 0 \
    --T_para 0.06 \
    --seed 1 \
    --onlylg 1\
    --ismasked 1 \
    --mask_percent 7 \
    --queue_length 0\
    --output_log_dir 'cpt' \
    --dev_only_q_encoder 1 \
    > log/new_unsup/lg_mask7${lg}-1.log 2>&1


python -m torch.distributed.launch --nproc_per_node=2 --master_port 29503 \
    unsup_train.py \
    --lg $lg \
    --sn '32' \
    --simclr 0 \
    --T_para 0.06 \
    --seed 1 \
    --onlylg 1\
    --ismasked 1 \
    --mask_percent 8 \
    --queue_length 0\
    --output_log_dir 'cpt' \
    --dev_only_q_encoder 1 \
    > log/new_unsup/lg_mask8${lg}-1.log 2>&1


python -m torch.distributed.launch --nproc_per_node=2 --master_port 29503 \
    unsup_train.py \
    --lg $lg \
    --sn '32' \
    --simclr 0 \
    --T_para 0.06 \
    --seed 1 \
    --onlylg 1\
    --ismasked 1 \
    --mask_percent 9 \
    --queue_length 0\
    --output_log_dir 'cpt' \
    --dev_only_q_encoder 1 \
    > log/new_unsup/lg_mask9${lg}-1.log 2>&1
done

