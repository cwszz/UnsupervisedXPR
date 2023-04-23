for lg in 'fr' ;
do
export CUDA_VISIBLE_DEVICES='4,5'

python -m torch.distributed.launch --nproc_per_node=2 --master_port 29506 \
    unsup_train.py \
    --lg $lg \
    --sn '32' \
    --simclr 0 \
    --T_para 0.06 \
    --seed 1 \
    --onlylg 0\
    --ismasked 1 \
    --mask_percent 8 \
    --all_sentence_num 16\
    --dev_sample_num 16\
    --test_sample_num 16\
    --queue_length 0\
    --output_log_dir 'cpt' \
    --dev_only_q_encoder 1 \
    > log/new_unsup/en_mask8_16${lg}-1.log 2>&1

python -m torch.distributed.launch --nproc_per_node=2 --master_port 29506 \
    unsup_train.py \
    --lg $lg \
    --sn '32' \
    --simclr 0 \
    --T_para 0.06 \
    --seed 1 \
    --onlylg 0\
    --ismasked 1 \
    --mask_percent 8 \
    --all_sentence_num 8\
    --dev_sample_num 8\
    --test_sample_num 8\
    --queue_length 0\
    --output_log_dir 'cpt' \
    --dev_only_q_encoder 1 \
    > log/new_unsup/en_mask8_8${lg}-1.log 2>&1

python -m torch.distributed.launch --nproc_per_node=2 --master_port 29506 \
    unsup_train.py \
    --lg $lg \
    --sn '32' \
    --simclr 0 \
    --T_para 0.06 \
    --seed 1 \
    --onlylg 0\
    --ismasked 1 \
    --mask_percent 8 \
    --all_sentence_num 4\
    --train_sample_num 4 \
    --dev_sample_num 4\
    --test_sample_num 4\
    --queue_length 0\
    --output_log_dir 'cpt' \
    --dev_only_q_encoder 1 \
    > log/new_unsup/en_mask8_4${lg}-1.log 2>&1

python -m torch.distributed.launch --nproc_per_node=2 --master_port 29506 \
    unsup_train.py \
    --lg $lg \
    --sn '32' \
    --simclr 0 \
    --T_para 0.06 \
    --seed 1 \
    --onlylg 0\
    --ismasked 1 \
    --mask_percent 8 \
    --all_sentence_num 2\
    --train_sample_num 2 \
    --dev_sample_num 2\
    --test_sample_num 2\
    --queue_length 0\
    --output_log_dir 'cpt' \
    --dev_only_q_encoder 1 \
    > log/new_unsup/en_mask8_2${lg}-1.log 2>&1

python -m torch.distributed.launch --nproc_per_node=2 --master_port 29506 \
    unsup_train.py \
    --lg $lg \
    --sn '32' \
    --simclr 0 \
    --T_para 0.06 \
    --seed 1 \
    --onlylg 0\
    --ismasked 1 \
    --mask_percent 8 \
    --all_sentence_num 1\
    --train_sample_num 1 \
    --dev_sample_num 1\
    --test_sample_num 1\
    --queue_length 0\
    --output_log_dir 'cpt' \
    --dev_only_q_encoder 1 \
    > log/new_unsup/en_mask8_1${lg}-1.log 2>&1

python -m torch.distributed.launch --nproc_per_node=2 --master_port 29506 \
    unsup_train.py \
    --lg $lg \
    --sn '32' \
    --simclr 0 \
    --T_para 0.06 \
    --seed 1 \
    --onlylg 0\
    --ismasked 1 \
    --mask_percent 8 \
    --all_sentence_num 0\
    --train_sample_num 0 \
    --dev_sample_num 0\
    --test_sample_num 0\
    --queue_length 0\
    --output_log_dir 'cpt' \
    --dev_only_q_encoder 1 \
    > log/new_unsup/en_mask8_0${lg}-1.log 2>&1

done

