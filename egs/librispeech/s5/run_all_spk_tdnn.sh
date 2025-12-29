#!/usr/bash

for i in 1; do
    CUDA_VISIBLE_DEVICES=5 local/chain/run_tdnn_clp.sh \
        --train_set my_f${i}_train \
        --affix f${i} \
        --stage 11
done
