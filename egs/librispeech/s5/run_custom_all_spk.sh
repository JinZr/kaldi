#!/bin/bash

for i in 1 2 3 4 5; do
    # ./run_custom.sh --my-train clp_kaldi_5_fold/speaker_k5/fold${i}/train/ --my-dev clp_kaldi_5_fold/speaker_k5/fold${i}/valid --combo train_100_fold${i} --stage 15
    ./run_custom.sh --fold "${i}" --nj 8
done
