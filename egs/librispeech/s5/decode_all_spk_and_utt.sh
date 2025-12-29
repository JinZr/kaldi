#!/usr/bash
for i in 1 2 3 4 5; do  
    ./decode_only.sh \
        --dir ./exp/chain_cleaned/tdnn_clp_100h_sp/ \
        --decode-set ./data/clp_kaldi_5_fold/speaker_k5/fold${i}/valid \
        --decode-set-name speaker_fold${i} &

    for s in 0 1 2 3; do
        ./decode_only.sh \
            --nj 1 \
            --dir ./exp/chain_cleaned/tdnn_clp_100h_sp/ \
            --decode-set ./data/clp_kaldi_5_fold/speaker_k5/fold${i}/severity_${s} \
            --decode-set-name speaker_fold${i}_severity_${s} &
    done
done
wait

#!/usr/bash
for i in 1 2 3 4 5; do  
    ./decode_only.sh \
        --dir ./exp/chain_cleaned/tdnn_clp_100h_sp/ \
        --decode-set ./data/clp_kaldi_5_fold/utterance_k5/fold${i}/valid \
        --decode-set-name utterance_fold${i} &

    for s in 0 1 2 3; do
        ./decode_only.sh \
            --nj 1 \
            --dir ./exp/chain_cleaned/tdnn_clp_100h_sp/ \
            --decode-set ./data/clp_kaldi_5_fold/utterance_k5/fold${i}/severity_${s} \
            --decode-set-name utterance_fold${i}_severity_${s} &
    done
done
wait