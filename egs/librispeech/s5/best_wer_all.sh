
for i in 1 2 3 4 5; do
    echo "===fold $i==="
    grep WER exp/chain_cleaned/tdnn_clp_100h_sp/decode_speaker_fold${i}_tgsmall/wer_* | ./utils/best_wer.sh

    grep WER exp/chain_cleaned/tdnn_clp_100h_sp/decode_speaker_fold${i}_severity_0_tgsmall/wer_* | ./utils/best_wer.sh

    grep WER exp/chain_cleaned/tdnn_clp_100h_sp/decode_speaker_fold${i}_severity_1_tgsmall/wer_* | ./utils/best_wer.sh

    grep WER exp/chain_cleaned/tdnn_clp_100h_sp/decode_speaker_fold${i}_severity_2_tgsmall/wer_* | ./utils/best_wer.sh

    grep WER exp/chain_cleaned/tdnn_clp_100h_sp/decode_speaker_fold${i}_severity_3_tgsmall/wer_* | ./utils/best_wer.sh
done