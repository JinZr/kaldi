#!/usr/bin/env bash
# Decode-only script for a Kaldi chain TDNN model (no i-vectors).
# Assumes your model was trained similarly to a standard run_tdnn chain recipe
# and decodes with hires MFCC features.
#
# Usage example:
#   ./decode_only.sh --dir exp/chain/tdnn_1a_sp --lang-test data/lang_test_tgsmall --decode-set mytest --nj 10
#
# Required:
#   - data/<decode-set>/ with wav.scp (and text if you want scoring)
#   - lang_test dir (e.g., data/lang_test_tgsmall) that CONTAINS G.fst (LM)
#   - model dir (e.g., exp/chain/tdnn_1a_sp) containing final.mdl, tree, etc.
# This script will:
#   1) Make hires MFCCs for data/<decode-set>_hires
#   2) Build graph at <dir>/graph if not present
#   3) Run steps/nnet3/decode.sh (acwt=1.0, post-decode-acwt=10.0)
#
# Notes:
#   - If conf/mfcc_hires.conf is missing, it will fallback to conf/mfcc.conf.
#   - No i-vectors are used (matches the provided training script). If your model needs i-vectors, adapt accordingly.

set -euo pipefail

# Default options
stage=0
nj=10
decode_set=
dir=exp/chain/tdnn_1a_sp
lang_test=data/lang_test_tgsmall   # use a 'lang_test_*' dir (has G.fst)
lm_affix=tgsmall                   # used to suffix graph/decode dirs
graph_dir=
decode_set_name=

# Source Kaldi helpers (expects you run this from your recipe root)
[ -f ./path.sh ] && . ./path.sh
[ -f ./cmd.sh ] && . ./cmd.sh

# Fallbacks if cmd.sh wasn't sourced
: "${decode_cmd:=run.pl}"

. utils/parse_options.sh || exit 1;

if [ -z "${decode_set}" ]; then
  echo "You must specify --decode-set (e.g., mytest)."
  echo "Example: $0 --dir exp/chain/tdnn_1a_sp --lang-test data/lang_test_tgsmall --decode-set mytest --nj 10"
  exit 1
fi

data=${decode_set}
data_hires=${decode_set}_hires

if [ ! -f ${data}/wav.scp ]; then
  echo "Missing ${data}/wav.scp. Prepare your data dir first (wav.scp, utt2spk, spk2utt, [text])." >&2
  exit 1
fi

mfcc_cfg=conf/mfcc_hires.conf
[ -f "$mfcc_cfg" ] || mfcc_cfg=conf/mfcc.conf

if [ $stage -le 0 ]; then
  echo "Stage 0: Make hires MFCC for ${data_hires}"
  utils/copy_data_dir.sh ${data} ${data_hires}
  steps/make_mfcc_pitch.sh --mfcc-config $mfcc_cfg --nj $nj --cmd "$decode_cmd" ${data_hires}
  steps/compute_cmvn_stats.sh ${data_hires}
  utils/fix_data_dir.sh ${data_hires}
fi

if [ -z "${graph_dir}" ]; then
  graph_dir=${dir}/graph_${lm_affix}
fi

# Sanity check and auto-fallback: lang_test must contain G.fst (LM FST)
if [ ! -f ${lang_test}/G.fst ]; then
  if [ -f data/lang_test_tgsmall/G.fst ]; then
    echo "Warning: ${lang_test} has no G.fst; falling back to data/lang_test_tgsmall"
    lang_test=data/lang_test_tgsmall
    lm_affix=tgsmall
    graph_dir=${dir}/graph_${lm_affix}
  else
    echo "The specified --lang-test (${lang_test}) does not contain G.fst."
    echo "Use a 'data/lang_test_*' directory (e.g., data/lang_test_tgsmall), not data/lang_chain."
    exit 1
  fi
fi

if [ $stage -le 1 ]; then
  if [ ! -f ${graph_dir}/HCLG.fst ]; then
    echo "Stage 1: Building decoding graph at ${graph_dir}"
    utils/mkgraph.sh --self-loop-scale 1.0 --remove-oov ${lang_test} ${dir} ${graph_dir}
  else
    echo "Graph already exists at ${graph_dir}"
  fi
fi

if [ $stage -le 2 ]; then
  decode_set_name=${decode_set_name:-$(basename ${decode_set})}
  decode_out=${dir}/decode_${decode_set_name}_${lm_affix}
  echo "Stage 2: Decode ${data_hires} using model ${dir} -> ${decode_out}"
  steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
    --nj ${nj} --cmd "$decode_cmd" \
    ${graph_dir} ${data_hires} ${decode_out}
fi

# Optional LM rescoring if additional LMs exist (only when base LM is tgsmall)
# if [ $stage -le 3 ] && [ "${lm_affix}" = "tgsmall" ]; then
#   if [ -d data/lang_test_tgmed ]; then
#     steps/lmrescore.sh data/lang_test_{tgsmall,tgmed} \
#       ${data_hires} ${dir}/decode_${decode_set_name}_{tgsmall,tgmed} || true
#   fi
#   if [ -d data/lang_test_tglarge ]; then
#     steps/lmrescore.sh data/lang_test_{tgsmall,tglarge} \
#       ${data_hires} ${dir}/decode_${decode_set_name}_{tgsmall,tglarge} || true
#   fi
#   if [ -d data/lang_test_fglarge ]; then
#     steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
#       data/lang_test_{tgsmall,fglarge} \
#       ${data_hires} ${dir}/decode_${decode_set_name}_{tgsmall,fglarge} || true
#   fi
# fi

echo "Done. See ${dir}/decode_${decode_set_name}_${lm_affix} for logs and results."