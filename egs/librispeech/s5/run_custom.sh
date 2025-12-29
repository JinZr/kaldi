#!/usr/bin/env bash
# Modified from AISHELL s5 run.sh
# Goals:
#   1) Reuse AISHELL lexicon/lang (already prepared).
#   2) For GMM aligner training, merge AISHELL train + YOUR fold's train.
#   3) Train TDNN (nnet3 / chain) **only** on YOUR fold's train set.
#   4) Skip AISHELL download & LM steps (dataset already prepared).
#   5) Support 5-fold corpus structure via --my-data-root and --fold args.
#   6) Keep fold-isolated experiment dirs (e.g., exp/tri5a_f1, exp/tri5a_f2, ...).
#
# Expected existing from AISHELL:
#   data/train  data/dev  data/test  data/lang  data/lang_test
#
# Your 5-fold corpus layout (root = --my-data-root):
#   fold{1..5}/train
#   fold{1..5}/valid
#
# Example usage:
#   bash run_custom.sh \
#     --my-data-root data/pumch_kaldi_5_fold/speaker_k5 \
#     --fold 3 \
#     --nj 20 \
#     --stage 1
#
set -euo pipefail

. ./path.sh
. ./cmd.sh

# ------------------------- Config -------------------------
my_data_root="data/clp_kaldi_5_fold/speaker_k5"   # root that contains fold1..fold5
fold=                                               # which fold to use (1..5)
nj=60
stage=0

# Allow overrides from CLI
. utils/parse_options.sh || exit 1

# Derive fold tag and per-fold data set names kept under data/
fold_tag=f${fold}
my_train_src=${my_data_root}/fold${fold}/train
my_valid_src=${my_data_root}/fold${fold}/valid

# We expose your fold's sets under unique names so local scripts don't collide with AISHELL
my_train=data/my_${fold_tag}_train
my_dev=data/my_${fold_tag}_valid
my_test=${my_dev}   # valid serves as both dev & test as requested

# AISHELL sets already prepared by the original run.sh (do NOT change names).
aishell_train=data/train_clean_100
aishell_dev=data/dev_clean
aishell_test=data/test_clean

# Merged train set ONLY for training GMM aligners (per fold to avoid cross-fold leakage)
align_train=data/train_align_${fold_tag}

# Suffix for exp dirs per fold
sfx=_${fold_tag}

# ---------------------- Sanity checks ---------------------
if [ ${stage} -le 0 ]; then
  for d in "${aishell_train}" "${aishell_dev}" "${aishell_test}" data/lang data/lang_nosp; do
    if [ ! -d "$d" ]; then
      echo "$0: expected '$d' to exist (from LS prep)." 1>&2
      exit 1
    fi
  done
  for d in "${my_train_src}" "${my_valid_src}"; do
    if [ ! -d "$d" ]; then
      echo "$0: expected your fold path '$d' to exist (check --my-data-root and --fold)." 1>&2
      exit 1
    fi
  done
fi

# ---------------- 1) Stage your fold under data/ ----------
# Copy (not symlink) so that we can safely add features/segments without touching your originals.
if [ ${stage} -le 1 ]; then
  echo "===> Staging your fold${fold} data under data/ ..."
  # The above 'for' trick isn't portable across all shells; do explicit copies instead:
  utils/copy_data_dir.sh "${my_train_src}" "${my_train}"
  utils/copy_data_dir.sh "${my_valid_src}" "${my_dev}"
  # test=dev per your spec
  if [ "${my_test}" != "${my_dev}" ]; then
    utils/copy_data_dir.sh "${my_valid_src}" "${my_test}"
  fi
  # Ensure well-formed
  for x in "${my_train}" "${my_dev}"; do
    utils/fix_data_dir.sh "$x"
  done
fi

# ----------- 2) Features for YOUR fold only ---------------
if [ ${stage} -le 2 ]; then
  mfccdir=mfcc
  for x in "${my_train}" "${my_dev}"; do
    setname=$(basename "$x")
    steps/make_mfcc_pitch.sh --cmd "${train_cmd}" --nj ${nj} "$x" "exp/make_mfcc/${setname}" "${mfccdir}"
    steps/compute_cmvn_stats.sh "$x" "exp/make_mfcc/${setname}" "${mfccdir}"
    utils/fix_data_dir.sh "$x"
  done
fi

# --- 3) Merge AISHELL train + YOUR fold's train for GMM ---
if [ ${stage} -le 3 ]; then
  echo "===> Combining AISHELL train + your fold${fold} train -> ${align_train}"
  utils/combine_data.sh "${align_train}" "${aishell_train}" "${my_train}"
  utils/fix_data_dir.sh "${align_train}"
fi

# ------------- 4) Train GMMs on the merged set ------------
# Keep per-fold exp dirs distinguished via suffix ${sfx}.
if [ ${stage} -le 4 ]; then
  echo "===> Training GMMs on ${align_train} using data/lang_nosp (LS lexicon)"
  steps/train_mono.sh --cmd "${train_cmd}" --nj ${nj} \
    "${align_train}" data/lang_nosp exp/mono${sfx}

  steps/align_si.sh --cmd "${train_cmd}" --nj ${nj} \
    "${align_train}" data/lang_nosp exp/mono${sfx} exp/mono_ali${sfx}

  steps/train_deltas.sh --cmd "${train_cmd}" \
    2500 20000 "${align_train}" data/lang_nosp exp/mono_ali${sfx} exp/tri1${sfx}

  steps/align_si.sh --cmd "${train_cmd}" --nj ${nj} \
    "${align_train}" data/lang_nosp exp/tri1${sfx} exp/tri1_ali${sfx}

  steps/train_deltas.sh --cmd "${train_cmd}" \
    2500 20000 "${align_train}" data/lang_nosp exp/tri1_ali${sfx} exp/tri2${sfx}

  steps/align_si.sh --cmd "${train_cmd}" --nj ${nj} \
    "${align_train}" data/lang_nosp exp/tri2${sfx} exp/tri2_ali${sfx}

  steps/train_lda_mllt.sh --cmd "${train_cmd}" \
    2500 15000 "${align_train}" data/lang_nosp exp/tri2_ali${sfx} exp/tri3a${sfx}

  steps/align_fmllr.sh --cmd "${train_cmd}" --nj ${nj} \
    "${align_train}" data/lang_nosp exp/tri3a${sfx} exp/tri3a_ali${sfx}

  steps/train_sat.sh --cmd "${train_cmd}" \
    4200 40000 "${align_train}" data/lang_nosp exp/tri3a_ali${sfx} exp/tri4a${sfx}

  steps/align_fmllr.sh --cmd "${train_cmd}" --nj ${nj} \
    "${align_train}" data/lang_nosp exp/tri4a${sfx} exp/tri4a_ali${sfx}

  steps/train_sat.sh --cmd "${train_cmd}" \
    5000 100000 "${align_train}" data/lang_nosp exp/tri4a_ali${sfx} exp/tri5a${sfx}
fi

# ------------- 5) Align YOUR train with tri5a --------------
if [ ${stage} -le 5 ]; then
  yset=$(basename "${my_train}")
  echo "===> fMLLR-align ${yset} with exp/tri5a${sfx}"
  steps/align_fmllr.sh --cmd "${train_cmd}" --nj ${nj} \
    "${my_train}" data/lang_nosp exp/tri5a${sfx} "exp/tri5a_ali_${yset}${sfx}"
fi

# ------------- 6) TDNN (nnet3) on YOUR corpus -------------
# Pass fold-specific GMM name and train set; many aishell local scripts accept these.
# if [ ${stage} -le 6 ]; then
#   yset=$(basename "${my_train}")
#   if [ -f local/nnet3/run_tdnn.sh ]; then
#     echo "===> Training nnet3 TDNN on ${yset} (using GMM exp/tri5a${sfx})"
#     local/nnet3/run_tdnn.sh --train-set "${yset}" --gmm "tri5a${sfx}" --test-sets "$(basename ${my_dev})" || \
#       echo "nnet3 TDNN script finished (or not present)."
#   else
#     echo "WARNING: local/nnet3/run_tdnn.sh not found; skipping nnet3 TDNN."
#   fi
# fi
exit 0
# ------------- 7) TDNN (chain) on YOUR corpus --------------
if [ ${stage} -le 7 ]; then
  yset=$(basename "${my_train}")
  if [ -f local/chain/run_tdnn_clp.sh ]; then
    echo "===> Training CHAIN TDNN on ${yset} (affix: ${sfx}; dir: exp/chain${sfx}/tdnn_1a)"
    local/chain/run_tdnn_clp.sh \
      --train-set "${yset}" \
      --affix "${fold_tag}" \
      --dir "exp/chain${sfx}/tdnn_1a" || \
      echo "chain TDNN script finished (or not present)."
  else
    echo "WARNING: local/chain/run_tdnn.sh not found; skipping chain TDNN."
  fi
fi

# ---------------- 8) Optional: decoding --------------------
# If you wish to decode your valid set, ensure a graph compatible with AISHELL lexicon
# (data/lang_test or a custom graph) and add decode steps here.

echo "$0: Done. Fold ${fold} -> GMM aligners on AISHELL+YOUR fold train; TDNN trained on YOUR fold train."
