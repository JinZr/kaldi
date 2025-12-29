#!/usr/bin/env bash
set -euo pipefail

# CLP 5-fold recipe modeled after LibriSpeech GMM alignment + TDNN (nnet3, no iVector).

stage=0
nj=20
fold=""
clp_root="./data/clp_kaldi_5_fold/speaker_k5"
wav_root=""
out_prefix=""
lang_dir="data/lang_nosp"
dict_dir=""
decode_severity=true
mfccdir=mfcc
mfccdir_hires=mfcc_hires

# GMM sizes (tune for your data scale)
tri1_leaves=800
tri1_gauss=8000
tri2_leaves=1200
tri2_gauss=12000
tri3_leaves=1800
tri3_gauss=18000
tri4_leaves=2200
tri4_gauss=22000

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

nj_by_spk() {
  local d="$1"
  local nspk
  nspk=$(wc -l <"data/${d}/spk2utt" 2>/dev/null || echo 1)
  if [ "$nspk" -lt 1 ]; then
    nspk=1
  fi
  if [ "$nj" -gt "$nspk" ]; then
    echo "$nspk"
  else
    echo "$nj"
  fi
}

if [ -z "$fold" ]; then
  echo "$0: --fold is required (1..5)." >&2
  exit 1
fi

if [ -z "$out_prefix" ]; then
  out_prefix="clp_spk_k5_f${fold}"
fi

train_set="${out_prefix}_train"
valid_set="${out_prefix}_valid"
sev_sets=("${out_prefix}_severity_0" "${out_prefix}_severity_1" "${out_prefix}_severity_2" "${out_prefix}_severity_3")
exp_suffix="_${out_prefix}"

if [ $stage -le 0 ]; then
  local/clp_k5_data_prep.sh --clp-root "$clp_root" --fold "$fold" \
    --out-prefix "$out_prefix" --wav-root "$wav_root"
fi

if [ $stage -le 1 ]; then
  if [ -n "$dict_dir" ] && [ ! -d "$lang_dir" ]; then
    utils/prepare_lang.sh "$dict_dir" "<UNK>" data/local/lang_tmp_nosp "$lang_dir"
  fi
  if [ ! -d "$lang_dir" ]; then
    echo "$0: missing $lang_dir; pass --dict-dir or create a lang dir first." >&2
    exit 1
  fi
fi

if [ $stage -le 2 ]; then
  for x in "$train_set" "$valid_set" "${sev_sets[@]}"; do
    [ -d "data/$x" ] || continue
    steps/make_mfcc_pitch.sh --cmd "$train_cmd" --nj "$nj" "data/$x" \
      "exp/make_mfcc/$x" "$mfccdir"
    steps/compute_cmvn_stats.sh "data/$x" "exp/make_mfcc/$x" "$mfccdir"
    utils/fix_data_dir.sh "data/$x"
  done
fi

if [ $stage -le 3 ]; then
  num_utts=$(wc -l <"data/$train_set/utt2spk")
  short_n=$((num_utts < 400 ? num_utts : 400))
  med_n=$((num_utts < 800 ? num_utts : 800))
  long_n=$((num_utts < 1200 ? num_utts : 1200))

  utils/subset_data_dir.sh --shortest "data/$train_set" "$short_n" "data/${train_set}_short"
  utils/subset_data_dir.sh "data/$train_set" "$med_n" "data/${train_set}_med"
  utils/subset_data_dir.sh "data/$train_set" "$long_n" "data/${train_set}_long"

  steps/train_mono.sh --boost-silence 1.25 --nj "$nj" --cmd "$train_cmd" \
    "data/${train_set}_short" "$lang_dir" "exp/mono${exp_suffix}"

  steps/align_si.sh --boost-silence 1.25 --nj "$nj" --cmd "$train_cmd" \
    "data/${train_set}_med" "$lang_dir" "exp/mono${exp_suffix}" "exp/mono_ali_${out_prefix}_med"

  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
    "$tri1_leaves" "$tri1_gauss" "data/${train_set}_med" "$lang_dir" \
    "exp/mono_ali_${out_prefix}_med" "exp/tri1${exp_suffix}"

  steps/align_si.sh --nj "$nj" --cmd "$train_cmd" \
    "data/${train_set}_long" "$lang_dir" "exp/tri1${exp_suffix}" "exp/tri1_ali_${out_prefix}_long"

  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    "$tri2_leaves" "$tri2_gauss" "data/${train_set}_long" "$lang_dir" \
    "exp/tri1_ali_${out_prefix}_long" "exp/tri2${exp_suffix}"

  steps/align_fmllr.sh --nj "$nj" --cmd "$train_cmd" \
    "data/$train_set" "$lang_dir" "exp/tri2${exp_suffix}" "exp/tri2_ali_${out_prefix}"

  steps/train_sat.sh --cmd "$train_cmd" \
    "$tri3_leaves" "$tri3_gauss" "data/$train_set" "$lang_dir" \
    "exp/tri2_ali_${out_prefix}" "exp/tri3${exp_suffix}"

  steps/align_fmllr.sh --nj "$nj" --cmd "$train_cmd" \
    "data/$train_set" "$lang_dir" "exp/tri3${exp_suffix}" "exp/tri3_ali_${out_prefix}"

  steps/train_sat.sh --cmd "$train_cmd" \
    "$tri4_leaves" "$tri4_gauss" "data/$train_set" "$lang_dir" \
    "exp/tri3_ali_${out_prefix}" "exp/tri4${exp_suffix}"
fi

if [ $stage -le 4 ]; then
  for x in "$train_set" "$valid_set"; do
    nj_align=$(nj_by_spk "$x")
    steps/align_fmllr.sh --nj "$nj_align" --cmd "$train_cmd" \
      "data/$x" "$lang_dir" "exp/tri4${exp_suffix}" "exp/tri4${exp_suffix}_ali_$x"
  done
  if $decode_severity; then
    for x in "${sev_sets[@]}"; do
      [ -d "data/$x" ] || continue
      nj_align=$(nj_by_spk "$x")
      steps/align_fmllr.sh --nj "$nj_align" --cmd "$train_cmd" \
        "data/$x" "$lang_dir" "exp/tri4${exp_suffix}" "exp/tri4${exp_suffix}_ali_$x"
    done
  fi
fi

if [ $stage -le 5 ]; then
  for x in "$train_set" "$valid_set" "${sev_sets[@]}"; do
    [ -d "data/$x" ] || continue
    utils/copy_data_dir.sh "data/$x" "data/${x}_hires"
    rm -f "data/${x}_hires/feats.scp" "data/${x}_hires/cmvn.scp" \
          "data/${x}_hires/utt2dur" "data/${x}_hires/utt2num_frames"
    steps/make_mfcc_pitch.sh --cmd "$train_cmd" --nj "$nj" \
      --mfcc-config conf/mfcc_hires.conf \
      "data/${x}_hires" "exp/make_mfcc/${x}_hires" "$mfccdir_hires"
    steps/compute_cmvn_stats.sh "data/${x}_hires" "exp/make_mfcc/${x}_hires" "$mfccdir_hires"
    utils/fix_data_dir.sh "data/${x}_hires"
  done
fi

if [ $stage -le 6 ]; then
  test_sets="$valid_set"
  graph_dir=""
  if [ -f data/lang_test_tgsmall/G.fst ]; then
    graph_dir="exp/tri4${exp_suffix}/graph_tgsmall"
    [ ! -f "$graph_dir/HCLG.fst" ] && utils/mkgraph.sh data/lang_test_tgsmall "exp/tri4${exp_suffix}" "$graph_dir"
  else
    echo "$0: data/lang_test_tgsmall/G.fst not found; skipping decode."
    test_sets=""
  fi
  if [ -n "$test_sets" ] && $decode_severity; then
    for x in "${sev_sets[@]}"; do
      [ -d "data/$x" ] && test_sets="${test_sets} $x"
    done
  fi
  local/nnet3/run_tdnn_noivector.sh \
    --train-set "$train_set" \
    --test-sets "$test_sets" \
    --gmm "tri4${exp_suffix}" \
    --lang-dir "$lang_dir" \
    --graph-dir "$graph_dir" \
    --affix "$out_prefix" \
    --decode-nj "$nj"
fi

echo "$0: done (fold=${fold})"
