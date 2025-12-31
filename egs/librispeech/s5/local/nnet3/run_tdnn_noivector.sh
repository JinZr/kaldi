#!/usr/bin/env bash
set -euo pipefail

# NNET3 TDNN training without iVectors (for small/medium corpora).

stage=0
train_stage=-10
decode_nj=10
train_set=train
test_sets=
gmm=tri4
ali_dir=
train_data_dir=
nnet3_affix=
affix=
tdnn_dim=512
num_epochs=4
num_jobs_initial=1
num_jobs_final=1
initial_lrate=0.001
final_lrate=0.0001
remove_egs=true
common_egs_dir=
reporting_email=
lang_dir=data/lang_nosp
graph_dir=

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

gmm_dir=exp/${gmm}
if [ -z "$ali_dir" ]; then
  ali_dir=exp/${gmm}_ali_${train_set}
fi
dir=exp/nnet3${nnet3_affix}/tdnn${affix:+_$affix}

if [ -z "$train_data_dir" ]; then
  train_data_dir=data/${train_set}_hires
fi

for f in "$train_data_dir/feats.scp" "$ali_dir/ali.1.gz" "$gmm_dir/final.mdl"; do
  [ ! -f "$f" ] && echo "$0: expected file $f to exist" && exit 1
done

input_dim=$(feat-to-dim scp:"$train_data_dir/feats.scp" -) || exit 1

if [ $stage -le 11 ]; then
  echo "$0: creating neural net configs"

  num_targets=$(tree-info "$ali_dir/tree" | grep num-pdfs | awk '{print $2}')
  mkdir -p "$dir/configs"

  cat <<EOF > "$dir/configs/network.xconfig"
  input dim=${input_dim} name=input
  fixed-affine-layer name=lda input=Append(-2,-1,0,1,2) affine-transform-file=$dir/configs/lda.mat

  relu-batchnorm-layer name=tdnn1 dim=${tdnn_dim}
  relu-batchnorm-layer name=tdnn2 dim=${tdnn_dim} input=Append(-1,0,1)
  relu-batchnorm-layer name=tdnn3 dim=${tdnn_dim} input=Append(-1,0,1)
  relu-batchnorm-layer name=tdnn4 dim=${tdnn_dim} input=Append(-3,0,3)
  relu-batchnorm-layer name=tdnn5 dim=${tdnn_dim} input=Append(-3,0,3)
  relu-batchnorm-layer name=tdnn6 dim=${tdnn_dim} input=Append(-6,-3,0)
  output-layer name=output dim=${num_targets} max-change=1.5
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file "$dir/configs/network.xconfig" \
    --config-dir "$dir/configs" || exit 1
fi

if [ $stage -le 12 ]; then
  steps/nnet3/train_dnn.py --stage="$train_stage" \
    --cmd="$decode_cmd" \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --trainer.num-epochs "$num_epochs" \
    --trainer.optimization.num-jobs-initial "$num_jobs_initial" \
    --trainer.optimization.num-jobs-final "$num_jobs_final" \
    --trainer.optimization.initial-effective-lrate "$initial_lrate" \
    --trainer.optimization.final-effective-lrate "$final_lrate" \
    --egs.dir "$common_egs_dir" \
    --cleanup.remove-egs "$remove_egs" \
    --cleanup.preserve-model-interval 50 \
    --feat-dir="$train_data_dir" \
    --ali-dir "$ali_dir" \
    --lang "$lang_dir" \
    --reporting.email="$reporting_email" \
    --dir="$dir" || exit 1
fi

if [ $stage -le 13 ] && [ -n "$test_sets" ]; then
  if [ -z "$graph_dir" ]; then
    if [ -d data/lang_test_tgsmall ]; then
      graph_dir="${gmm_dir}/graph_tgsmall"
      [ ! -f "$graph_dir/HCLG.fst" ] && utils/mkgraph.sh data/lang_test_tgsmall "$gmm_dir" "$graph_dir"
    else
      graph_dir="${gmm_dir}/graph"
      [ ! -f "$graph_dir/HCLG.fst" ] && utils/mkgraph.sh "$lang_dir" "$gmm_dir" "$graph_dir"
    fi
  fi
  if [ ! -f "$graph_dir/HCLG.fst" ]; then
    echo "$0: missing graph at $graph_dir; skipping decode"
    exit 0
  fi

  rm -f "$dir/.error" 2>/dev/null || true
  for test in $test_sets; do
    if [ ! -d "data/${test}_hires" ]; then
      echo "$0: data/${test}_hires missing; skipping $test"
      continue
    fi
    (
      steps/nnet3/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
        "$graph_dir" "data/${test}_hires" "$dir/decode_${test}" || exit 1
    ) || touch "$dir/.error" &
  done
  wait
  [ -f "$dir/.error" ] && echo "$0: there was a problem while decoding" && exit 1
fi

exit 0
