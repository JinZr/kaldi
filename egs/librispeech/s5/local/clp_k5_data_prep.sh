#!/usr/bin/env bash
set -euo pipefail

# Prepare CLP 5-fold data into Kaldi-friendly data/ dirs.
# This script cleans stale feats/cmvn and optionally rebases wav.scp paths.

clp_root="/Users/zrjin/Downloads/clp_kaldi_5_fold/speaker_k5"
fold=""
out_prefix=""
wav_root=""
stage=0

. ./path.sh
. ./utils/parse_options.sh || exit 1

if [ -z "$fold" ]; then
  echo "$0: --fold is required (1..5)." >&2
  exit 1
fi

if [ -z "$out_prefix" ]; then
  out_prefix="clp_spk_k5_f${fold}"
fi

src_fold="${clp_root}/fold${fold}"
if [ ! -d "$src_fold" ]; then
  echo "$0: missing source fold dir: $src_fold" >&2
  exit 1
fi

mkdir -p data

copy_and_clean() {
  local src_dir="$1"
  local dst_dir="$2"

  utils/copy_data_dir.sh "$src_dir" "$dst_dir"

  rm -f "$dst_dir/feats.scp" \
        "$dst_dir/cmvn.scp" \
        "$dst_dir/frame_shift" \
        "$dst_dir/utt2dur" \
        "$dst_dir/utt2num_frames"
  rm -rf "$dst_dir/conf" "$dst_dir/log" "$dst_dir/data" \
         "$dst_dir"/split* "$dst_dir/.backup"

  if [ -n "$wav_root" ]; then
    awk -v root="$wav_root" '{
      if ($2 !~ /^\//) $2 = root "/" $2;
      print
    }' "$dst_dir/wav.scp" >"$dst_dir/wav.scp.tmp"
    mv "$dst_dir/wav.scp.tmp" "$dst_dir/wav.scp"
  fi

  utils/fix_data_dir.sh "$dst_dir"
}

if [ $stage -le 0 ]; then
  for split in train valid severity_0 severity_1 severity_2 severity_3; do
    src_dir="${src_fold}/${split}"
    if [ -d "$src_dir" ]; then
      dst_dir="data/${out_prefix}_${split}"
      echo "$0: staging $src_dir -> $dst_dir"
      copy_and_clean "$src_dir" "$dst_dir"
    fi
  done
fi

echo "$0: done (fold=${fold}, out_prefix=${out_prefix})"
