#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split a Kaldi data directory into severity subgroups (0,1,2,3) for CLP.
Rule: if speaker-id contains "NH" -> severity 0; else use clp_id_to_label.
Creates kaldi-format sub-dirs under --out-root/severity_{k}.
"""

import argparse
import os
from collections import defaultdict

# --- mapping provided by you ---
clp_id_to_label = {
    # control
    "SV_HN": 0,
    # mild
    "AH_HN": 1, "CL_HN": 1, "CM_HN": 1, "CN_HN": 1, "DS_HN": 1, "EC1_HN": 1,
    "ES1_HN": 1, "IC_HN": 1, "JA2011_HN": 1, "MV_HN": 1, "UG_HN": 1,
    # moderate
    "CB_HN": 2, "EM_HN": 2, "ES_HN": 2, "GR_HN": 2, "IL_HN": 2, "JB_HN": 2,
    "JB1_HN": 2, "LS_HN": 2, "MO_HN": 2, "PC_HN": 2, "RD_HN": 2, "RM_HN": 2,
    "SW_HN": 2, "ZD_HN": 2,
    # severe
    "AC_HN": 3, "BQ_HN": 3, "CW_HN": 3, "DV_HN": 3, "EC_HN": 3, "GE_HN": 3,
    "GC_HN": 3, "HSB_HN": 3, "JA10_HN": 3, "JD_HN": 3, "JG8_HN": 3, "JL_HN": 3,
    "RT_HN": 3, "SW1_HN": 3, "TO_HN": 3, "JM_HN": 3,
}

# Known kaldi files (will only filter those that exist)
UTT_KEYED_FILES = [
    "utt2spk", "text", "wav.scp", "segments", "feats.scp",
    "cmvn.scp", "utt2dur", "utt2num_frames", "utt2uniq"
]
SPK_KEYED_FILES = [
    "spk2gender", "spk2age"
]
RECO_KEYED_FILES = ["reco2file_and_channel"]


def read_spk2utt(path):
    spk2utt = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            spk, utts = parts[0], parts[1:]
            spk2utt[spk] = utts
    return spk2utt


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def write_kaldi_map(path, mapping):
    # mapping: key -> [vals]
    with open(path, "w", encoding="utf-8") as f:
        for k in sorted(mapping.keys()):
            vals = mapping[k]
            if vals:
                f.write(f"{k} {' '.join(vals)}\n")


def write_utt2spk(path, utt2spk):
    with open(path, "w", encoding="utf-8") as f:
        for utt in sorted(utt2spk.keys()):
            f.write(f"{utt} {utt2spk[utt]}\n")


def filter_file_by_first_token(in_path, out_path, allowed_keys):
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            tok = line.split()[0]
            if tok in allowed_keys:
                fout.write(line)


def derive_recordings_for_subset(src_dir, keep_utts):
    seg = os.path.join(src_dir, "segments")
    recs = set()
    if os.path.exists(seg):
        with open(seg, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    utt, rec = parts[0], parts[1]
                    if utt in keep_utts:
                        recs.add(rec)
    else:
        # fallback: some setups key wav.scp by utt-id; use utts as rec-ids
        recs = set(keep_utts)
    return recs


def resolve_mapping_key(spk, mapping_keys):
    """Try exact match, else longest substring match."""
    if spk in mapping_keys:
        return spk
    # longest-first to avoid partial collisions
    for k in sorted(mapping_keys, key=len, reverse=True):
        if k in spk:
            return k
    return None


def decide_severity(spk, nh_token, mapping, on_missing):
    if nh_token and nh_token in spk:
        return 0
    key = resolve_mapping_key(spk, mapping.keys())
    if key is None:
        if on_missing == "zero":
            return 0
        elif on_missing == "skip":
            return None
        else:
            raise KeyError(f"Speaker '{spk}' not found in mapping and does not contain '{nh_token}'.")
    return mapping[key]


def main():
    ap = argparse.ArgumentParser(description="Split Kaldi data dir by severity (CLP).")
    ap.add_argument("--src", required=True, help="Source Kaldi data dir (must contain spk2utt).")
    ap.add_argument("--out-root", required=True, help="Output root dir; subdirs severity_{k} will be made.")
    ap.add_argument("--nh-token", default="NH", help="Substring that marks normal/control (default: NH).")
    ap.add_argument("--on-missing", choices=["error","skip","zero"], default="error",
                    help="If a non-NH speaker missing in mapping: error|skip|zero (default: error).")
    args = ap.parse_args()

    spk2utt_src_path = os.path.join(args.src, "spk2utt")
    if not os.path.exists(spk2utt_src_path):
        raise FileNotFoundError(f"Missing {spk2utt_src_path}")

    spk2utt_src = read_spk2utt(spk2utt_src_path)

    # Assign speakers to severities
    sev_to_spks = defaultdict(list)
    unknown_spks = []

    for spk in spk2utt_src.keys():
        try:
            sev = decide_severity(spk, args.nh_token, clp_id_to_label, args.on_missing)
        except KeyError:
            unknown_spks.append(spk)
            continue
        if sev is None:  # on-missing=skip
            continue
        if sev not in (0,1,2,3):
            raise ValueError(f"Invalid severity {sev} for speaker {spk}")
        sev_to_spks[sev].append(spk)

    if args.on_missing == "error" and unknown_spks:
        raise SystemExit(
            "ERROR: Some speakers not found in mapping and not marked as NH.\n"
            "Speakers:\n  " + "\n  ".join(sorted(unknown_spks)) + "\n"
            "Use --on-missing skip|zero to proceed, or extend clp_id_to_label."
        )

    ensure_dir(args.out_root)

    # Determine which known files are present to filter
    present_utt_files = [fn for fn in UTT_KEYED_FILES if os.path.exists(os.path.join(args.src, fn))]
    present_spk_files = [fn for fn in SPK_KEYED_FILES if os.path.exists(os.path.join(args.src, fn))]
    present_reco_files = [fn for fn in RECO_KEYED_FILES if os.path.exists(os.path.join(args.src, fn))]

    for sev, spk_list in sorted(sev_to_spks.items()):
        # Collect utts for this subgroup
        utt2spk = {}
        for spk in spk_list:
            for utt in spk2utt_src[spk]:
                utt2spk[utt] = spk
        if not utt2spk:
            continue

        out_dir = os.path.join(args.out_root, f"severity_{sev}")
        ensure_dir(out_dir)

        # Write utt2spk & spk2utt
        write_utt2spk(os.path.join(out_dir, "utt2spk"), utt2spk)
        spk2utt_subset = defaultdict(list)
        for utt, spk in utt2spk.items():
            spk2utt_subset[spk].append(utt)
        write_kaldi_map(os.path.join(out_dir, "spk2utt"), spk2utt_subset)

        # Filter utt-keyed files
        keep_utts = set(utt2spk.keys())
        for fn in present_utt_files:
            if fn == "utt2spk":
                continue  # already written
            srcp = os.path.join(args.src, fn)
            outp = os.path.join(out_dir, fn)
            filter_file_by_first_token(srcp, outp, keep_utts)

        # Filter spk-keyed files
        keep_spks = set(spk_list)
        for fn in present_spk_files:
            srcp = os.path.join(args.src, fn)
            outp = os.path.join(out_dir, fn)
            filter_file_by_first_token(srcp, outp, keep_spks)

        # Filter reco-keyed files (if any), using recordings from segments
        if present_reco_files:
            recs = derive_recordings_for_subset(args.src, keep_utts)
            for fn in present_reco_files:
                srcp = os.path.join(args.src, fn)
                if not os.path.exists(srcp):
                    continue
                outp = os.path.join(out_dir, fn)
                filter_file_by_first_token(srcp, outp, recs)

        # Quick summary file
        with open(os.path.join(out_dir, "SPLIT_INFO.txt"), "w", encoding="utf-8") as f:
            f.write(f"Severity: {sev}\nSpeakers: {len(spk_list)}\nUtterances: {len(utt2spk)}\n")

    print("Done. Created sub-dirs under:", args.out_root)
    for sev in sorted(sev_to_spks.keys()):
        print(f"  severity_{sev}: {len(sev_to_spks[sev])} speakers")


if __name__ == "__main__":
    main()