#!/usr/bin/env bash
# Download the TartanGround dataset and convert it to edex format by wrapping
# the existing provisioning + transform scripts.
#
# Usage: prepare_tartan.sh [OPTIONS]
#
# Options:
#   --raw-dir DIR        Directory to download into. Default: <repo>/datasets/tartan/raw
#   --output-dir DIR     Directory for converted data. Default: <repo>/datasets/converted
#   --variant NAME       'multisensor' or 'multicamera'. Default: multisensor.
#                        Use 'multicamera' for EDEX conversion.
#   --force-download     Remove any existing download/conversion output first.
#   --download-only      Download but skip conversion.
#   -h, --help           Show this help.
#
# This wraps existing workflows and does not reimplement conversion:
#   dataset_preparation/tartan/download.py  download via the tartanair package
#   dataset_converter  sequences -> in-place edex + gt
#
# Note: dataset_converter handles the classic TartanAir layout (image_left/
# image_right + pose_left.txt/pose_right.txt) and rewrites each sequence in
# place (image_left->00, image_right->01, gt.txt, cfg.edex). This script stages
# classic layouts directly, or maps TartanGround lcam_*/rcam_* stereo pairs plus
# metadata pose files into that classic layout first. Conversion runs on staged
# files under --output-dir so the raw download is preserved.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "${script_dir}/../../../../.." && pwd -P)"

raw_dir="${repo_root}/datasets/tartan/raw"
output_dir="${repo_root}/datasets/converted"
variant="multisensor"
force_download=0
download_only=0
python_bin="${PYTHON_BIN:-python3}"

usage() {
    sed -n '2,/^$/p' "$0" | grep '^#' | sed 's/^# \?//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --raw-dir)
            [[ $# -lt 2 ]] && { echo "error: --raw-dir requires a value" >&2; exit 2; }
            raw_dir="$2"; shift 2 ;;
        --output-dir)
            [[ $# -lt 2 ]] && { echo "error: --output-dir requires a value" >&2; exit 2; }
            output_dir="$2"; shift 2 ;;
        --variant)
            [[ $# -lt 2 ]] && { echo "error: --variant requires a value" >&2; exit 2; }
            variant="$2"; shift 2 ;;
        --force-download)
            force_download=1; shift ;;
        --download-only)
            download_only=1; shift ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "error: unknown option '$1'" >&2; exit 2 ;;
    esac
done

echo "Raw dir    : ${raw_dir}"
echo "Output dir : ${output_dir}"
echo "Variant    : ${variant}"
echo ""

seq_path="${raw_dir}/dataset/tartan_ground"
converted_dir="${output_dir}/tartan/${variant}"

if [[ "${download_only}" -eq 0 && "${force_download}" -eq 0 && -e "${converted_dir}" ]]; then
    echo "error: ${converted_dir} already exists; use --force-download to overwrite" >&2
    exit 1
fi

download_args=("${raw_dir}" --variant "${variant}")
[[ "${force_download}" -eq 1 ]] && download_args+=(--force)
bash "${script_dir}/download_tartan.sh" "${download_args[@]}"

[[ "${download_only}" -eq 1 ]] && exit 0

# dataset_converter rewrites each sequence in place, so work on staged files
# under --output-dir and keep them (they hold the converted result).
if [[ "${force_download}" -eq 1 ]]; then
    rm -rf -- "${converted_dir}"
fi

echo ""
echo "Staging converter-compatible sequences …"
PYTHONPATH="${script_dir}${PYTHONPATH:+:${PYTHONPATH}}" "${python_bin}" -m stage_sequences \
    --seq-path "${seq_path}" \
    --output-dir "${converted_dir}"

echo "Converting to edex …"
# --save_gt_folder/--save_edex_folder are required but unused for output (the
# converter writes in place); point them at fresh paths it can create.
PYTHONPATH="${script_dir}${PYTHONPATH:+:${PYTHONPATH}}" "${python_bin}" -m dataset_converter \
    --seq_path "${converted_dir}" \
    --save_gt_folder "${converted_dir}/.gt_unused" \
    --save_edex_folder "${converted_dir}/.edex_unused"

if ! find "${converted_dir}" -name cfg.edex -type f | grep -q .; then
    echo "error: conversion produced no cfg.edex under ${converted_dir}" >&2
    exit 1
fi

echo ""
echo "done — converted sequences (00/01 images, gt.txt, cfg.edex) under ${converted_dir}"
