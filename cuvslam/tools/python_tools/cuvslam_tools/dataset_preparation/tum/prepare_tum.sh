#!/usr/bin/env bash
# Download the TUM RGB-D (freiburg3) archive and lay out the dataset so the
# example tracking scripts can consume it.
#
# Usage: prepare_tum.sh [OPTIONS]
#
# Options:
#   --raw-dir DIR        Directory for raw archives.   Default: <repo>/datasets/tum/raw
#   --output-dir DIR     Directory for prepared data.  Default: <repo>/datasets/converted
#   --force-download     Re-download archives even when they already exist.
#   --download-only      Download archives but skip dataset layout.
#   -h, --help           Show this help.
#
# This is a provisioning wrapper only: it extracts the archive and copies the
# rig calibration file into place. It does not convert the dataset to the
# cuVSLAM reporter format.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "${script_dir}/../../../../.." && pwd -P)"

raw_dir="${repo_root}/datasets/tum/raw"
output_dir="${repo_root}/datasets/converted"
force_download=0
download_only=0

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
echo ""

download_args=("${raw_dir}")
[[ "${force_download}" -eq 1 ]] && download_args+=(--force)
bash "${script_dir}/download_tum.sh" "${download_args[@]}"

[[ "${download_only}" -eq 1 ]] && exit 0

echo ""
dataset_dir="${output_dir}/tum"
sequence_name="rgbd_dataset_freiburg3_long_office_household"
sequence_dir="${dataset_dir}/${sequence_name}"
archive="${raw_dir}/${sequence_name}.tgz"

echo "Extracting ${sequence_name}.tgz …"
mkdir -p "${dataset_dir}"
tar -xzf "${archive}" -C "${dataset_dir}"

if [[ ! -d "${sequence_dir}" ]]; then
    echo "error: expected ${sequence_dir} after extraction" >&2
    exit 1
fi

echo "Copying rig calibration …"
cp -f "${script_dir}/freiburg3_rig.yaml" "${sequence_dir}/freiburg3_rig.yaml"

echo ""
echo "done — dataset ready at ${sequence_dir}"
