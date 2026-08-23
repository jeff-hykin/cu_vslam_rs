#!/usr/bin/env bash
# Download KITTI odometry raw archives.
#
# Usage: download_kitti.sh [OPTIONS] [OUT_DIR]
#
#   OUT_DIR        Directory to save archives.
#                  Defaults to <repo_root>/datasets/kitti/raw
#   --force        Re-download archives even when they already exist.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "${script_dir}/../../../../.." && pwd -P)"
out_dir="${repo_root}/datasets/kitti/raw"
force=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force) force=1; shift ;;
        -*) echo "error: unknown option '$1'" >&2; exit 2 ;;
        *)  out_dir="$1"; shift ;;
    esac
done

readonly -a urls=(
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_gray.zip"
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_calib.zip"
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_poses.zip"
)

download_file() {
    local url="$1"
    local dest="${out_dir}/$(basename -- "${url}")"
    local partial="${dest}.download"

    if [[ -s "${dest}" && "${force}" -eq 0 ]]; then
        echo "using existing ${dest}"
        return
    fi

    mkdir -p "${out_dir}"
    [[ "${force}" -eq 1 ]] && rm -f -- "${dest}" "${partial}"

    echo "downloading $(basename -- "${url}") …"
    curl -fL --retry 5 --retry-delay 5 -C - -o "${partial}" "${url}"

    mv -f -- "${partial}" "${dest}"
}

for url in "${urls[@]}"; do
    download_file "${url}"
done

echo "done — files saved to ${out_dir}"
