#!/usr/bin/env bash
# Download the TUM RGB-D (freiburg3 long_office_household) raw archive.
#
# Usage: download_tum.sh [OPTIONS] [OUT_DIR]
#
#   OUT_DIR        Directory to save archives.
#                  Defaults to <repo_root>/datasets/tum/raw
#   --force        Re-download archives even when they already exist.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "${script_dir}/../../../../.." && pwd -P)"
out_dir="${repo_root}/datasets/tum/raw"
force=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force) force=1; shift ;;
        -*) echo "error: unknown option '$1'" >&2; exit 2 ;;
        *)  out_dir="$1"; shift ;;
    esac
done

readonly -a urls=(
    "https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_long_office_household.tgz"
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

    # curl -f only catches HTTP >=400; a 200 with an HTML error/landing page
    # would slip through. The archives are gzip (.tgz), so verify the magic
    # bytes (1f 8b) before accepting the download.
    local magic
    magic="$(head -c 2 -- "${partial}" | od -An -tx1 | tr -d ' ')"
    if [[ "${magic}" != "1f8b" ]]; then
        echo "error: $(basename -- "${url}") is not a gzip archive (magic '${magic}'); the server likely returned an error page" >&2
        rm -f -- "${partial}"
        exit 1
    fi

    mv -f -- "${partial}" "${dest}"
}

for url in "${urls[@]}"; do
    download_file "${url}"
done

echo "done — files saved to ${out_dir}"
