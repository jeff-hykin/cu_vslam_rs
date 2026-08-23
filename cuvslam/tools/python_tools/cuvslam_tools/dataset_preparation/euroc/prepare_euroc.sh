#!/usr/bin/env bash
# Download the EuRoC MAV machine_hall bundle, extract MH_01_easy, and convert it
# to edex format with the package-local EuRoC transform scripts.
#
# Usage: prepare_euroc.sh [OPTIONS]
#
# Options:
#   --raw-dir DIR        Directory for raw archives.   Default: <repo>/datasets/euroc/raw
#   --output-dir DIR     Directory for converted data. Default: <repo>/datasets/converted
#   --force-download     Re-download archives even when they already exist.
#   --download-only      Download archives but skip conversion.
#   -h, --help           Show this help.
#
# This wraps package-local conversion scripts:
#   euroc_to_edex.py     EuRoC sequence -> edex
#   euroc_gt_to_tum.py   ground-truth CSV -> TUM format

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "${script_dir}/../../../../.." && pwd -P)"

raw_dir="${repo_root}/datasets/euroc/raw"
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
bash "${script_dir}/download_euroc.sh" "${download_args[@]}"

[[ "${download_only}" -eq 1 ]] && exit 0

# euroc_to_edex.py needs numpy + pyyaml; install if missing so the existing
# transform can run.
if ! python3 -c "import numpy, yaml" 2>/dev/null; then
    echo "installing euroc_to_edex python deps (numpy, pyyaml) …"
    python3 -m pip install numpy pyyaml
fi

seq_name="MH_01_easy"
bundle_name="machine_hall.zip"
nested_zip="machine_hall/${seq_name}/${seq_name}.zip"   # path inside the bundle
seq_dir="${output_dir}/euroc/${seq_name}"
edex_dir="${seq_dir}/edex"
to_edex="${script_dir}/euroc_to_edex.py"
gt_to_tum="${script_dir}/euroc_gt_to_tum.py"

echo ""
echo "Extracting ${seq_name} from ${bundle_name} …"
mkdir -p "${seq_dir}"
# Stage under output-dir (same filesystem as the destination), not /tmp, which
# is often small/tmpfs and would fail extracting a multi-hundred-MB sequence.
stage_dir="$(mktemp -d -p "${output_dir}" euroc_stage.XXXXXX)"
trap 'rm -rf -- "${stage_dir}"' EXIT

# Pull just the nested per-sequence archive out of the bundle, then mav0 from it.
# unzip exits non-zero when the member is absent, so guard it (set -e) to emit a
# clear message instead of unzip's "caution: filename not matched".
if ! unzip -q -o "${raw_dir}/${bundle_name}" "${nested_zip}" -d "${stage_dir}" \
        || [[ ! -f "${stage_dir}/${nested_zip}" ]]; then
    echo "error: ${nested_zip} not found inside ${bundle_name}" >&2
    exit 1
fi
unzip -q -o "${stage_dir}/${nested_zip}" -d "${seq_dir}"

if [[ ! -d "${seq_dir}/mav0" ]]; then
    echo "error: expected ${seq_dir}/mav0 after extraction" >&2
    exit 1
fi

echo ""
echo "Converting to edex …"
python3 "${to_edex}" "${seq_dir}" --output "${edex_dir}"

# Verify the converter produced the core edex artifacts (catch a partial or
# silent no-op conversion before reporting success).
for artifact in stereo.edex frame_metadata.jsonl IMU.jsonl; do
    if [[ ! -s "${edex_dir}/${artifact}" ]]; then
        echo "error: euroc_to_edex did not produce ${edex_dir}/${artifact}" >&2
        exit 1
    fi
done
if [[ -z "$(ls -A "${edex_dir}/images" 2>/dev/null)" ]]; then
    echo "error: euroc_to_edex produced no images under ${edex_dir}/images" >&2
    exit 1
fi

echo ""
echo "Converting ground truth to TUM format …"
python3 "${gt_to_tum}" --dataset "${seq_dir}"

if [[ ! -s "${seq_dir}/gt_pose_tum.txt" ]]; then
    echo "error: euroc_gt_to_tum did not produce ${seq_dir}/gt_pose_tum.txt" >&2
    exit 1
fi

echo ""
echo "done — edex written to ${edex_dir}"
