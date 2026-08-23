#!/usr/bin/env bash
set -euo pipefail

ext_src=${1:-ext_src}
fetch_tmp=/tmp/cuvslam_fetch_ext

echo "* Configuring cuVSLAM to fetch external sources"
rm -rf ${fetch_tmp}
cmake -S . -B ${fetch_tmp}/build \
    -DFETCHCONTENT_BASE_DIR=${fetch_tmp}/_deps \
    -DUSE_CUDA=OFF \
    -DUSE_NVTX=ON \
    -DUSE_RERUN=ON \
    -DCMAKE_BUILD_TYPE=Release

echo "* Recreating ${ext_src}"
rm -rf "${ext_src}" && mkdir -p "${ext_src}"

echo "* Copying fetched source trees"
for src in "${fetch_tmp}"/_deps/*-src; do
    cp -a "${src}" "${ext_src}/$(basename "${src}" -src)"
done

echo "* Fetched external sources to ${ext_src}"
