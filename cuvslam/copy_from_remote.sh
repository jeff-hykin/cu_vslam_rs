#!/bin/bash
set -e

REMOTE=${1:?Usage: $0 <remote-host>}
LOCAL_DIR=../build/remote

echo "Copying from ${REMOTE}:cuvslam/build/bin to ${LOCAL_DIR}"
mkdir -p "${LOCAL_DIR}"
rsync -av "${REMOTE}:cuvslam/build/bin" "${LOCAL_DIR}/"
