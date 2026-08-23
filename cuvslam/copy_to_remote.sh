#!/bin/bash
set -e

REMOTE=${1:?Usage: $0 <remote-host>}

echo "Copying to ${REMOTE}:cuvslam/src"
rsync -av --delete --filter=':- .gitignore' --exclude=".*" . "${REMOTE}:cuvslam/src"
