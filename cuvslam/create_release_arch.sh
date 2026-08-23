#!/bin/bash
set -e # stop script on any error
set -v # echo each command

# update remote path to your own
cd ..
mkdir -p install/cuvslam/
rsync src/libs/public/cuvslam.h install/cuvslam/include/
rsync build/release/bin/libcuvslam.so install/cuvslam/lib_x86_64/
rsync build/remote/bin/libcuvslam.so install/cuvslam/lib_aarch64_jetpack50/
cd install
tar -cJf cuvslam.tar.xz cuvslam
