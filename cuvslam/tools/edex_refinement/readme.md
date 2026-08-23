

Install Ceres

```

# google-glog + gflags
sudo apt-get install libgoogle-glog-dev libgflags-dev
# Use ATLAS for BLAS & LAPACK
sudo apt-get install libatlas-base-dev
# Eigen3
sudo apt-get install libeigen3-dev
# SuiteSparse (optional)
sudo apt-get install libsuitesparse-dev

sudo apt install libabsl-dev

git clone --recurse-submodules https://github.com/ceres-solver/ceres-solver
cd ceres-solver
#git checkout 6fb3dae4eeef855568e47ebbb29a8ba4f3c9153f
cmake -S . -B build -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF -DUSE_CUDA=ON -Dcudss_DIR=$CUDSS_DIR
cmake --build build --parallel 4
cmake --install build
#sed -i 's/find_dependency(cudss 0.3.0)/find_dependency(cudss)/' /usr/local/lib/cmake/Ceres/CeresConfig.cmake
```
