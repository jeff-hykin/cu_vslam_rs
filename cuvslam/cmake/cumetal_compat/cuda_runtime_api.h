#pragma once

// CuMetal's own cuda_runtime_api.h is a one-line forwarder to cuda_runtime.h, so shadowing it here
// costs nothing and lets us close one API-fidelity gap: CuMetal declares the resource-type
// constants as an anonymous enum nested inside cudaResourceDesc, whereas the CUDA Toolkit declares
// enum cudaResourceType at namespace scope. cuVSLAM (like most CUDA code) spells them unqualified.

#include <cuda_runtime.h>

typedef decltype(cudaResourceDesc::cudaResourceTypeArray) cudaResourceType;

constexpr cudaResourceType cudaResourceTypeArray = cudaResourceDesc::cudaResourceTypeArray;
constexpr cudaResourceType cudaResourceTypeMipmappedArray = cudaResourceDesc::cudaResourceTypeMipmappedArray;
constexpr cudaResourceType cudaResourceTypeLinear = cudaResourceDesc::cudaResourceTypeLinear;
constexpr cudaResourceType cudaResourceTypePitch2D = cudaResourceDesc::cudaResourceTypePitch2D;
