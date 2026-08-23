
/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * NVIDIA software released under the NVIDIA Community License is intended to be used to enable
 * the further development of AI and robotics technologies. Such software has been designed, tested,
 * and optimized for use with NVIDIA hardware, and this License grants permission to use the software
 * solely with such hardware.
 * Subject to the terms of this License, NVIDIA confirms that you are free to commercially use,
 * modify, and distribute the software with NVIDIA hardware. NVIDIA does not claim ownership of any
 * outputs generated using the software or derivative works thereof. Any code contributions that you
 * share with NVIDIA are licensed to NVIDIA as feedback under this License and may be incorporated
 * in future releases without notice or attribution.
 * By using, reproducing, modifying, distributing, performing, or displaying any portion or element
 * of the software or derivative works thereof, you agree to be bound by this License.
 */

#pragma once

#include <cuda_runtime_api.h>

#include "common/error.h"

namespace cuvslam::cuda {

#if CUDART_VERSION < 12000
// Backport cudaGraphExecUpdateResultInfo and new versions of cudaGraphExecUpdate, cudaGraphInstantiate for CUDA 11
struct cudaGraphExecUpdateResultInfo {
  cudaGraphExecUpdateResult result;
  cudaGraphNode_t errorNode;
};
inline cudaError_t cudaGraphExecUpdate(cudaGraphExec_t hGraphExec, cudaGraph_t graph,
                                       cudaGraphExecUpdateResultInfo* resultInfo) {
  return ::cudaGraphExecUpdate(hGraphExec, graph, &resultInfo->errorNode, &resultInfo->result);
}
inline cudaError_t cudaGraphInstantiate(cudaGraphExec_t* pGraphExec, cudaGraph_t graph,
                                        unsigned long long /*flags*/ = 0) {
  return ::cudaGraphInstantiate(pGraphExec, graph, nullptr, nullptr, 0);
}
#else
// Use the built-in versions from cuda_runtime_api.h
#endif

}  // namespace cuvslam::cuda
