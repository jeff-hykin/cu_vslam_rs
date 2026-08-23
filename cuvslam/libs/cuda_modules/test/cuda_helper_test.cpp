
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

#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>

#include "common/include_gtest.h"
#include "cuda_modules/cuda_helper.h"

namespace test {
namespace {

struct FourByteElement {
  uint32_t value;
};

template <typename T>
size_t OverflowingElementCount() {
  return std::numeric_limits<size_t>::max() / sizeof(T) + 1;
}

}  // namespace

TEST(Cuda, CheckedMulRejectsOverflow) {
  ASSERT_EQ(cuvslam::cuda::CheckedMul(7, 6), 42);
  ASSERT_EQ(cuvslam::cuda::CheckedMul(std::numeric_limits<size_t>::max(), 0), 0);
  ASSERT_THROW(cuvslam::cuda::CheckedMul(std::numeric_limits<size_t>::max(), 2), std::bad_array_new_length);
}

TEST(Cuda, CheckedBytesRejectsOverflow) {
  ASSERT_EQ(cuvslam::cuda::CheckedBytes<FourByteElement>(3), sizeof(FourByteElement) * 3);
  ASSERT_THROW(cuvslam::cuda::CheckedBytes<FourByteElement>(OverflowingElementCount<FourByteElement>()),
               std::bad_array_new_length);
}

TEST(Cuda, GpuArraysRejectOverflowingElementCountsBeforeAllocation) {
  // These counts would wrap when converted to bytes. The expected behavior is
  // to fail locally with a C++ allocation error, before any CUDA allocator sees
  // a misleadingly small byte count.
  const size_t overflowing_count = OverflowingElementCount<FourByteElement>();

  ASSERT_THROW(
      {
        cuvslam::cuda::GPUArray<FourByteElement> array(overflowing_count);
        (void)array;
      },
      std::bad_array_new_length);
  ASSERT_THROW(
      {
        cuvslam::cuda::GPUArrayPinned<FourByteElement> array(overflowing_count);
        (void)array;
      },
      std::bad_array_new_length);
  ASSERT_THROW(
      {
        cuvslam::cuda::GPUOnlyArray<FourByteElement> array(overflowing_count);
        (void)array;
      },
      std::bad_array_new_length);
}

TEST(Cuda, GpuImageRejectsOverflowingPitchBeforeAllocation) {
  // GPUImage passes row bytes to cudaMallocPitch. This regression case proves a
  // huge column count is rejected before the pitch can wrap.
  ASSERT_THROW(
      {
        cuvslam::cuda::GPUImage<FourByteElement> image(OverflowingElementCount<FourByteElement>(), 1);
        (void)image;
      },
      std::bad_array_new_length);
}

TEST(Cuda, CopyTopNRejectsOutOfRangeCounts) {
  // copy_top_n used to rely on assert(top_n <= size_), which disappears in
  // release builds. These checks keep the copy boundary active in production.
  cuvslam::cuda::GPUArray<FourByteElement> array(1);
  ASSERT_THROW(array.copy_top_n(cuvslam::cuda::ToGPU, 2, nullptr), std::out_of_range);

  cuvslam::cuda::GPUArrayPinned<FourByteElement> pinned_array(1);
  ASSERT_THROW(pinned_array.copy_top_n(cuvslam::cuda::ToGPU, 2, nullptr), std::out_of_range);
}

TEST(Cuda, CheckCompatibility) {
  std::string message;
  bool ok = cuvslam::cuda::CheckCompatibility(message);
  ASSERT_TRUE(ok) << message;
}

TEST(Cuda, WarmUpGpu) { ASSERT_NO_THROW(cuvslam::cuda::WarmUpGpu()); }

}  // namespace test
