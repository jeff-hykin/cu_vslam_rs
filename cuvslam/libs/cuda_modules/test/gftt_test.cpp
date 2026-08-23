
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

#include <chrono>
#include <iostream>
#include <random>

#include "common/environment.h"
#include "common/include_gtest.h"
#include "cuda_modules/gftt.h"
#include "cuda_modules/gradient_pyramid.h"
#include "cuda_modules/image_pyramid.h"
#include "sof/gftt.h"
#include "sof/gradient_pyramid.h"
#include "sof/image_pyramid_float.h"
#include "sof/sof_config.h"
#include "utils/image_loader.h"

namespace test {
using namespace cuvslam;
using namespace cuvslam::cuda;

bool load_random_image(ImageMatrixT& image_1) {
  std::string sequence_folder = cuvslam::Environment::GetVar(cuvslam::Environment::CUVSLAM_DATASETS);
  if (!cuvslam::IsPathEndWithSlash(sequence_folder)) {
    sequence_folder += "/";
  }
  std::string kitti_images_folder = sequence_folder + "kitti/00/00";
  size_t total_images = 4541;

  std::mt19937 gen(::testing::UnitTest::GetInstance()->random_seed());
  std::uniform_int_distribution<size_t> distrib(0, total_images - 2);

  size_t image_id = distrib(gen);

  std::string image_1_name = std::to_string(image_id);

  while (image_1_name.size() < 4) {
    image_1_name = "0" + image_1_name;
  }
  cuvslam::utils::ImageLoaderT img_1_loader;
  if (!img_1_loader.load(kitti_images_folder + "/00.0." + image_1_name + ".png")) {
    return false;
  }

  image_1 = img_1_loader.getImage().cast<float>();
  return true;
}

TEST(Cuda, DISABLED_GFTT) {
  ImageMatrixT img1;
  ASSERT_TRUE(load_random_image(img1));

  GPUImageT device_img1(img1);

  Stream s;

  GaussianGPUImagePyramid cuda_pyramid_1(img1.cols(), img1.rows());
  cuda_pyramid_1.build(device_img1, false, s.get_stream());

  GPUGradientPyramid cuda_grad_1(img1.cols(), img1.rows());
  cuda_grad_1.set(cuda_pyramid_1, s.get_stream());

  sof::ImagePyramidT img1_p(img1);
  img1_p.buildPyramid();
  sof::Settings sof_settings;
  sof::GradientPyramidT img1_g;
  img1_g.set(img1_p);
  img1_g.forceNonLazyEvaluation(0);

  cuvslam::sof::GFTT gftt;
  cuvslam::cuda::GFTT cuda_gftt;

  auto& cuda_gradX = cuda_grad_1.gradX().base();
  auto& cuda_gradY = cuda_grad_1.gradY().base();
  GPUImageT cuda_value(cuda_gradX.cols(), cuda_gradX.rows());
  ImageMatrixT cpu_value(cuda_gradX.rows(), cuda_gradX.cols());

  ImageMatrixT& gradX = img1_g.gradX()[0];
  ImageMatrixT& gradY = img1_g.gradY()[0];

  gftt.compute(gradX, gradY);
  const ImageMatrixT& value = gftt.get();

  cuda_gftt.compute(cuda_gradX, cuda_gradY, cuda_value, s.get_stream());
  cuda_value.copy(GPUCopyDirection::ToCPU, cpu_value.data(), s.get_stream());

  cudaStreamSynchronize(s.get_stream());

  if (!value.isApprox(cpu_value, 0.01f)) {
    std::cout << "basic = " << std::endl << value.block<10, 10>(0, 0) << std::endl;
    std::cout << "cuda = " << std::endl << cpu_value.block<10, 10>(0, 0) << std::endl;
  }

  ASSERT_TRUE(value.isApprox(cpu_value, 0.01f));
}

TEST(Cuda, DISABLED_GFTTSpeedUp) {
  ImageMatrixT img1;
  ASSERT_TRUE(load_random_image(img1));

  GPUImageT device_img1(img1);

  GaussianGPUImagePyramid cuda_pyramid_1(img1.cols(), img1.rows());
  Stream s;
  cuda_pyramid_1.build(device_img1, false, s.get_stream());

  GPUGradientPyramid cuda_grad_1(img1.cols(), img1.rows());
  cuda_grad_1.set(cuda_pyramid_1, s.get_stream());
  cudaStreamSynchronize(s.get_stream());

  sof::ImagePyramidT img1_p(img1);
  img1_p.buildPyramid();
  sof::Settings sof_settings;
  sof::GradientPyramidT img1_g;
  img1_g.set(img1_p);
  img1_g.forceNonLazyEvaluation(0);
  cuvslam::sof::GFTT gftt;
  cuvslam::cuda::GFTT cuda_gftt;

  auto& cuda_gradX = cuda_grad_1.gradX().base();
  auto& cuda_gradY = cuda_grad_1.gradY().base();
  GPUImageT cuda_value(cuda_gradX.cols(), cuda_gradX.rows());

  ImageMatrixT& gradX = img1_g.gradX()[0];
  ImageMatrixT& gradY = img1_g.gradY()[0];

  auto time_basic_start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 100; i++) {
    gftt.compute(gradX, gradY);
  }
  auto duration_basic = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() -
                                                                             time_basic_start);

  auto time_cuda_start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 100; i++) {
    cuda_gftt.compute(cuda_gradX, cuda_gradY, cuda_value, s.get_stream());
  }
  cudaStreamSynchronize(s.get_stream());
  auto duration_cuda =
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - time_cuda_start);
  std::cout << "Basic time, nano_sec = " << duration_basic.count() / 100 << std::endl;
  std::cout << "Cuda time, nano_sec = " << duration_cuda.count() / 100 << std::endl;
  float speedup = static_cast<float>(duration_basic.count()) / static_cast<float>(duration_cuda.count());
  std::cout << "Speedup, times = " << speedup << std::endl;
  ASSERT_TRUE(duration_basic >= duration_cuda);
}

}  // namespace test
