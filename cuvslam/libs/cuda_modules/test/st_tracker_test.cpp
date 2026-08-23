
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
#include "cuda_modules/cuda_kernels/cuda_kernels.h"
#include "cuda_modules/gradient_pyramid.h"
#include "cuda_modules/image_pyramid.h"
#include "cuda_modules/st_tracker.h"
#include "sof/gradient_pyramid.h"
#include "sof/image_pyramid_float.h"
#include "sof/selection.h"
#include "sof/sof.h"
#include "sof/sof_config.h"
#include "sof/st_tracker.h"
#include "utils/image_loader.h"

namespace test {
using namespace cuvslam;

#define MAX_TRACKS 1000

bool load_random_images_st(ImageMatrixT& image_1, ImageMatrixT& image_2) {
  std::string sequence_folder = Environment::GetVar(Environment::CUVSLAM_DATASETS);
  if (!IsPathEndWithSlash(sequence_folder)) {
    sequence_folder += "/";
  }
  std::string kitti_images_folder = sequence_folder + "kitti/00/00";
  size_t total_images = 4541;

  std::mt19937 gen(::testing::UnitTest::GetInstance()->random_seed());
  std::uniform_int_distribution<size_t> distrib(0, total_images - 2);

  size_t image_id = distrib(gen);

  std::string image_1_name = std::to_string(image_id);
  std::string image_2_name = std::to_string(image_id + 1);

  while (image_1_name.size() < 4) {
    image_1_name = "0" + image_1_name;
  }
  while (image_2_name.size() < 4) {
    image_2_name = "0" + image_2_name;
  }
  utils::ImageLoaderT img_1_loader;
  utils::ImageLoaderT img_2_loader;
  if (!img_1_loader.load(kitti_images_folder + "/00.0." + image_1_name + ".png")) {
    return false;
  }
  if (!img_2_loader.load(kitti_images_folder + "/00.0." + image_2_name + ".png")) {
    return false;
  }

  image_1 = img_1_loader.getImage().cast<float>();
  image_2 = img_2_loader.getImage().cast<float>();
  return true;
}

TEST(Cuda, DISABLED_STTracker) {
  ImageMatrixT img1, img2;
  ASSERT_TRUE(load_random_images_st(img1, img2));

  cuda::GPUImageT device_img1(img1);
  cuda::GPUImageT device_img2(img2);

  size_t img_width = img1.cols();
  size_t img_height = img1.rows();

  cuda::GaussianGPUImagePyramid cuda_pyramid_1(img_width, img_height);
  cuda::GaussianGPUImagePyramid cuda_pyramid_2(img_width, img_height);

  cuda::Stream s;
  cuda_pyramid_1.build(device_img1, false, s.get_stream());
  cuda_pyramid_2.build(device_img2, false, s.get_stream());
  cudaStreamSynchronize(s.get_stream());

  cuda::GPUGradientPyramid cuda_grad_1(img_width, img_height);
  cuda::GPUGradientPyramid cuda_grad_2(img_width, img_height);

  cuda_grad_1.set(cuda_pyramid_1, s.get_stream());
  cuda_grad_2.set(cuda_pyramid_2, s.get_stream());
  cudaStreamSynchronize(s.get_stream());

  sof::ImagePyramidT img1_p(img1);
  sof::ImagePyramidT img2_p(img2);
  img1_p.buildPyramid();
  img2_p.buildPyramid();

  sof::Settings sof_settings;

  sof::GradientPyramidT img1_g;
  sof::GradientPyramidT img2_g;
  img1_g.set(img1_p);
  img2_g.set(img2_p);

  sof::TracksVector tracks;
  std::vector<Vector2T> new_tracks;

  sof::GoodFeaturesToTrackDetector detector;
  detector.computeGFTTAndSelectFeatures(img1_g, 0, 0, 0, 0, nullptr, tracks, MAX_TRACKS, new_tracks);

  sof::STTracker tracker(0, 20);
  cuda::GPUSTFeatureTracker cuda_tracker(0, 20);

  std::vector<Vector2T> uvs;
  std::vector<Matrix2T> infos;

  for (auto& xy : new_tracks) {
    Vector2T xy_r = xy;
    Matrix2T info;
    if (tracker.trackPoint(img1_g, img2_g, img1_p, img2_p, xy, xy_r, info, 2048.f, 0.8f)) {
      uvs.push_back(std::move(xy_r));
      infos.push_back(std::move(info));
    }
  }

  int num_tracks = new_tracks.size();
  cuda::GPUArrayPinned<cuda::TrackData> tracks_data(MAX_TRACKS);
  for (int i = 0; i < num_tracks; i++) {
    tracks_data[i].track = {new_tracks[i].x(), new_tracks[i].y()};
    tracks_data[i].offset = {0, 0};
    tracks_data[i].track_status = false;
    tracks_data[i].search_radius_px = 2048.f;
    tracks_data[i].ncc_threshold = 0.8f;
  }

  tracks_data.copy(cuda::GPUCopyDirection::ToGPU, s.get_stream());
  cuda_tracker.track_points(cuda_grad_1, cuda_grad_2, cuda_pyramid_1, cuda_pyramid_2, tracks_data, num_tracks,
                            s.get_stream());
  tracks_data.copy(cuda::GPUCopyDirection::ToCPU, s.get_stream());
  cudaStreamSynchronize(s.get_stream());

  std::vector<Vector2T> uvs_cuda;
  std::vector<Matrix2T> infos_cuda;
  for (int i = 0; i < num_tracks; i++) {
    if (tracks_data[i].track_status) {
      uvs_cuda.push_back({tracks_data[i].track.x, tracks_data[i].track.y});

      Matrix2T info_mat;
      info_mat << tracks_data[i].info[0], tracks_data[i].info[1], tracks_data[i].info[2], tracks_data[i].info[3];
      infos_cuda.push_back(std::move(info_mat));
    }
  }

  if (uvs_cuda.size() != uvs.size()) {
    std::cout << "uvs.size() = " << uvs.size() << std::endl;
    std::cout << "uvs_cuda.size() = " << uvs_cuda.size() << std::endl;
  }
  ASSERT_TRUE(uvs_cuda.size() == uvs.size());
}

TEST(Cuda, DISABLED_STTrackerSpeedupFullmap) {
  ImageMatrixT img1, img2;
  ASSERT_TRUE(load_random_images_st(img1, img2));

  cuda::GPUImageT device_img1(img1);
  cuda::GPUImageT device_img2(img2);

  size_t img_width = img1.cols();
  size_t img_height = img1.rows();

  cuda::GaussianGPUImagePyramid cuda_pyramid_1(img_width, img_height);
  cuda::GaussianGPUImagePyramid cuda_pyramid_2(img_width, img_height);

  cuda::Stream s;
  cuda_pyramid_1.build(device_img1, false, s.get_stream());
  cuda_pyramid_2.build(device_img2, false, s.get_stream());
  cudaStreamSynchronize(s.get_stream());

  cuda::GPUGradientPyramid cuda_grad_1(img_width, img_height);
  cuda::GPUGradientPyramid cuda_grad_2(img_width, img_height);

  cuda_grad_1.set(cuda_pyramid_1, s.get_stream());
  cuda_grad_2.set(cuda_pyramid_2, s.get_stream());
  cudaStreamSynchronize(s.get_stream());

  sof::ImagePyramidT img1_p(img1);
  sof::ImagePyramidT img2_p(img2);
  img1_p.buildPyramid();
  img2_p.buildPyramid();

  sof::Settings sof_settings;

  sof::GradientPyramidT img1_g;
  sof::GradientPyramidT img2_g;
  img1_g.set(img1_p);
  img2_g.set(img2_p);

  sof::TracksVector tracks;
  std::vector<Vector2T> new_tracks;

  sof::GoodFeaturesToTrackDetector detector;
  detector.computeGFTTAndSelectFeatures(img1_g, 0, 0, 0, 0, nullptr, tracks, MAX_TRACKS, new_tracks);

  sof::STTracker tracker(0, 20);
  cuda::GPUSTFeatureTracker cuda_tracker(0, 20);

  std::vector<Vector2T> uvs;
  std::vector<Matrix2T> infos;

  Vector2T xy_r;
  Matrix2T info;

  auto time_basic_start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 100; i++) {
    for (auto& xy : new_tracks) {
      tracker.trackPoint(img1_g, img2_g, img1_p, img2_p, xy, xy_r, info, 2048.f, 0.8f);
    }
  }
  auto duration_basic = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() -
                                                                             time_basic_start);

  int num_tracks = new_tracks.size();
  cuda::GPUArrayPinned<cuda::TrackData> tracks_data(num_tracks);
  for (int i = 0; i < num_tracks; i++) {
    tracks_data[i].track = {new_tracks[i].x(), new_tracks[i].y()};
    tracks_data[i].offset = {0, 0};
    tracks_data[i].track_status = false;
    tracks_data[i].ncc_threshold = 0.8f;
  }

  tracks_data.copy(cuda::GPUCopyDirection::ToGPU, s.get_stream());
  cudaStreamSynchronize(s.get_stream());
  auto time_cuda_start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 100; i++) {
    cuda_tracker.track_points(cuda_grad_1, cuda_grad_2, cuda_pyramid_1, cuda_pyramid_2, tracks_data, num_tracks,
                              s.get_stream());
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

TEST(Cuda, DISABLED_STTrackerSpeedupPartialmap) {
  ImageMatrixT img1, img2;
  ASSERT_TRUE(load_random_images_st(img1, img2));

  cuda::GPUImageT device_img1(img1);
  cuda::GPUImageT device_img2(img2);

  size_t img_width = img1.cols();
  size_t img_height = img1.rows();

  cuda::GaussianGPUImagePyramid cuda_pyramid_1(img_width, img_height);
  cuda::GaussianGPUImagePyramid cuda_pyramid_2(img_width, img_height);

  cuda::Stream s;
  cuda_pyramid_1.build(device_img1, false, s.get_stream());
  cuda_pyramid_2.build(device_img2, false, s.get_stream());
  cudaStreamSynchronize(s.get_stream());

  cuda::GPUGradientPyramid cuda_grad_1(img_width, img_height);
  cuda::GPUGradientPyramid cuda_grad_2(img_width, img_height);

  cuda_grad_1.set(cuda_pyramid_1, s.get_stream());
  cuda_grad_2.set(cuda_pyramid_2, s.get_stream());
  cudaStreamSynchronize(s.get_stream());

  sof::ImagePyramidT img1_p(img1);
  sof::ImagePyramidT img2_p(img2);
  img1_p.buildPyramid();
  img2_p.buildPyramid();

  sof::Settings sof_settings;

  sof::GradientPyramidT img1_g;
  sof::GradientPyramidT img2_g;
  img1_g.set(img1_p);
  img2_g.set(img2_p);

  sof::TracksVector tracks;
  std::vector<Vector2T> new_tracks;

  sof::GoodFeaturesToTrackDetector detector;
  detector.computeGFTTAndSelectFeatures(img1_g, 0, 0, 0, 0, nullptr, tracks, MAX_TRACKS, new_tracks);

  sof::STTracker tracker(20, 0);
  cuda::GPUSTFeatureTracker cuda_tracker(20, 0);

  std::vector<Vector2T> uvs;
  std::vector<Matrix2T> infos;

  Vector2T xy_r;
  Matrix2T info;

  auto time_basic_start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 100; i++) {
    for (auto& xy : new_tracks) {
      tracker.trackPoint(img1_g, img2_g, img1_p, img2_p, xy, xy_r, info, 2048.f, 0.8f);
    }
  }
  auto duration_basic = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() -
                                                                             time_basic_start);

  int num_tracks = new_tracks.size();
  cuda::GPUArrayPinned<cuda::TrackData> tracks_data(num_tracks);
  for (int i = 0; i < num_tracks; i++) {
    tracks_data[i].track = {new_tracks[i].x(), new_tracks[i].y()};
    tracks_data[i].offset = {0, 0};
    tracks_data[i].track_status = false;
    tracks_data[i].ncc_threshold = 0.8f;
  }

  tracks_data.copy(cuda::GPUCopyDirection::ToGPU, s.get_stream());
  cudaStreamSynchronize(s.get_stream());
  auto time_cuda_start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 100; i++) {
    cuda_tracker.track_points(cuda_grad_1, cuda_grad_2, cuda_pyramid_1, cuda_pyramid_2, tracks_data, num_tracks,
                              s.get_stream());
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
