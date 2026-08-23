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

#include "sof/st_tracker.h"

#include "common/include_gtest.h"
#include "sof/image_pyramid_float.h"
#include "sof/sof_config.h"
#include "utils/image_loader.h"

#include <random>

namespace test::sof {
using namespace cuvslam;
using namespace cuvslam::sof;

bool load_image(ImageMatrixT& image, const std::string& image_name) {
  const std::string test_folder = CUVSLAM_TEST_ASSETS;
  utils::ImageLoaderT loader;
  if (!loader.load(test_folder + image_name)) {
    return false;
  }
  image = loader.getImage().cast<float>();
  return true;
}

// Rotates input image 90 degrees clockwise. Output size is (input.cols() x input.rows()).
bool rotate90(ImageMatrixT& rotated_image, const ImageMatrixT& input) {
  const size_t in_rows = input.rows();
  const size_t in_cols = input.cols();
  rotated_image = ImageMatrixT::Zero(in_cols, in_rows);
  for (size_t i = 0; i < in_rows; i++) {
    for (size_t j = 0; j < in_cols; j++) {
      rotated_image(j, in_rows - 1 - i) = input(i, j);
    }
  }
  return true;
}

TEST(STTracker, TrackRotatedImageShiftOnly) {
  STTracker tracker(10, 0);

  ImageMatrixT previous_image;
  load_image(previous_image, "sof/left.png");
  ImageMatrixT current_image;
  rotate90(current_image, previous_image);

  ImagePyramidT previous_pyramid(previous_image);
  previous_pyramid.buildPyramid();
  ImagePyramidT current_pyramid(current_image);
  current_pyramid.buildPyramid();
  Settings sof_settings;
  GradientPyramidT current_gradient;
  current_gradient.set(current_pyramid);
  current_gradient.forceNonLazyEvaluation(0);

  // Features in original image (row, col). After 90° CW rotation,
  // rotated image size is (original.cols() x original.rows()).
  const std::vector<Vector2T> newSelectedFeatures = {{130, 81}, {152, 201}, {255, 161}, {372, 157}};
  const size_t rotated_width = previous_image.rows();
  const std::vector<Vector2T> default_current_features = {{rotated_width - 1 - 81, 130},
                                                          {rotated_width - 1 - 201, 152},
                                                          {rotated_width - 1 - 161, 255},
                                                          {rotated_width - 1 - 157, 372}};

  std::mt19937 gen(::testing::UnitTest::GetInstance()->random_seed());
  std::uniform_real_distribution<> uniform_dist{-5, 5};

  Matrix2T guess_affine_map;

  float angle_in_radians = -90 * PI / 180;
  float cos_x = cos(angle_in_radians);
  float sin_x = sin(angle_in_radians);
  guess_affine_map << cos_x, sin_x, -sin_x, cos_x;

  for (int test_case = 0; test_case < 20; test_case++) {
    std::vector<Vector2T> current_features;
    current_features.reserve(default_current_features.size());
    for (const Vector2T& x : default_current_features) {
      current_features.emplace_back(x + Vector2T(uniform_dist(gen), uniform_dist(gen)));
    }

    size_t num_well_tracked_features = 0;
    for (size_t i = 0; i < newSelectedFeatures.size(); i++) {
      const Vector2T& prev_feature = newSelectedFeatures[i];
      const Vector2T& curr_feature = current_features[i];
      std::vector<STFeaturePatch> image_patches;
      uint32_t levels_mask;

      if (!STTracker::BuildPointCache(previous_pyramid, prev_feature, levels_mask, image_patches)) {
        std::cout << "Failed to build cache" << std::endl;
        continue;
      }

      float ncc = 0;
      float current_info[4];

      float curr_uv[] = {curr_feature[0], curr_feature[1]};

      bool res = tracker.TrackPointWithCache(current_gradient, current_pyramid, levels_mask, image_patches.size(),
                                             &image_patches[0], curr_uv, curr_uv, ncc, current_info, 300, 0.8,
                                             guess_affine_map);

      if (res) {
        num_well_tracked_features++;
      }
    }

    ASSERT_TRUE(num_well_tracked_features == newSelectedFeatures.size());
  }
}

TEST(STTracker, TrackRotatedImageFullMap) {
  STTracker tracker(5, 10);

  ImageMatrixT previous_image;
  load_image(previous_image, "sof/left.png");
  ImageMatrixT current_image;
  rotate90(current_image, previous_image);

  ImagePyramidT previous_pyramid(previous_image);
  previous_pyramid.buildPyramid();
  ImagePyramidT current_pyramid(current_image);
  current_pyramid.buildPyramid();
  Settings sof_settings;
  GradientPyramidT current_gradient;
  current_gradient.set(current_pyramid);
  current_gradient.forceNonLazyEvaluation(0);

  const std::vector<Vector2T> newSelectedFeatures = {{130, 81}, {152, 201}, {255, 161}, {372, 157}};
  const size_t rotated_width = previous_image.rows();
  const std::vector<Vector2T> default_current_features = {{rotated_width - 1 - 81, 130},
                                                          {rotated_width - 1 - 201, 152},
                                                          {rotated_width - 1 - 161, 255},
                                                          {rotated_width - 1 - 157, 372}};

  std::mt19937 gen(::testing::UnitTest::GetInstance()->random_seed());
  std::uniform_real_distribution<> uniform_dist{-5, 5};

  Matrix2T guess_affine_map;

  for (int test_case = 0; test_case < 20; test_case++) {
    std::vector<Vector2T> current_features;
    current_features.reserve(default_current_features.size());
    for (const Vector2T& x : default_current_features) {
      current_features.emplace_back(x + Vector2T(uniform_dist(gen), uniform_dist(gen)));
    }

    float angle_in_radians = (-90 + uniform_dist(gen)) * PI / 180;
    float cos_x = cos(angle_in_radians);
    float sin_x = sin(angle_in_radians);
    guess_affine_map << cos_x, sin_x, -sin_x, cos_x;

    size_t num_well_tracked_features = 0;
    for (size_t i = 0; i < newSelectedFeatures.size(); i++) {
      const Vector2T& prev_feature = newSelectedFeatures[i];
      const Vector2T& curr_feature = current_features[i];
      std::vector<STFeaturePatch> image_patches;
      uint32_t levels_mask;

      if (!STTracker::BuildPointCache(previous_pyramid, prev_feature, levels_mask, image_patches)) {
        std::cout << "Failed to build cache" << std::endl;
        continue;
      }

      float ncc = 0;
      float current_info[4];

      float curr_uv[] = {curr_feature[0], curr_feature[1]};

      bool res = tracker.TrackPointWithCache(current_gradient, current_pyramid, levels_mask, image_patches.size(),
                                             &image_patches[0], curr_uv, curr_uv, ncc, current_info, 300, 0.8,
                                             guess_affine_map);

      if (res) {
        num_well_tracked_features++;
      }
    }

    ASSERT_TRUE(num_well_tracked_features == newSelectedFeatures.size());
  }
}
}  // namespace test::sof
