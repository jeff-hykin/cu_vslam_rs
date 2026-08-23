
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

#include "common/environment.h"
#include "common/include_gtest.h"
#include "sof/selection.h"
#include "sof/sof.h"
#include "sof/sof_config.h"
#include "sof/st_tracker.h"
#include "utils/image_loader.h"

#include <fstream>
#include <random>

#include "gflags/gflags.h"
#ifdef WITH_OPENCV
#include "opencv2/opencv.hpp"
#endif

#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

DEFINE_int32(dx, 5, "dx in pixels (integer)");
DEFINE_int32(dy, 5, "dy in pixels (integer)");
DEFINE_int32(num_features, 100, "number of features to detect and track");
DEFINE_double(noise, 1., "noise level");
DEFINE_string(test_file, "sof/left.png", "test image");
DEFINE_string(left, "sof/left.png", "left test image");
DEFINE_string(right, "sof/right.png", "right test image");

namespace test {
using namespace cuvslam;

static void TestShift(cuvslam::sof::IFeatureTracker& tracker) {
  const std::string testdata_folder = CUVSLAM_TEST_ASSETS;

  utils::ImageLoaderT image_loader;
  ASSERT_TRUE(image_loader.load(testdata_folder + FLAGS_test_file));

  const ImageMatrixT& prev_image = image_loader.getImage().cast<float>();
  const size_t w = prev_image.cols();
  const size_t h = prev_image.rows();
  auto current_image = ImageMatrixT();
  current_image.resize(h, w);
  current_image.setZero();

  const auto deltas = {
      Vector2N(FLAGS_dx, FLAGS_dy),
  };

  std::mt19937 engine(::testing::UnitTest::GetInstance()->random_seed());
  const float noise_level = static_cast<float>(FLAGS_noise);
  std::normal_distribution noise(0.f, noise_level);

  for (const auto& delta : deltas) {
    std::cout << "\n\nDelta: (" << delta.x() << ", " << delta.y() << ")" << std::endl;
    // move right
    current_image.block(0, 0, h - delta.y(), w - delta.x()) =
        prev_image.block(delta.y(), delta.x(), h - delta.y(), w - delta.x());

    if (noise_level > 0.f) {
      std::cout << "adding gaussian noise with sigma = " << noise_level << "\n";
      for (int i = 0; i < current_image.rows(); ++i) {
        for (int j = 0; j < current_image.cols(); ++j) {
          current_image(i, j) += noise(engine);
        }
      }
    }

    auto prev_pyramid = cuvslam::sof::ImagePyramidT(prev_image);
    auto current_pyramid = cuvslam::sof::ImagePyramidT(current_image);
    EXPECT_TRUE(prev_pyramid.buildPyramid() > 0);
    EXPECT_TRUE(current_pyramid.buildPyramid() > 0);
    cuvslam::sof::Settings sof_settings;
    cuvslam::sof::GradientPyramidT cur_gradients_pyramid;
    cur_gradients_pyramid.set(current_pyramid);

    cuvslam::sof::TracksVector tracks;
    std::vector<Vector2T> new_tracks;

    cuvslam::sof::GoodFeaturesToTrackDetector detector;
    detector.computeGFTTAndSelectFeatures(cur_gradients_pyramid, 0, 0, 0, 0, nullptr, tracks, FLAGS_num_features,
                                          new_tracks);

    size_t n_tracked = 0;
    size_t n_acc1 = 0;
    size_t n_acc2 = 0;
    size_t n_acc3 = 0;
    size_t n_acc4 = 0;

    std::ofstream ofs("tracks.txt");

    for (auto& xy : new_tracks) {
      const Vector2T prev_track(xy(0), xy(1));
      Vector2T current_track = prev_track;
      Matrix2T info;  // ignore
      if (tracker.trackPoint(cur_gradients_pyramid, cur_gradients_pyramid, prev_pyramid, current_pyramid, prev_track,
                             current_track, info)) {
        ++n_tracked;
        Vector2T motion = current_track - prev_track;
        const Vector2T track_delta = current_track - prev_track + Vector2T(delta.x(), delta.y());

        const auto err = track_delta.norm();

        if (err < 0.05) {
          ++n_acc1;
        }

        if (err < 0.1f) {
          ++n_acc2;
        }

        if (err < 0.5f) {
          ++n_acc3;
        }

        if (err < 2.0f) {
          ++n_acc4;
        }

        // EXPECT_TRUE(track_delta   - delta == Vector2T(0, 0));

        ofs << xy(0) << " " << xy(1) << " ";
        ofs << motion(0) << " " << motion(1) << "\n";
      }
    }

    auto total = FLAGS_num_features;

    std::cout << "tracked:\t" << n_tracked << std::endl;
    std::cout << "total:\t\t" << total << std::endl;
    std::cout << "track_rate:\t" << 100.f * n_tracked / total << "%" << std::endl;
    std::cout << "\terr <0.05:\t" << 100.f * n_acc1 / n_tracked << "%" << std::endl;
    std::cout << "\terr <0.1:\t" << 100.f * n_acc2 / n_tracked << "%" << std::endl;
    std::cout << "\terr <0.5:\t" << 100.f * n_acc3 / n_tracked << "%" << std::endl;
    std::cout << "\terr <2.f:\t" << 100.f * n_acc4 / n_tracked << "%" << std::endl;
  }
}

TEST(KLTTracker, ShiftNewLK) { TestShift(*sof::CreateTracker(sof::TrackerType::KLT)); }

TEST(KLTTracker, ShiftOldLK) { TestShift(*sof::CreateTracker(sof::TrackerType::LK)); }

TEST(STTracker, ShiftST2) {
  auto tracker = cuvslam::sof::STTracker(20, 0);
  TestShift(tracker);
}

TEST(STTracker, ShiftST6) {
  auto tracker = cuvslam::sof::STTracker(0, 20);
  TestShift(tracker);
}

static void TestStereo(cuvslam::sof::IFeatureTracker& tracker) {
  const std::string root = CUVSLAM_TEST_ASSETS;

  utils::ImageLoaderT left_loader;
  ASSERT_TRUE(left_loader.load(root + FLAGS_left));

  utils::ImageLoaderT right_loader;
  ASSERT_TRUE(right_loader.load(root + FLAGS_right));

  const ImageMatrixT& left_image = left_loader.getImage().cast<float>();
  const ImageMatrixT& right_image = right_loader.getImage().cast<float>();

  cuvslam::sof::ImagePyramidT left_p(left_image);
  left_p.buildPyramid();
  cuvslam::sof::ImagePyramidT right_p(right_image);
  right_p.buildPyramid();
  cuvslam::sof::Settings sof_settings;
  cuvslam::sof::GradientPyramidT right_g;
  right_g.set(right_p);

  cuvslam::sof::GradientPyramidT left_g;
  left_g.set(left_p);

  cuvslam::sof::TracksVector tracks;
  std::vector<Vector2T> new_tracks;

  cuvslam::sof::GoodFeaturesToTrackDetector detector;
  detector.computeGFTTAndSelectFeatures(left_g, 0, 0, 0, 0, nullptr, tracks, FLAGS_num_features, new_tracks);

  std::ofstream tracker_log("tracker_log.txt");

  std::cout << "total " << new_tracks.size() << " points\n";

  int num_tracked = 0;
  for (auto& xy : new_tracks) {
    Vector2T xy_r = xy;
    Matrix2T info;  // ignore
    if (tracker.trackPoint(left_g, right_g, left_p, right_p, xy, xy_r, info)) {
      ++num_tracked;
      tracker_log << xy(0) << " " << xy(1) << " " << xy_r(0) << " " << xy_r(1) << " " << xy_r(0) - xy(0) << " "
                  << xy_r(1) - xy(1) << "\n";
    }
  }
  std::cout << "tracked " << num_tracked << " points\n";
}

TEST(KLTTracker, StereoNewLK) { TestStereo(*sof::CreateTracker(sof::TrackerType::KLT)); }

TEST(KLTTracker, StereoOldLK) { TestStereo(*sof::CreateTracker(sof::TrackerType::LK)); }

void TestSequence(cuvslam::sof::IFeatureTracker& tracker) {
  std::string fmt = "%s/kitti/01/01.0.%04d.png";
  std::vector buffer(4096, '\0');

  auto root = Environment::GetVar(Environment::CUVSLAM_DATASETS);

  std::sprintf(buffer.data(), fmt.c_str(), root.c_str(), 1);

  utils::ImageLoaderT loader;
  if (!loader.load(buffer.data())) {
    return;
  }
  ImageMatrixT image_p = loader.getImage().cast<float>();

  std::vector<Vector2T> tracks;

  for (int i = 2; i < 1101; ++i) {
    std::sprintf(buffer.data(), fmt.c_str(), root.c_str(), i);
    if (!loader.load(buffer.data())) {
      return;
    }
    ImageMatrixT image_c = loader.getImage().cast<float>();

    cuvslam::sof::ImagePyramidT pyramid_p(image_p);
    pyramid_p.buildPyramid();

    cuvslam::sof::ImagePyramidT pyramid_c(image_c);
    pyramid_c.buildPyramid();
    cuvslam::sof::Settings sof_settings;
    cuvslam::sof::GradientPyramidT grad_p;
    grad_p.set(pyramid_p);

    cuvslam::sof::GradientPyramidT grad_c;
    grad_c.set(pyramid_c);

    // pretend that we have a new KF
    if (tracks.size() < 200) {
      cuvslam::sof::GoodFeaturesToTrackDetector detector;

      // it is OK to pick tracks on top of each other: we just
      // want to make sure that tracking works
      cuvslam::sof::TracksVector fake_tracks;
      std::vector<Vector2T> new_tracks;
      detector.computeGFTTAndSelectFeatures(grad_p, 0, 0, 0, 0, nullptr, fake_tracks, FLAGS_num_features, new_tracks);

      tracks.insert(tracks.end(), new_tracks.begin(), new_tracks.end());
    }

    std::vector<Vector2T> current_tracks;
    current_tracks.swap(tracks);

#ifdef WITH_OPENCV
    auto image = cv::Mat(image_c.rows(), image_c.cols(), CV_32FC1, image_c.data()).clone();
    image *= 1.f / 255.f;
    cv::Mat color_image;
    cv::cvtColor(image, color_image, CV_GRAY2RGB);
#endif

    for (auto& xy : current_tracks) {
      Vector2T new_xy = xy;
      Matrix2T info;  // ignore
      if (tracker.trackPoint(grad_p, grad_c, pyramid_p, pyramid_c, xy, new_xy, info)) {
        tracks.push_back(new_xy);

#ifdef WITH_OPENCV
        cv::circle(color_image, cv::Point2f(new_xy(0), new_xy(1)), 3, cv::Scalar(0, 1, 0), 1);
        cv::line(color_image, cv::Point2f(new_xy(0), new_xy(1)), cv::Point2f(xy(0), xy(1)), cv::Scalar(0, 1, 0), 1);
#endif

      } else {
#ifdef WITH_OPENCV
        cv::circle(color_image, cv::Point2f(new_xy(0), new_xy(1)), 3, cv::Scalar(0, 0, 1), 1);
#endif
      }
    }

#ifdef WITH_OPENCV
    cv::imshow("cam 0", color_image);
    cv::waitKey();
#endif

    image_p.swap(image_c);
  }
}

TEST(KLTTracker, SequenceNewLK) { TestSequence(*sof::CreateTracker(sof::TrackerType::KLT)); }

TEST(KLTTracker, SequenceOldLK) { TestSequence(*sof::CreateTracker(sof::TrackerType::LK)); }

}  // namespace test
