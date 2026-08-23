
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

#include <iostream>

#include "ceres/ceres.h"
#include "ceres/loss_function.h"
#include "ceres/rotation.h"
#include "gflags/gflags.h"
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include "common/environment.h"
#include "common/include_json.h"
#include "edex/edex.h"
#include "utils/image_loader.h"

using namespace cuvslam;

DEFINE_string(edex_filename, "", "Path to edex file.");
DEFINE_string(output_folder, "/tmp/", "Path to output folder.");
DEFINE_double(visualization_boundary, 50, "The radius of the surrounding to visualize in meters.");
DEFINE_bool(visualize_ba, false, "Visualize the bundle adjustment from top-down and side-view.");
DEFINE_bool(visualize_frames, false, "Visualize every frame in the bundle adjustment.");
DEFINE_bool(optimize_intrinsics, true, "Optimize for intrinsics.");
DEFINE_bool(output_progress_to_stdout, false, "Output progress of Ceres to stdout.");
DEFINE_int32(max_iterations, 50, "Maximum number of iterations for bundle adjustment.");

static constexpr double kEPS = 1e-4;

template <typename T>
bool calculatePredictedObservation(const T* const ptr_pose_angleaxis, const T* const ptr_pose_translation,
                                   const T* const ptr_point, const T* const ptr_fx, const T* const ptr_fy,
                                   const T* const ptr_cx, const T* const ptr_cy, T* predicted_u, T* predicted_v) {
  T V_world_point[3], V_camera_point[3];

  // Transform the point from world to camera coordinate
  V_world_point[0] = ptr_point[0] - ptr_pose_translation[0];
  V_world_point[1] = ptr_point[1] - ptr_pose_translation[1];
  V_world_point[2] = ptr_point[2] - ptr_pose_translation[2];
  ceres::AngleAxisRotatePoint(ptr_pose_angleaxis, V_world_point, V_camera_point);

  if (V_camera_point[2] < kEPS) {
    return false;
  }

  *predicted_u = *ptr_fx * (V_camera_point[0] / V_camera_point[2]) + *ptr_cx;
  *predicted_v = *ptr_fy * (V_camera_point[1] / V_camera_point[2]) + *ptr_cy;

  return true;
}

struct IntrinsicsReprojectionError {
  IntrinsicsReprojectionError(double observed_x, double observed_y) : observed_x(observed_x), observed_y(observed_y) {}

  template <typename T>
  bool operator()(const T* const ptr_pose_angleaxis, const T* const ptr_pose_translation, const T* const ptr_point,
                  const T* const ptr_fx, const T* const ptr_fy, const T* const ptr_cx, const T* const ptr_cy,
                  T* residuals) const {
    T predicted_x;
    T predicted_y;

    if (!calculatePredictedObservation(ptr_pose_angleaxis, ptr_pose_translation, ptr_point, ptr_fx, ptr_fy, ptr_cx,
                                       ptr_cy, &predicted_x, &predicted_y)) {
      // Prevent divide-by-zero and points behind the camera.
      residuals[0] = T(0);
      residuals[1] = T(0);
      return true;
    }

    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - T(observed_x);
    residuals[1] = predicted_y - T(observed_y);

    return true;
  }

  double observed_x;
  double observed_y;
};

// MyIterationCallback prints the iteration number, the cost and the value of
// the parameter blocks every iteration.
class CameraCallback : public ceres::IterationCallback {
public:
  // Take in pointers of the pose blocks
  CameraCallback(int frame_id, const Eigen::Vector3d* pose_angleaxis, const Eigen::Vector3d* pose_translation,
                 const int image_width, const int image_height, const double* fx, const double* fy, const double* cx,
                 const double* cy, const std::string output_folder)
      : frame_id_(frame_id),
        pose_angleaxis_(pose_angleaxis),
        pose_translation_(pose_translation),
        image_width_(image_width),
        image_height_(image_height),
        fx_(fx),
        fy_(fy),
        cx_(cx),
        cy_(cy),
        output_folder_(output_folder){};

  ~CameraCallback() override = default;
  ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary) final {
    // Create image
    cv::Mat image(image_height_, image_width_, CV_8UC3, 0.0);

    for (size_t i = 0; i < points_.size(); i++) {
      // Draw point in image
      double observation_u = observations_[i].first;
      double observation_v = observations_[i].second;
      double predicted_u;
      double predicted_v;

      calculatePredictedObservation<double>(pose_angleaxis_->data(), pose_translation_->data(), points_[i]->data(), fx_,
                                            fy_, cx_, cy_, &predicted_u, &predicted_v);

      cv::circle(image, cv::Point(observation_u, observation_v), 3, cv::Scalar(0, 0, 255), -1);
      cv::circle(image, cv::Point(predicted_u, predicted_v), 2, cv::Scalar(0, 255, 0), -1);
      cv::line(image, cv::Point(observation_u, observation_v), cv::Point(predicted_u, predicted_v),
               cv::Scalar(255, 0, 0), 1);
    }

    // Add text of intrinsics to image
    std::string intrinsics_string = "fx: " + std::to_string(*fx_) + " fy: " + std::to_string(*fy_) +
                                    " cx: " + std::to_string(*cx_) + " cy: " + std::to_string(*cy_);
    cv::putText(image, intrinsics_string, cv::Point(5, image_height_ - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 255, 255), 1);

    // Convert iteration number to string and zero pad it
    std::string iteration_string = std::to_string(summary.iteration);
    std::string zero_padded_iteration_string = std::string(6 - iteration_string.length(), '0') + iteration_string;

    // Save image
    std::string filename =
        output_folder_ + "/image_" + std::to_string(frame_id_) + "_" + zero_padded_iteration_string + ".png";
    cv::imwrite(filename, image);

    return ceres::SOLVER_CONTINUE;
  }

  // Add the points and observations to the problem
  void addPoint(const Eigen::Vector3d* point, const std::pair<double, double> observation) {
    points_.push_back(point);
    observations_.push_back(observation);
  }

private:
  int frame_id_;
  const Eigen::Vector3d* pose_angleaxis_ = nullptr;
  const Eigen::Vector3d* pose_translation_ = nullptr;
  int image_width_, image_height_;
  const double *fx_, *fy_, *cx_, *cy_;
  std::vector<const Eigen::Vector3d*> points_;
  std::vector<std::pair<double, double>> observations_;
  const std::string output_folder_;
};

class SystemCallback : public ceres::IterationCallback {
public:
  // Take in pointers of the pose blocks
  SystemCallback(const double zoom, const double* fx, const double* fy, const double* cx, const double* cy,
                 const std::string output_folder)
      : zoom_(zoom), fx_(fx), fy_(fy), cx_(cx), cy_(cy), output_folder_(output_folder){};

  ~SystemCallback() override = default;
  ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary) final {
    // Convert iteration number to string and zero pad it
    std::string iteration_string = std::to_string(summary.iteration);
    std::string zero_padded_iteration_string = std::string(6 - iteration_string.length(), '0') + iteration_string;

    std::string intrinsics_string = "fx: " + std::to_string(*fx_) + " fy: " + std::to_string(*fy_) +
                                    " cx: " + std::to_string(*cx_) + " cy: " + std::to_string(*cy_);

    int min_value = -zoom_;
    int max_value = zoom_;

    {
      // Create image
      cv::Mat image(image_width_, image_height, CV_8UC3, 0.0);

      // Draw points
      for (size_t i = 0; i < points_.size(); i++) {
        int u = (points_[i]->operator()(0) - min_value) / (max_value - min_value) * image_height;
        int v = (points_[i]->operator()(2) - min_value) / (max_value - min_value) * image_width_;
        cv::circle(image, cv::Point(u, v), 1, cv::Scalar(0, 0, 255), -1);
      }

      // Draw poses
      for (size_t i = 0; i < translations_.size(); i++) {
        int u = (translations_[i]->operator()(0) - min_value) / (max_value - min_value) * image_height;
        int v = (translations_[i]->operator()(2) - min_value) / (max_value - min_value) * image_width_;
        cv::circle(image, cv::Point(u, v), 1, cv::Scalar(255, 0, 0), -1);
      }

      // Write intrinsics on the image
      cv::putText(image, intrinsics_string, cv::Point(5, image_height - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                  cv::Scalar(0, 255, 255), 1);

      // Save image
      std::string filename = output_folder_ + "/xz_" + zero_padded_iteration_string + ".png";
      cv::imwrite(filename, image);
    }

    {
      // Create image
      cv::Mat image(image_width_, image_height, CV_8UC3, 0.0);

      // Draw points
      for (size_t i = 0; i < points_.size(); i++) {
        int u = (points_[i]->operator()(0) - min_value) / (max_value - min_value) * image_height;
        int v = (points_[i]->operator()(1) - min_value) / (max_value - min_value) * image_width_;
        cv::circle(image, cv::Point(u, v), 1, cv::Scalar(0, 0, 255), -1);
      }

      // Draw poses
      for (size_t i = 0; i < translations_.size(); i++) {
        int u = (translations_[i]->operator()(0) - min_value) / (max_value - min_value) * image_height;
        int v = (translations_[i]->operator()(1) - min_value) / (max_value - min_value) * image_width_;
        cv::circle(image, cv::Point(u, v), 1, cv::Scalar(255, 0, 0), -1);
      }

      // Write intrinsics on the image
      cv::putText(image, intrinsics_string, cv::Point(5, image_height - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                  cv::Scalar(0, 255, 255), 1);

      // Save image
      std::string filename = output_folder_ + "/xy_" + zero_padded_iteration_string + ".png";
      cv::imwrite(filename, image);
    }

    {
      // Create image
      cv::Mat image(image_width_, image_height, CV_8UC3, 0.0);

      // Draw points
      for (size_t i = 0; i < points_.size(); i++) {
        int u = (points_[i]->operator()(1) - min_value) / (max_value - min_value) * image_height;
        int v = (points_[i]->operator()(2) - min_value) / (max_value - min_value) * image_width_;
        cv::circle(image, cv::Point(u, v), 1, cv::Scalar(0, 0, 255), -1);
      }

      // Draw poses
      for (size_t i = 0; i < translations_.size(); i++) {
        int u = (translations_[i]->operator()(1) - min_value) / (max_value - min_value) * image_height;
        int v = (translations_[i]->operator()(2) - min_value) / (max_value - min_value) * image_width_;
        cv::circle(image, cv::Point(u, v), 1, cv::Scalar(255, 0, 0), -1);
      }

      // Write intrinsics on the image
      cv::putText(image, intrinsics_string, cv::Point(5, image_height - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                  cv::Scalar(0, 255, 255), 1);

      // Save image
      std::string filename = output_folder_ + "/yz_" + zero_padded_iteration_string + ".png";
      cv::imwrite(filename, image);
    }

    return ceres::SOLVER_CONTINUE;
  }

  // Add the points and observations to the problem
  void addPoint(const Eigen::Vector3d* point) { points_.push_back(point); }

  void addTranslation(const Eigen::Vector3d* translation) { translations_.push_back(translation); }

private:
  std::vector<const Eigen::Vector3d*> translations_;
  std::vector<const Eigen::Vector3d*> points_;
  double zoom_;
  int image_width_ = 1000;
  int image_height = 1000;
  const double *fx_, *fy_, *cx_, *cy_;
  const std::string output_folder_;
};

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, /*remove flags = */ true);
  Trace::SetVerbosity(Trace::Verbosity::Debug);

  cuvslam::edex::EdexFile edex_file;
  edex_file.read(FLAGS_edex_filename);
  ceres::Problem problem;

  // Initialize a new map for 3D points since Ceres uses double precision.
  std::map<TrackId, Eigen::Vector3d> var_points;
  for (const auto& track : edex_file.tracks3D_) {
    var_points[track.first] = track.second.cast<double>();
  }

  // Initialize the pose parameters for each frame
  std::vector<Eigen::Vector3d> var_pose_angleaxis(edex_file.rigPositions_.size());
  std::vector<Eigen::Vector3d> var_pose_translations(edex_file.rigPositions_.size());
  for (const auto& camera : edex_file.rigPositions_) {
    auto frame_id = camera.first;
    auto transform = camera.second;

    // Convert the rotation matrix to double and set to default memory layout
    // TODO(hrabeti): Figure out what the ordering of operations during cost
    // function calculation should be to avoid this transposition.
    Eigen::Matrix3d rotation_matrix = transform.linear().cast<double>().transpose();
    ceres::RotationMatrixToAngleAxis<double>(rotation_matrix.data(), var_pose_angleaxis[frame_id].data());

    // Set the translation
    Eigen::Vector3d translation = transform.translation().cast<double>();
    var_pose_translations[frame_id] = translation;
  }

  std::cout << edex_file.cameras_.size() << std::endl;
  assert(edex_file.cameras_.size() == 1);  // We only support monocular cameras atm

  int image_width = edex_file.cameras_[0].intrinsics.resolution[0];
  int image_height = edex_file.cameras_[0].intrinsics.resolution[1];
  double var_fx = edex_file.cameras_[0].intrinsics.focal[0];
  double var_fy = edex_file.cameras_[0].intrinsics.focal[1];
  double var_cx = edex_file.cameras_[0].intrinsics.principal[0];
  double var_cy = edex_file.cameras_[0].intrinsics.principal[1];
  std::cout << "Intrinsics initialization - fx: " << var_fx << " fy: " << var_fy << " cx: " << var_cx
            << " cy: " << var_cy << std::endl;

  std::vector<std::unique_ptr<ceres::AutoDiffCostFunction<IntrinsicsReprojectionError, 2, 3, 3, 3, 1, 1, 1, 1>>>
      cost_functions;

  ceres::Solver::Options options;

  std::vector<std::unique_ptr<CameraCallback>> camera_callbacks;
  SystemCallback system_callback(FLAGS_visualization_boundary, &var_fx, &var_fy, &var_cx, &var_cy, FLAGS_output_folder);
  if (FLAGS_visualize_ba) {
    for (const auto& point : var_points) {
      system_callback.addPoint(&point.second);
    }
    for (const auto& translation : var_pose_translations) {
      system_callback.addTranslation(&translation);
    }
    options.callbacks.push_back(&system_callback);
  }

  for (const auto& frame : edex_file.cameras_[0].tracks2D) {
    auto frame_id = frame.first;
    auto observations = frame.second;

    camera_callbacks.emplace_back(std::make_unique<CameraCallback>(
        frame_id, &var_pose_angleaxis[frame_id], &var_pose_translations[frame_id], image_width, image_height, &var_fx,
        &var_fy, &var_cx, &var_cy, FLAGS_output_folder));

    for (const auto& observation : observations) {
      const double u = observation.second[0];
      const double v = observation.second[1];
      cost_functions.emplace_back(
          std::make_unique<ceres::AutoDiffCostFunction<IntrinsicsReprojectionError, 2, 3, 3, 3, 1, 1, 1, 1>>(
              new IntrinsicsReprojectionError(u, v)));

      double* ptr_pose_angleaxis = var_pose_angleaxis[frame_id].data();
      double* ptr_pose_translation = var_pose_translations[frame_id].data();
      double* ptr_point = var_points[observation.first].data();

      problem.AddResidualBlock(cost_functions.back().get(), new ceres::CauchyLoss(3), ptr_pose_angleaxis,
                               ptr_pose_translation, ptr_point, &var_fx, &var_fy, &var_cx, &var_cy);

      camera_callbacks.back()->addPoint(&var_points[observation.first], std::make_pair(u, v));
    }

    if (FLAGS_visualize_frames) {
      options.callbacks.push_back(camera_callbacks.back().get());
    }
  }

  std::cout << "Number of 3d points: " << edex_file.tracks3D_.size() << std::endl;
  std::cout << "Number of cost functions: " << cost_functions.size() << std::endl;

  // Set initial pose and features observed from the initial pose as constant.
  // This keeps the scale (at least of the beginning of the trajectory) the
  // same as the input.
  problem.SetParameterBlockConstant(var_pose_angleaxis[0].data());
  problem.SetParameterBlockConstant(var_pose_translations[0].data());
  if (!FLAGS_optimize_intrinsics) {
    problem.SetParameterBlockConstant(&var_fx);
    problem.SetParameterBlockConstant(&var_fy);
    problem.SetParameterBlockConstant(&var_cx);
    problem.SetParameterBlockConstant(&var_cy);
  }

  options.minimizer_progress_to_stdout = FLAGS_output_progress_to_stdout;
  if (FLAGS_visualize_frames || FLAGS_visualize_ba) {
    options.update_state_every_iteration = true;
  }
  options.max_num_iterations = FLAGS_max_iterations;

  ceres::Solver::Summary summary;
  std::cout << "Starting solve..." << std::endl;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << "\n";

  std::cout << "Intrinsics initialization - fx: " << edex_file.cameras_[0].intrinsics.focal[0]
            << " fy: " << edex_file.cameras_[0].intrinsics.focal[1]
            << " cx: " << edex_file.cameras_[0].intrinsics.principal[0]
            << " cy: " << edex_file.cameras_[0].intrinsics.principal[1] << std::endl;
  std::cout << "Intrinsics estimate - fx: " << var_fx << " fy: " << var_fy << " cx: " << var_cx << " cy: " << var_cy
            << std::endl;

  std::string edex_filename = FLAGS_output_folder + "/refined.edex";
  std::cout << "Writing new edex file to " << edex_filename << std::endl;

  // Set intrinsics to the estimated values.
  edex_file.cameras_[0].intrinsics.focal[0] = var_fx;
  edex_file.cameras_[0].intrinsics.focal[1] = var_fy;
  edex_file.cameras_[0].intrinsics.principal[0] = var_cx;
  edex_file.cameras_[0].intrinsics.principal[1] = var_cy;

  // Set the 3d point locations to the estimated values.
  for (const auto& track : edex_file.tracks3D_) {
    edex_file.tracks3D_[track.first] = var_points[track.first].cast<float>();
  }

  // Set the poses to the estimated values.
  for (const auto& camera : edex_file.rigPositions_) {
    // Set the rotation matrix
    Eigen::Matrix3d rotation_matrix;
    ceres::AngleAxisToRotationMatrix(var_pose_angleaxis[camera.first].data(), rotation_matrix.data());
    edex_file.rigPositions_[camera.first].linear() = rotation_matrix.transpose().cast<float>();

    // Set the translation vector
    edex_file.rigPositions_[camera.first].translation() = var_pose_translations[camera.first].cast<float>();
  }

  edex_file.write(edex_filename);

  return 0;
}
