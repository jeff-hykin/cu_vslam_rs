
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

#include "refinement/refinement.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include "ceres/ceres.h"
#include "refinement/cost_pinhole.h"
#include "refinement/cost_rational_polynomial.h"
#include "refinement/loss_functions.h"

namespace cuvslam::refinement {

Pose invertPose(const Pose& pose_const) {
  Pose pose = pose_const;
  Eigen::Map<Eigen::Quaternionf> rotation_quaternion_map(pose.rotation.data());
  Eigen::Map<Eigen::Vector3f> translation_map(pose.translation.data());
  rotation_quaternion_map = rotation_quaternion_map.inverse();
  translation_map = -rotation_quaternion_map.toRotationMatrix() * translation_map;
  return pose;
}

Eigen::Matrix4f poseToMatrix(const Pose& pose_const) {
  const Eigen::Map<const Eigen::Quaternionf> rotation_quaternion_map(pose_const.rotation.data());
  const Eigen::Map<const Eigen::Vector3f> translation_map(pose_const.translation.data());

  Eigen::Matrix4f matrix = Eigen::Matrix4f::Identity();
  matrix.block<3, 3>(0, 0) = rotation_quaternion_map.toRotationMatrix();
  matrix.block<3, 1>(0, 3) = translation_map;
  return matrix;
}

/**
 * @brief Callback to export the state at each iteration.
 *
 * This callback captures the state of poses, points, and camera parameters
 * after each iteration when export_iteration_state is enabled.
 */
class StateExportCallback : public ceres::IterationCallback {
public:
  StateExportCallback(const std::unordered_map<uint64_t, Eigen::Vector3d>* angleaxis_rig_from_world,
                      const std::unordered_map<uint64_t, Eigen::Vector3d>* translation_rig_from_world,
                      const std::unordered_map<uint64_t, Eigen::Vector3d>* points, const double* fx, const double* fy,
                      const double* cx, const double* cy, const std::array<double, 6>* radial_distortion,
                      const std::array<double, 2>* tangential_distortion, BundleAdjustmentProblemSummary& summary,
                      const BundleAdjustmentProblemOptions& options, const BundleAdjustmentProblem& problem,
                      const double max_reprojection_error, const bool visualize_frames = true)
      : angleaxis_rig_from_world_(angleaxis_rig_from_world),
        translation_rig_from_world_(translation_rig_from_world),
        points_(points),
        fx_(fx),
        fy_(fy),
        cx_(cx),
        cy_(cy),
        radial_distortion_(radial_distortion),
        tangential_distortion_(tangential_distortion),
        summary_(summary),
        options_(options),
        problem_(problem),
        max_reprojection_error_(max_reprojection_error),
        visualize_frames_(visualize_frames) {
    // Create output directory for visualization
    if (visualize_frames_) {
      std::filesystem::create_directories("visualization");
    }
  }

  ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary) override {
    // Create a copy of the current state of rigs_from_world
    std::unordered_map<uint64_t, Pose> rigs_from_world;
    for (const auto& [frame_id, angle_axis] : *angleaxis_rig_from_world_) {
      Pose pose;
      // Convert angle-axis to rotation matrix
      Eigen::Vector4d ceres_quaternion;
      ceres::AngleAxisToQuaternion<double>(angle_axis.data(), ceres_quaternion.data());

      // Ceres Quaternion has w, x, y, z order.
      // https://github.com/ceres-solver/ceres-solver/blob/master/include/ceres/rotation.h#L38
      pose.rotation[3] = ceres_quaternion[0];
      pose.rotation[0] = ceres_quaternion[1];
      pose.rotation[1] = ceres_quaternion[2];
      pose.rotation[2] = ceres_quaternion[3];

      // Store translation
      Eigen::Map<Eigen::Vector3f> t(pose.translation.data());
      t = translation_rig_from_world_->at(frame_id).cast<float>();

      rigs_from_world[frame_id] = pose;
    }
    summary_.iteration_rigs_from_world.push_back(rigs_from_world);

    // Create a copy of the current state of points_in_world
    std::unordered_map<uint64_t, Landmark> points_in_world;
    for (const auto& [point_id, point] : *points_) {
      Landmark landmark;
      Eigen::Map<Eigen::Vector3f> coords(landmark.coords.data());
      coords = point.cast<float>();
      points_in_world[point_id] = landmark;
    }
    summary_.iteration_points_in_world.push_back(points_in_world);

    // Create a copy of the current state of the camera parameters
    Rig rig = problem_.rig;  // Copy the entire original rig structure
    // Update only the optimized camera parameters
    rig.cameras[0].focal[0] = *fx_;
    rig.cameras[0].focal[1] = *fy_;
    rig.cameras[0].principal[0] = *cx_;
    rig.cameras[0].principal[1] = *cy_;

    // Update distortion parameters if using polynomial distortion model
    if (problem_.rig.cameras[0].distortion.model == Distortion::Model::Polynomial && radial_distortion_ != nullptr &&
        tangential_distortion_ != nullptr) {
      // OpenCV distortion order: [k1, k2, p1, p2, k3, k4, k5, k6]
      rig.cameras[0].distortion.parameters[0] = (*radial_distortion_)[0];      // k1
      rig.cameras[0].distortion.parameters[1] = (*radial_distortion_)[1];      // k2
      rig.cameras[0].distortion.parameters[2] = (*tangential_distortion_)[0];  // p1
      rig.cameras[0].distortion.parameters[3] = (*tangential_distortion_)[1];  // p2
      rig.cameras[0].distortion.parameters[4] = (*radial_distortion_)[2];      // k3
      rig.cameras[0].distortion.parameters[5] = (*radial_distortion_)[3];      // k4
      rig.cameras[0].distortion.parameters[6] = (*radial_distortion_)[4];      // k5
      rig.cameras[0].distortion.parameters[7] = (*radial_distortion_)[5];      // k6
    }

    summary_.iteration_cameras.push_back(rig);

    // Create visualization images for each frame
    if (visualize_frames_) {
      visualizeFrames(summary.iteration);
    }

    return ceres::SOLVER_CONTINUE;
  }

private:
  // Struct to represent an RGB color
  struct Color {
    unsigned char r, g, b;
  };

  // Function to write a PPM image file
  void writePPM(const std::string& filename, int width, int height, const std::vector<Color>& pixels) {
    std::ofstream file(filename);
    if (!file.is_open()) {
      std::cerr << "Failed to open file: " << filename << std::endl;
      return;
    }

    // Write the PPM header
    file << "P3\n" << width << " " << height << "\n255\n";

    // Write the pixel data
    for (const auto& pixel : pixels) {
      file << static_cast<int>(pixel.r) << " " << static_cast<int>(pixel.g) << " " << static_cast<int>(pixel.b) << " ";
    }
    file.close();
  }

  // Function to draw a circle on the image
  void drawCircle(std::vector<Color>& image, int width, int x, int y, int radius, const Color& color) {
    for (int dy = -radius; dy <= radius; dy++) {
      for (int dx = -radius; dx <= radius; dx++) {
        // Check if the point is within the circle
        if (dx * dx + dy * dy <= radius * radius) {
          int px = x + dx;
          int py = y + dy;
          // Check if the pixel is within the image
          if (px >= 0 && px < width && py >= 0 && py < (int)image.size() / width) {
            int index = py * width + px;
            if (index >= 0 && index < (int)image.size()) {
              image[index] = color;
            }
          }
        }
      }
    }
  }

  // Function to draw a line on the image
  void drawLine(std::vector<Color>& image, int width, int x0, int y0, int x1, int y1, const Color& color) {
    int dx = abs(x1 - x0);
    int dy = abs(y1 - y0);
    int sx = (x0 < x1) ? 1 : -1;
    int sy = (y0 < y1) ? 1 : -1;
    int err = dx - dy;

    while (true) {
      // Check if the pixel is within the image
      if (x0 >= 0 && x0 < width && y0 >= 0 && y0 < (int)image.size() / width) {
        int index = y0 * width + x0;
        if (index >= 0 && index < (int)image.size()) {
          image[index] = color;
        }
      }

      if (x0 == x1 && y0 == y1) break;
      int e2 = 2 * err;
      if (e2 > -dy) {
        err -= dy;
        x0 += sx;
      }
      if (e2 < dx) {
        err += dx;
        y0 += sy;
      }
    }
  }

  void visualizeFrames(int iteration) {
    // Define colors
    // const Color RED = {255, 0, 0};       // For observations
    // const Color BLUE = {0, 0, 255};      // For projected points
    // const Color GREEN = {0, 255, 0};     // For points hwithin max_reprojection_error
    // const Color YELLOW = {255, 255, 0};  // For points beyond max_reprojection_error

    // // For each frame, create a visualization
    // for (const auto& [frame_id, observations] : problem_.observations) {
    //   if (frame_id > 50) {
    //     // continue;
    //   }

    //   // Get the camera parameters
    //   const double fx = *fx_;
    //   const double fy = *fy_;
    //   const double cx = *cx_;
    //   const double cy = *cy_;

    //   // Get frame size from the problem
    //   int width = problem_.rig.cameras[0].size[0];
    //   int height = problem_.rig.cameras[0].size[1];

    //   // Create a blank image (black background)
    //   std::vector<Color> image(width * height, {0, 0, 0});

    //   // Get the camera pose
    //   const Eigen::Vector3d& angle_axis = angleaxis_rig_from_world_->at(frame_id);
    //   const Eigen::Vector3d& translation = translation_rig_from_world_->at(frame_id);

    //   // For each observation in this frame
    //   for (const auto& observation : observations) {
    //     // Draw the original observation as a red dot
    //     drawCircle(image, width, static_cast<int>(observation.u), static_cast<int>(observation.v), 3, RED);

    //     // Get the 3D point
    //     const Eigen::Vector3d& point_3d = points_->at(observation.id);

    //     // Project the 3D point into the image using calculatePredictedObservation
    //     double u_proj, v_proj;

    //     bool in_front_of_camera = false;

    //     if (problem_.rig.cameras[0].distortion.model == Distortion::Model::Polynomial) {
    //       // Use rational polynomial model
    //       in_front_of_camera = refinement::rational_polynomial::calculatePredictedObservation(
    //           angle_axis.data(), translation.data(), point_3d.data(), &fx, &fy, &cx, &cy, radial_distortion_->data(),
    //           tangential_distortion_->data(), &u_proj, &v_proj, options_.symmetric_focal_length);
    //     } else {
    //       // Use pinhole model
    //       in_front_of_camera = refinement::pinhole::calculatePredictedObservation(
    //           angle_axis.data(), translation.data(), point_3d.data(), &fx, &fy, &cx, &cy, &u_proj, &v_proj,
    //           options_.symmetric_focal_length);
    //     }

    //     if (!in_front_of_camera) {
    //       // Skip points behind the camera
    //       continue;
    //     }

    //     // Calculate reprojection error (squared distance between observed and projected points)
    //     double dx = observation.u - u_proj;
    //     double dy = observation.v - v_proj;
    //     double reprojection_error = std::sqrt(dx * dx + dy * dy);

    //     // Determine line color based on reprojection error
    //     const Color& line_color = (reprojection_error <= max_reprojection_error_) ? GREEN : YELLOW;
    //     // Draw a line connecting them (green if within threshold, yellow if beyond)
    //     drawLine(image, width, static_cast<int>(observation.u), static_cast<int>(observation.v),
    //              static_cast<int>(u_proj), static_cast<int>(v_proj), line_color);

    //     // Check if projected point is within the image
    //     if (u_proj < 0 || u_proj >= width || v_proj < 0 || v_proj >= height) {
    //       continue;
    //     }
    //     // Draw the projected point as a blue dot
    //     drawCircle(image, width, static_cast<int>(u_proj), static_cast<int>(v_proj), 2, BLUE);
    //   }

    //   // Save the image
    //   std::string filename =
    //       "visualization/frame_" + std::to_string(frame_id) + "_iter_" + std::to_string(iteration) + ".ppm";
    //   writePPM(filename, width, height, image);
    // }

    // // Create a visualization of X,Z coordinates for poses and features
    // int viz_width = 1024;
    // int viz_height = 1024;

    // // Find bounds of all points and poses
    // double min_x = std::numeric_limits<double>::max();
    // double max_x = std::numeric_limits<double>::lowest();
    // double min_y = std::numeric_limits<double>::max();
    // double max_y = std::numeric_limits<double>::lowest();
    // double min_z = std::numeric_limits<double>::max();
    // double max_z = std::numeric_limits<double>::lowest();

    // // Check poses
    // for (const auto& [frame_id, translation] : *translation_rig_from_world_) {
    //   // Get the orientation as angle-axis
    //   const auto& orientation = angleaxis_rig_from_world_->at(frame_id);

    //   // Convert angle-axis to rotation matrix
    //   Eigen::Matrix3d rotation_matrix;
    //   ceres::AngleAxisToRotationMatrix<double>(orientation.data(), rotation_matrix.data());

    //   // Compute the inverse rotation (transpose for rotation matrices)
    //   Eigen::Matrix3d inverse_rotation = rotation_matrix.transpose();

    //   // Apply the inverse rotation to the translation
    //   Eigen::Vector3d rotated_translation = -(inverse_rotation * translation);

    //   min_x = std::min(min_x, rotated_translation[0]);
    //   max_x = std::max(max_x, rotated_translation[0]);
    //   min_y = std::min(min_y, rotated_translation[1]);
    //   max_y = std::max(max_y, rotated_translation[1]);
    //   min_z = std::min(min_z, rotated_translation[2]);
    //   max_z = std::max(max_z, rotated_translation[2]);
    // }

    // // // Check 3D points
    // // for (const auto& [point_id, point] : *points_) {
    // //   min_x = std::min(min_x, point.x());
    // //   max_x = std::max(max_x, point.x());
    // //   min_z = std::min(min_z, point.z());
    // //   max_z = std::max(max_z, point.z());
    // // }

    // // Add padding
    // double padding = 0;
    // min_x -= padding;
    // max_x += padding;
    // min_y -= padding;
    // max_y += padding;
    // min_z -= padding;
    // max_z += padding;

    // // XZ visualization
    // {
    //   std::vector<Color> xz_image(viz_width * viz_height, {0, 0, 0});  // White background
    //   std::vector<Color> xy_image(viz_width * viz_height, {0, 0, 0});  // White background
    //   std::vector<Color> yz_image(viz_width * viz_height, {0, 0, 0});  // White background

    //   // Draw poses in red
    //   for (const auto& [frame_id, translation] : *translation_rig_from_world_) {
    //     // Get the orientation as angle-axis
    //     const auto& orientation = angleaxis_rig_from_world_->at(frame_id);

    //     // Convert angle-axis to rotation matrix
    //     Eigen::Matrix3d rotation_matrix;
    //     ceres::AngleAxisToRotationMatrix<double>(orientation.data(), rotation_matrix.data());

    //     // Compute the inverse rotation (transpose for rotation matrices)
    //     Eigen::Matrix3d inverse_rotation = rotation_matrix.transpose();

    //     // Apply the inverse rotation to the translation
    //     Eigen::Vector3d rotated_translation = -(inverse_rotation * translation);

    //     // Use the rotated translation for visualization
    //     int x = static_cast<int>((rotated_translation[0] - min_x) / (max_x - min_x) * viz_width);
    //     int y = static_cast<int>((rotated_translation[1] - min_y) / (max_y - min_y) * viz_width);
    //     int z = static_cast<int>((rotated_translation[2] - min_z) / (max_z - min_z) * viz_width);

    //     drawCircle(xz_image, viz_width, x, z, 1, {255, 0, 0});  // Red for poses
    //     drawCircle(xy_image, viz_width, x, y, 1, {255, 0, 0});  // Red for poses
    //     drawCircle(yz_image, viz_width, y, z, 1, {255, 0, 0});  // Red for poses
    //   }

    //   // Draw 3D points in green
    //   for (const auto& [point_id, point] : *points_) {
    //     int x = static_cast<int>((point.x() - min_x) / (max_x - min_x) * viz_width);
    //     int y = static_cast<int>((point.y() - min_y) / (max_y - min_y) * viz_width);
    //     int z = static_cast<int>((point.z() - min_z) / (max_z - min_z) * viz_width);

    //     // Only draw points if they are within the image bounds
    //     if (x >= 0 && x < viz_width && y >= 0 && y < viz_height) {
    //       drawCircle(xy_image, viz_width, x, y, 1, {0, 255, 0});  // Green for points
    //     }
    //     if (z >= 0 && z < viz_height && y >= 0 && y < viz_width) {
    //       drawCircle(yz_image, viz_width, y, z, 1, {0, 255, 0});  // Green for points
    //     }
    //     if (x >= 0 && x < viz_width && z >= 0 && z < viz_height) {
    //       drawCircle(xz_image, viz_width, x, z, 1, {0, 255, 0});  // Green for points
    //     }
    //   }

    //   // Draw origin and [1,1,1] in blue
    //   int x_0 = static_cast<int>((0 - min_x) / (max_x - min_x) * viz_width);
    //   int y_0 = static_cast<int>((0 - min_y) / (max_y - min_y) * viz_width);
    //   int z_0 = static_cast<int>((0 - min_z) / (max_z - min_z) * viz_width);
    //   int x_1 = static_cast<int>((1 - min_x) / (max_x - min_x) * viz_width);
    //   int y_1 = static_cast<int>((1 - min_y) / (max_y - min_y) * viz_width);
    //   int z_1 = static_cast<int>((1 - min_z) / (max_z - min_z) * viz_width);

    //   // Only draw points if they are within the image bounds

    //   drawCircle(xy_image, viz_width, x_0, y_0, 3, {0, 0, 255});  // Blue for points
    //   drawCircle(xy_image, viz_width, x_1, y_0, 3, {0, 0, 255});  // Blue for points
    //   drawCircle(xy_image, viz_width, x_0, y_1, 3, {0, 0, 255});  // Blue for points
    //   drawCircle(xy_image, viz_width, x_1, y_1, 3, {0, 0, 255});  // Blue for points
    //   drawCircle(yz_image, viz_width, y_0, z_0, 3, {0, 0, 255});  // Blue for points
    //   drawCircle(yz_image, viz_width, y_1, z_0, 3, {0, 0, 255});  // Blue for points
    //   drawCircle(yz_image, viz_width, y_0, z_1, 3, {0, 0, 255});  // Blue for points
    //   drawCircle(yz_image, viz_width, y_1, z_1, 3, {0, 0, 255});  // Blue for points
    //   drawCircle(xz_image, viz_width, x_0, z_0, 3, {0, 0, 255});  // Blue for points
    //   drawCircle(xz_image, viz_width, x_1, z_0, 3, {0, 0, 255});  // Blue for points
    //   drawCircle(xz_image, viz_width, x_0, z_1, 3, {0, 0, 255});  // Blue for points
    //   drawCircle(xz_image, viz_width, x_1, z_1, 3, {0, 0, 255});  // Blue for points

    //   // Save the X,Z visualization
    //   std::stringstream ss;
    //   ss << std::setw(3) << std::setfill('0') << iteration;
    //   std::string padded_iteration = ss.str();

    //   std::string xz_filename = "visualization/xz_viz_iter_" + padded_iteration + ".ppm";
    //   writePPM(xz_filename, viz_width, viz_height, xz_image);
    //   std::string xy_filename = "visualization/xy_viz_iter_" + padded_iteration + ".ppm";
    //   writePPM(xy_filename, viz_width, viz_height, xy_image);
    //   std::string yz_filename = "visualization/yz_viz_iter_" + padded_iteration + ".ppm";
    //   writePPM(yz_filename, viz_width, viz_height, yz_image);
    // }
  }

  const std::unordered_map<uint64_t, Eigen::Vector3d>* angleaxis_rig_from_world_;
  const std::unordered_map<uint64_t, Eigen::Vector3d>* translation_rig_from_world_;
  const std::unordered_map<uint64_t, Eigen::Vector3d>* points_;
  const double* fx_;
  const double* fy_;
  const double* cx_;
  const double* cy_;
  const std::array<double, 6>* radial_distortion_;
  const std::array<double, 2>* tangential_distortion_;
  BundleAdjustmentProblemSummary& summary_;
  const BundleAdjustmentProblemOptions& options_;
  const BundleAdjustmentProblem& problem_;
  const double max_reprojection_error_;
  const bool visualize_frames_;
};

/**
 * @brief Adds residual blocks to the Ceres problem for bundle adjustment.
 *
 * @param problem The bundle adjustment problem containing the observations,
 * poses, and points.
 * @param pose_angleaxis A map from frame id to the angle-axis representation of
 * the pose.
 * @param pose_translations A map from frame id to the translation vector of the
 * pose.
 * @param points A map from point id to the 3D coordinates of the points.
 * @param pinhole_cost_functions A vector to store the created pinhole cost functions.
 * @param rational_polynomial_cost_functions A vector to store the created rational polynomial cost functions.
 * @param ceres_problem The Ceres problem to which the residual blocks will be
 * added.
 * @param loss_function The loss function to be used for the residual blocks.
 * @param fx The focal length x parameter.
 * @param fy The focal length y parameter.
 * @param cx The principal point x parameter.
 * @param cy The principal point y parameter.
 * @param radial_distortion Storage for radial distortion parameters.
 * @param tangential_distortion Storage for tangential distortion parameters.
 * @param options The options for the refinement.
 */
void AddResidualBlocksToCeresProblem(
    refinement::BundleAdjustmentProblem& problem,
    std::unordered_map<uint64_t, Eigen::Vector3d>& angleaxis_rig_from_world,
    std::unordered_map<uint64_t, Eigen::Vector3d>& translation_rig_from_world,
    std::unordered_map<uint64_t, Eigen::Vector3d>& points, std::vector<Eigen::Matrix4d>& camera_from_rigs,
    std::vector<std::unique_ptr<ceres::AutoDiffCostFunction<pinhole::ReprojectionError, 2, 3, 3, 3, 4>>>&
        pinhole_cost_functions,
    std::vector<std::unique_ptr<ceres::AutoDiffCostFunction<rational_polynomial::ReprojectionError, 2, 3, 3, 3, 4, 8>>>&
        rational_polynomial_cost_functions,
    ceres::Problem& ceres_problem, ceres::LossFunction* loss_function, std::vector<std::vector<double>>& intrinsics,
    std::vector<std::vector<double>>& distortion_parameters, const BundleAdjustmentProblemOptions& options) {
  // Iterate over each frame in the problem's observations
  for (auto& frame : problem.observations) {
    auto frame_id = frame.first;
    auto observations = frame.second;

    Eigen::Matrix4f rig_from_world = poseToMatrix(invertPose(problem.world_from_rigs.at(frame_id)));

    // Convert rotation matrix to AngleAxisd, then to Vector3d (angle * axis)
    Eigen::AngleAxisd angle_axis(rig_from_world.block<3, 3>(0, 0).cast<double>());
    angleaxis_rig_from_world[frame_id] = angle_axis.angle() * angle_axis.axis();

    // Convert the translation vector of the current frame to double precision
    translation_rig_from_world[frame_id] = rig_from_world.block<3, 1>(0, 3).cast<double>();

    // Iterate over each observation in the current frame
    for (const auto& observation : observations) {
      // If the 3D point corresponding to the observation is not already in the
      // map, add it
      if (points.count(observation.id) == 0) {
        Eigen::Map<Eigen::Vector3f> point(problem.points_in_world.at(observation.id).coords.data());
        points[observation.id] = point.cast<double>();
      }

      // Get a pointer to the 3D point
      double* ptr_point = points[observation.id].data();

      // Check if camera index is valid
      if (observation.camera_index >= problem.rig.cameras.size()) {
        std::stringstream ss;
        ss << "Camera index is out of bounds: " << observation.camera_index << " for point id: " << observation.id
           << " for frame id: " << frame_id;
        throw std::runtime_error(ss.str());
      }

      auto camera = problem.rig.cameras[observation.camera_index];
      if (camera.distortion.model == Distortion::Model::Polynomial) {
        // Create a new rational polynomial cost function for the current observation
        rational_polynomial_cost_functions.emplace_back(
            std::make_unique<ceres::AutoDiffCostFunction<rational_polynomial::ReprojectionError, 2, 3, 3, 3, 4, 8>>(
                new rational_polynomial::ReprojectionError(&camera_from_rigs[observation.camera_index],
                                                           double(observation.u), double(observation.v),
                                                           options.symmetric_focal_length)));

        // Add a residual block to the Ceres problem using the rational polynomial cost function
        ceres_problem.AddResidualBlock(
            rational_polynomial_cost_functions.back().get(), loss_function, angleaxis_rig_from_world[frame_id].data(),
            translation_rig_from_world[frame_id].data(), ptr_point, &intrinsics[observation.camera_index][0],
            &distortion_parameters[observation.camera_index][0]);
      } else {
        // Create a new pinhole cost function for the current observation
        pinhole_cost_functions.emplace_back(
            std::make_unique<ceres::AutoDiffCostFunction<pinhole::ReprojectionError, 2, 3, 3, 3, 4>>(
                new pinhole::ReprojectionError(&camera_from_rigs[observation.camera_index], double(observation.u),
                                               double(observation.v), options.symmetric_focal_length)));

        // Add a residual block to the Ceres problem using the pinhole cost function
        ceres_problem.AddResidualBlock(
            pinhole_cost_functions.back().get(), loss_function, angleaxis_rig_from_world[frame_id].data(),
            translation_rig_from_world[frame_id].data(), ptr_point, &intrinsics[observation.camera_index][0]);
      }
    }
  }
}

/**
 * @brief Updates the problem with the new poses and points.
 *
 * @param problem The problem to update.
 * @param angleaxis_rig_from_world A map from frame id to the angle-axis representation of
 * the pose.
 * @param translation_rig_from_world A map from frame id to the translation vector of the
 * pose.
 * @param points A map from point id to the 3D coordinates of the points.
 */
void UpdateProblemWithNewPosesAndPoints(refinement::BundleAdjustmentProblem& problem,
                                        const std::unordered_map<uint64_t, Eigen::Vector3d>& angleaxis_rig_from_world,
                                        const std::unordered_map<uint64_t, Eigen::Vector3d>& translation_rig_from_world,
                                        const std::unordered_map<uint64_t, Eigen::Vector3d>& points) {
  // Update the problem with the new poses
  for (auto& frame : problem.observations) {
    auto frame_id = frame.first;
    auto observations = frame.second;

    // We optimize rig_from_world but the problem is stored as world_from_rig so we need to invert the pose.
    // Invert orientation
    Eigen::Vector3f angleaxis_rig_from_world_float = angleaxis_rig_from_world.at(frame_id).cast<float>();
    Eigen::Map<Eigen::Quaternionf> world_from_rigs_quaternion_map(problem.world_from_rigs.at(frame_id).rotation.data());
    world_from_rigs_quaternion_map =
        Eigen::AngleAxisf(-angleaxis_rig_from_world_float.norm(), angleaxis_rig_from_world_float.normalized());

    // Invert translation
    Eigen::Map<Eigen::Vector3f> translation_map_world_from_rig(problem.world_from_rigs.at(frame_id).translation.data());
    Eigen::Matrix3f R_world_from_rig = world_from_rigs_quaternion_map.toRotationMatrix();
    translation_map_world_from_rig = -(R_world_from_rig * translation_rig_from_world.at(frame_id).cast<float>()).eval();
  }

  // Update the problem with the new points
  for (auto& point : points) {
    Eigen::Map<Eigen::Vector3f> point_map(problem.points_in_world.at(point.first).coords.data());
    point_map = point.second.cast<float>();
  }
}

/**
 * @brief Uses Ceres to set up the bundle adjustment problem and refine it.
 *
 * @param problem The problem to refine.
 * @param options The options for the refinement.
 * @param summary The summary of the refinement.
 * @param refined_problem The refined problem.

 * @return The refined problem.
 */
BundleAdjustmentProblemSummary refine(const BundleAdjustmentProblem& problem,
                                      const BundleAdjustmentProblemOptions& options,
                                      BundleAdjustmentProblem& refined_problem) {
  if (problem.rig.imus.size() > 0) {
    throw std::runtime_error("Refinement currently does not support IMUs");
  }
  if (problem.rig.cameras[0].distortion.model == Distortion::Model::Polynomial) {
    if (problem.rig.cameras[0].distortion.parameters.size() < 8) {
      throw std::runtime_error(
          "Refinement currently only supports 8 distortion parameters for rational polynomial distortion ordered as "
          "[k1, k2, p1, p2, k3, k4, k5, k6]");
    }
  }

  refined_problem = problem;

  // Since Ceres takes ownership of the cost functions, we need to set the
  // ownership to DO_NOT_TAKE_OWNERSHIP as we're already managing the memory.
  ceres::Problem::Options problem_options;
  problem_options.cost_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  problem_options.manifold_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  ceres::Problem ceres_problem(problem_options);

  // Vector of all cost functions
  std::vector<std::unique_ptr<ceres::AutoDiffCostFunction<pinhole::ReprojectionError, 2, 3, 3, 3, 4>>>
      pinhole_cost_functions;
  std::vector<std::unique_ptr<ceres::AutoDiffCostFunction<rational_polynomial::ReprojectionError, 2, 3, 3, 3, 4, 8>>>
      rational_polynomial_cost_functions;

  // These variables are used to store the poses and points in the ceres
  // problem. We use double precision as Ceres uses double precision internally.
  std::unordered_map<uint64_t, Eigen::Vector3d> angleaxis_rig_from_world;
  std::unordered_map<uint64_t, Eigen::Vector3d> translation_rig_from_world;
  std::unordered_map<uint64_t, Eigen::Vector3d> points;
  std::vector<Eigen::Matrix4d> camera_from_rigs;
  for (const auto& camera : problem.rig.cameras) {
    camera_from_rigs.push_back(poseToMatrix(invertPose(camera.rig_from_camera)).cast<double>());
  }

  // Double version of intrinsics for Ceres
  std::vector<std::vector<double>> intrinsics;
  std::vector<std::vector<double>> distortion_parameters;

  for (const auto& camera : problem.rig.cameras) {
    intrinsics.push_back(
        std::vector<double>{camera.focal[0], camera.focal[1], camera.principal[0], camera.principal[1]});
    if (camera.distortion.model == Distortion::Model::Polynomial) {
      distortion_parameters.push_back(
          std::vector<double>(camera.distortion.parameters.begin(), camera.distortion.parameters.end()));
    }
  }

  ceres::LossFunction* loss_function_ptr = nullptr;
  BinaryLoss binary_loss(options.max_reprojection_error * options.max_reprojection_error);
  ceres::CauchyLoss cauchy_loss(options.cauchy_loss_scale);
  ceres::ComposedLoss loss_function(&cauchy_loss, ceres::DO_NOT_TAKE_OWNERSHIP, &binary_loss,
                                    ceres::DO_NOT_TAKE_OWNERSHIP);
  if (options.use_loss_function) {
    loss_function_ptr = &loss_function;
  }

  // Add residual blocks for all observations
  AddResidualBlocksToCeresProblem(refined_problem, angleaxis_rig_from_world, translation_rig_from_world, points,
                                  camera_from_rigs, pinhole_cost_functions, rational_polynomial_cost_functions,
                                  ceres_problem, loss_function_ptr, intrinsics, distortion_parameters, options);

  // Add fixed frames and fixed points
  for (auto& frame_id : problem.fixed_frames) {
    ceres_problem.SetParameterBlockConstant(angleaxis_rig_from_world[frame_id].data());
    ceres_problem.SetParameterBlockConstant(translation_rig_from_world[frame_id].data());
  }
  for (auto& point_id : problem.fixed_points) {
    ceres_problem.SetParameterBlockConstant(points[point_id].data());
  }

  // Add fixed intrinsics
  if (!options.estimate_intrinsics) {
    for (const auto& intrinsic : intrinsics) {
      ceres_problem.SetParameterBlockConstant(intrinsic.data());
    }
    for (const auto& distortion_parameter : distortion_parameters) {
      ceres_problem.SetParameterBlockConstant(distortion_parameter.data());
    }
  }

  // Set solver options
  ceres::Solver::Options ceres_options;
  ceres_options = options.ceres_options;
  ceres_options.minimizer_progress_to_stdout = options.verbose;

  // Initialize the summary
  BundleAdjustmentProblemSummary summary;

  // Add the iteration state export callback if requested
  std::unique_ptr<StateExportCallback> state_callback;
  if (options.export_iteration_state) {
    // state_callback = std::make_unique<StateExportCallback>(
    //     &angleaxis_rig_from_world, &translation_rig_from_world, &points, &intrinsics, &distortion_parameters,
    //     summary, options, refined_problem, options.max_reprojection_error);
    // ceres_options.callbacks.push_back(state_callback.get());  // ceres_options.max_num_iterations = 1;
    throw std::runtime_error("StateExportCallback not implemented");

    ceres_options.update_state_every_iteration = true;
  }

  // Solve the problem
  ceres::Solver::Summary ceres_summary;
  ceres::Solve(ceres_options, &ceres_problem, &ceres_summary);

  summary.ceres_summary = ceres_summary;

  // Print the summary if verbose
  if (options.verbose) {
    std::cout << ceres_summary.FullReport() << std::endl;
  }

  // Call the function to update the problem
  UpdateProblemWithNewPosesAndPoints(refined_problem, angleaxis_rig_from_world, translation_rig_from_world, points);

  // Update the intrinsics
  for (size_t i = 0; i < intrinsics.size(); i++) {
    refined_problem.rig.cameras[i].focal[0] = intrinsics[i][0];
    if (options.symmetric_focal_length) {
      refined_problem.rig.cameras[i].focal[1] = intrinsics[i][0];
    } else {
      refined_problem.rig.cameras[i].focal[1] = intrinsics[i][1];
    }
    refined_problem.rig.cameras[i].principal[0] = intrinsics[i][2];
    refined_problem.rig.cameras[i].principal[1] = intrinsics[i][3];
    if (problem.rig.cameras[i].distortion.model == Distortion::Model::Polynomial) {
      refined_problem.rig.cameras[i].distortion.parameters =
          std::vector<float>(distortion_parameters[i].begin(), distortion_parameters[i].end());
    }
  }

  return summary;
}

}  // namespace cuvslam::refinement
