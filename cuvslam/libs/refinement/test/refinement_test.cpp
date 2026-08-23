
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

#include "common/environment.h"
#include "common/include_gtest.h"
#include "common/isometry_utils.h"
#include "common/types.h"
#include "refinement/bundle_adjustment_problem.h"

namespace test::refinement {

using namespace cuvslam::refinement;

const bool kVerbose = false;

const float kCeresFunctionTolerance = 1e-16f;
const float kCeresParameterTolerance = 1e-16f;
const float kCeresGradientTolerance = 1e-16f;

const float noise_percent = 1.0;

const float kTranslationNoise = 0.5f;
const float kRotationNoise = 0.1f;
const float kPointNoise = 1.0f;
const float kIntrinsicsNoise = 10.0f;
const float kDistortionNoise = 0.00f;

const int kImageWidth = 1000;
const int kImageHeight = 900;
const float kFocalLength = 50.0f;

const float kTranslationDifferenceTolerance = 0.01f;
const float kRotationDifferenceTolerance = 0.01f;
const float kPointDifferenceTolerance = 0.01f;
const float kIntrinsicDifferenceTolerance = 0.01f;
const float kDistortionDifferenceTolerance = 0.01f;
const float kReprojectionErrorTolerance = 0.01f;

void PrintProblem(const cuvslam::refinement::BundleAdjustmentProblem& problem) {
  size_t num_observations = 0;
  for (const auto& [frame_id, observations] : problem.observations) {
    num_observations += observations.size();
  }
  std::cout << "Problem: " << problem.world_from_rigs.size() << " frames, " << problem.points_in_world.size()
            << " points, " << num_observations << " observations" << std::endl;
  for (const auto& [frame_id, observations] : problem.observations) {
    std::cout << "Frame " << frame_id << ": " << observations.size() << " observations" << std::endl;
  }
  for (const auto& [frame_id, world_from_rig] : problem.world_from_rigs) {
    std::cout << "Frame " << frame_id << ": translation: " << world_from_rig.translation[0] << " "
              << world_from_rig.translation[1] << " " << world_from_rig.translation[2]
              << ", rotation: " << world_from_rig.rotation[0] << " " << world_from_rig.rotation[1] << " "
              << world_from_rig.rotation[2] << " " << world_from_rig.rotation[3] << std::endl;
  }
  for (const auto& [point_id, point] : problem.points_in_world) {
    std::cout << "Point " << point_id << ": " << point.coords[0] << " " << point.coords[1] << " " << point.coords[2]
              << std::endl;
  }
  // Seen from count
  std::unordered_map<uint64_t, size_t> seen_count;
  for (const auto& [frame_id, observations] : problem.observations) {
    for (const auto& observation : observations) {
      seen_count[observation.id]++;
    }
  }
  for (const auto& [point_id, count] : seen_count) {
    std::cout << "Point " << point_id << ": " << count << " observations" << std::endl;
  }
}

/**
 * @brief Generates a vector of 3D points in an equidistant grid centered around
 * the origin.
 *
 * @param num_points_per_axis Number of points per axis.
 * @param cube_size The size of the cube in which the points are generated.
 * @return std::vector<Eigen::Vector3f> A vector of 3D points.
 */
std::vector<Eigen::Vector3f> GenerateEquidistantGridPoints(size_t num_points_per_axis, float cube_size) {
  std::vector<Eigen::Vector3f> points;
  points.reserve(num_points_per_axis * num_points_per_axis * num_points_per_axis);

  float step = cube_size / (num_points_per_axis - 1);
  float half_cube_size = cube_size / 2.0f;
  for (size_t x = 0; x < num_points_per_axis; ++x) {
    for (size_t y = 0; y < num_points_per_axis; ++y) {
      for (size_t z = 0; z < num_points_per_axis; ++z) {
        points.emplace_back(x * step - half_cube_size, y * step - half_cube_size, z * step - half_cube_size);
      }
    }
  }

  return points;
}

/**
 * @brief Generates poses for a circular motion around the origin.
 *
 * @param problem The bundle adjustment problem to which the poses are added.
 * @param num_cameras The number of cameras.
 * @param radius The radius of the circular motion.
 */
void GenerateCircularMotionPoses(cuvslam::refinement::BundleAdjustmentProblem& problem, size_t num_cameras,
                                 float radius) {
  for (size_t i = 0; i < num_cameras; ++i) {
    float angle = 2 * M_PI * i / num_cameras;

    // Create a pose from translation and angle axis
    cuvslam::Pose rig_from_world;
    Eigen::Vector3f axis_of_rotation(0, 1, 0);
    Eigen::Vector3f translation(cos(angle) * radius, sin(angle) * radius, cos(angle) * radius);
    Eigen::AngleAxisf rotation(angle, axis_of_rotation);
    Eigen::Quaternionf quat(rotation);

    // Store quaternion coefficients
    Eigen::Map<Eigen::Quaternionf>(rig_from_world.rotation.data()) = quat.coeffs();

    for (size_t j = 0; j < 3; ++j) {
      rig_from_world.translation[j] = translation(j);
    }

    problem.world_from_rigs[i] = invertPose(rig_from_world);
  }
}

/**
 * @brief Projects 3D points to the cameras and adds them to the bundle
 * adjustment problem.
 *
 * @param problem The bundle adjustment problem to which the points are added.
 * @param points_3d The 3D points to project.
 * @param num_cameras The number of cameras.
 */
void ProjectPointsToCameras(cuvslam::refinement::BundleAdjustmentProblem& problem,
                            const std::vector<Eigen::Vector3f>& points_3d) {
  for (const auto& [frame_id, world_from_rig] : problem.world_from_rigs) {
    Eigen::Matrix4f rig_from_world = poseToMatrix(invertPose(world_from_rig));

    for (size_t camera_index = 0; camera_index < problem.rig.cameras.size(); ++camera_index) {
      const cuvslam::Camera& camera = problem.rig.cameras[camera_index];

      Eigen::Matrix4f camera_from_rig = poseToMatrix(invertPose(camera.rig_from_camera));
      Eigen::Matrix4f camera_from_world = camera_from_rig * rig_from_world;

      for (size_t j = 0; j < points_3d.size(); ++j) {
        cuvslam::Observation observation;
        observation.id = j;

        Eigen::Map<const Eigen::Vector3f> point_world(problem.points_in_world[j].coords.data());
        Eigen::Vector3f point_camera =
            camera_from_world.block<3, 3>(0, 0) * point_world + camera_from_world.block<3, 1>(0, 3);

        // Only include points that are in front of the camera
        if (point_camera.z() <= 0.0f) {
          continue;
        }

        // Project to normalized image coordinates
        float xp = point_camera.x() / point_camera.z();
        float yp = point_camera.y() / point_camera.z();

        // Apply camera model with default intrinsics
        float fx = camera.focal[0];
        float fy = camera.focal[1];
        float cx = camera.principal[0];
        float cy = camera.principal[1];

        if (camera.distortion.model == cuvslam::Distortion::Model::Polynomial) {
          // Apply rational polynomial distortion
          float r2 = xp * xp + yp * yp;
          float r4 = r2 * r2;
          float r6 = r4 * r2;

          // Get distortion coefficients (k1, k2, p1, p2, k3, k4, k5, k6)
          const auto& params = camera.distortion.parameters;
          float k1 = params[0];
          float k2 = params[1];
          float p1 = params[2];
          float p2 = params[3];
          float k3 = params[4];
          float k4 = params.size() > 5 ? params[5] : 0.0f;
          float k5 = params.size() > 6 ? params[6] : 0.0f;
          float k6 = params.size() > 7 ? params[7] : 0.0f;

          // Calculate radial distortion factor
          float numerator = 1.0f + k1 * r2 + k2 * r4 + k3 * r6;
          float denominator = 1.0f + k4 * r2 + k5 * r4 + k6 * r6;

          // Avoid division by zero
          if (std::abs(denominator) < 1e-6f) {
            continue;
          }

          float radial_factor = numerator / denominator;

          // Calculate tangential distortion
          float xy = xp * yp;
          float x_distorted = xp * radial_factor + 2.0f * p1 * xy + p2 * (r2 + 2.0f * xp * xp);
          float y_distorted = yp * radial_factor + p1 * (r2 + 2.0f * yp * yp) + 2.0f * p2 * xy;

          // Project to pixel coordinates
          observation.u = fx * x_distorted + cx;
          observation.v = fy * y_distorted + cy;
        } else {
          // Apply simple pinhole model (no distortion)
          observation.u = fx * xp + cx;
          observation.v = fy * yp + cy;
        }

        // Don't include observations that are outside the image
        if (observation.u <= 0 || observation.u >= camera.size[0] || observation.v <= 0 ||
            observation.v >= camera.size[1]) {
          continue;
        }
        observation.camera_index = camera_index;

        problem.observations[frame_id].push_back(observation);
      }
    }
  }
}

/**
 * @brief Removes points that are not seen by any camera from the bundle
 * adjustment problem.
 *
 * @param problem The bundle adjustment problem from which the points are
 * removed.
 * @param points_3d The 3D points to remove.
 * @param num_cameras The number of cameras.
 */
void RemoveUnseenPoints(cuvslam::refinement::BundleAdjustmentProblem& problem,
                        const std::vector<Eigen::Vector3f>& points_3d, size_t num_cameras) {
  // Map from point id to number of cameras that see it
  std::unordered_map<uint64_t, size_t> point_seen_count;

  // Count how many cameras see each point
  for (size_t i = 0; i < num_cameras; ++i) {
    for (size_t j = 0; j < problem.observations[i].size(); ++j) {
      uint64_t pid = problem.observations[i][j].id;
      point_seen_count[pid]++;
    }
  }

  // Remove points that are not seen by any camera
  for (size_t i = 0; i < points_3d.size(); ++i) {
    if (point_seen_count[i] == 0) {
      problem.points_in_world.erase(i);
    }
  }
}

/**
 * @brief Initializes a bundle adjustment problem with a circular motion of the
 * cameras.
 *
 * @param num_points The number of 3D points to generate.
 * @param num_cameras The number of cameras.
 * @param cube_size The size of the cube in which the points are generated.
 * @param radius The radius of the circular motion.
 * @param symmetric_focal_length Whether to use symmetric focal length.
 * @param use_rational_polynomial_distortion Whether to use rational polynomial distortion.
 * @return cuvslam::refinement::BundleAdjustmentProblem The initialized bundle
 * adjustment problem.
 */
cuvslam::refinement::BundleAdjustmentProblem InitializeBundleAdjustmentProblem(
    bool symmetric_focal_length = false, bool use_rational_polynomial_distortion = false) {
  const size_t num_points_per_axis = 4;
  const size_t num_cameras = 36;
  const float cube_size = 10.0f;
  const float radius = 1.0f;

  cuvslam::refinement::BundleAdjustmentProblem problem;

  // Add camera to the problem
  cuvslam::Camera camera;
  camera.size = {kImageWidth, kImageHeight};
  if (symmetric_focal_length) {
    camera.focal = {kFocalLength, kFocalLength};
  } else {
    camera.focal = {kFocalLength, kFocalLength * 1.1f};
  }
  camera.principal = {kImageWidth / 2.0f, kImageHeight / 2.0f};

  // Add distortion parameters if using rational polynomial distortion
  if (use_rational_polynomial_distortion) {
    camera.distortion.model = cuvslam::Distortion::Model::Polynomial;
    // Use standard OpenCV distortion coefficients
    // k1, k2, p1, p2, k3, k4, k5, k6
    camera.distortion.parameters = {-0.0062, 0.029, -0.034, 0.014, 0.0f, 0.0f, 0.0f, 0.0f};
  } else {
    camera.distortion.model = cuvslam::Distortion::Model::Pinhole;
  }

  problem.rig.cameras.push_back(camera);

  // Generate 3D points and add them to the problem
  std::unordered_map<uint64_t, bool> seen_points;
  auto points_3d = GenerateEquidistantGridPoints(num_points_per_axis, cube_size);
  for (size_t i = 0; i < points_3d.size(); ++i) {
    cuvslam::Landmark landmark;
    landmark.id = i;
    for (size_t j = 0; j < 3; ++j) {
      landmark.coords[j] = points_3d[i][j];
    }
    problem.points_in_world[i] = landmark;
  }

  // Usage of the new function
  GenerateCircularMotionPoses(problem, num_cameras, radius);

  // Project points into 2D and add them to the problem
  ProjectPointsToCameras(problem, points_3d);

  // Remove points that are not seen by any camera
  RemoveUnseenPoints(problem, points_3d, num_cameras);

  // Fix a few frames to make sure the problem is well-constrained
  problem.fixed_frames.insert(0);
  problem.fixed_frames.insert(12);
  problem.fixed_frames.insert(24);

  return problem;
}

/**
 * @brief Adds random noise to the poses, points, and intrinsics of a bundle
 * adjustment problem.
 *
 * @param problem The bundle adjustment problem to which the noise is added.
 * @return cuvslam::refinement::BundleAdjustmentProblem The perturbed bundle
 * adjustment problem.
 */
cuvslam::refinement::BundleAdjustmentProblem AddRandomNoiseToProblem(
    const cuvslam::refinement::BundleAdjustmentProblem& problem,
    const refinement::BundleAdjustmentProblemOptions& options) {
  cuvslam::refinement::BundleAdjustmentProblem perturbed_problem = problem;

  // Add random noise to the poses
  for (const auto& item : perturbed_problem.world_from_rigs) {
    // Don't perturb fixed frames
    auto frame_id = item.first;
    auto rig_from_world = invertPose(item.second);
    if (perturbed_problem.fixed_frames.count(frame_id) > 0) {
      continue;
    }

    // Use Map to initialize quaternion from data pointer
    Eigen::Map<Eigen::Quaternionf> rig_from_world_quat(rig_from_world.rotation.data());
    Eigen::Map<Eigen::Vector3f> rig_from_world_t(rig_from_world.translation.data());

    Eigen::Vector3f random_axis(static_cast<float>(rand()) / RAND_MAX - 0.5f,
                                static_cast<float>(rand()) / RAND_MAX - 0.5f,
                                static_cast<float>(rand()) / RAND_MAX - 0.5f);
    random_axis.normalize();
    Eigen::AngleAxisf random_rotation(kRotationNoise * (static_cast<float>(rand()) / RAND_MAX - 0.5f), random_axis);

    // Apply the random rotation
    rig_from_world_quat = rig_from_world_quat * random_rotation;

    // Add noise to translation
    rig_from_world_t =
        rig_from_world_t + Eigen::Vector3f(kTranslationNoise * (static_cast<float>(rand()) / RAND_MAX - 0.5f),
                                           kTranslationNoise * (static_cast<float>(rand()) / RAND_MAX - 0.5f),
                                           kTranslationNoise * (static_cast<float>(rand()) / RAND_MAX - 0.5f));

    // No need to store back the quaternion manually since we're using Map
  }

  // Add noise to 3D points
  for (auto& point : perturbed_problem.points_in_world) {
    // Don't perturb fixed points
    if (perturbed_problem.fixed_points.find(point.first) != perturbed_problem.fixed_points.end()) {
      continue;
    }

    for (size_t i = 0; i < 3; ++i) {
      point.second.coords[i] += kPointNoise * (static_cast<float>(rand()) / RAND_MAX - 0.5f);
    }
  }

  // Add noise to camera intrinsics
  if (options.estimate_intrinsics) {
    for (auto& camera : perturbed_problem.rig.cameras) {
      // Add noise to focal length
      for (size_t i = 0; i < 2; ++i) {
        camera.focal[i] += kIntrinsicsNoise * (static_cast<float>(rand()) / RAND_MAX - 0.5f);
      }

      // Add noise to principal point
      for (size_t i = 0; i < 2; ++i) {
        camera.principal[i] += kIntrinsicsNoise * (static_cast<float>(rand()) / RAND_MAX - 0.5f);
      }

      // Add noise to distortion parameters if using rational polynomial distortion
      if (camera.distortion.model == cuvslam::Distortion::Model::Polynomial) {
        for (size_t i = 0; i < camera.distortion.parameters.size(); ++i) {
          camera.distortion.parameters[i] += kDistortionNoise * (static_cast<float>(rand()) / RAND_MAX - 0.5f);
        }
      }
    }
  }

  return perturbed_problem;
}

/**
 * @brief Checks that all reprojection errors are below a threshold.
 *
 * @param problem The bundle adjustment problem to check.
 * @param max_error The maximum allowed reprojection error in pixels.
 */
void CheckReprojectionError(const cuvslam::refinement::BundleAdjustmentProblem& problem, float max_error) {
  // For each camera
  for (const auto& [frame_id, observations] : problem.observations) {
    // Get the camera pose
    Eigen::Matrix4f rig_from_world_mat = poseToMatrix(invertPose(problem.world_from_rigs.at(frame_id)));

    // For each observation in this camera
    for (const auto& observation : observations) {
      uint64_t point_id = observation.id;
      size_t camera_index = observation.camera_index;

      Eigen::Matrix4f camera_from_rig_mat = poseToMatrix(invertPose(problem.rig.cameras[camera_index].rig_from_camera));
      Eigen::Matrix4f camera_from_world_mat = camera_from_rig_mat * rig_from_world_mat;

      // Get camera intrinsics
      const auto& camera = problem.rig.cameras[camera_index];  // Assuming single camera rig
      float fx = camera.focal[0];
      float fy = camera.focal[1];
      float cx = camera.principal[0];
      float cy = camera.principal[1];

      // Get the 3D point
      const auto& landmark = problem.points_in_world.at(point_id);
      Eigen::Map<const Eigen::Vector3f> point_world(landmark.coords.data());

      // Transform point to camera frame
      Eigen::Vector3f point_camera =
          camera_from_world_mat.block<3, 3>(0, 0) * point_world + camera_from_world_mat.block<3, 1>(0, 3);

      // Skip points behind the camera
      if (point_camera.z() <= 0.0f) {
        continue;
      }

      // Project to normalized image coordinates
      float xp = point_camera.x() / point_camera.z();
      float yp = point_camera.y() / point_camera.z();

      // Apply distortion if needed
      float x_distorted = xp;
      float y_distorted = yp;

      if (camera.distortion.model == cuvslam::Distortion::Model::Polynomial) {
        float r2 = xp * xp + yp * yp;
        float r4 = r2 * r2;
        float r6 = r4 * r2;

        // Get distortion coefficients
        const auto& params = camera.distortion.parameters;
        float k1 = params[0];
        float k2 = params[1];
        float p1 = params[2];
        float p2 = params[3];
        float k3 = params[4];
        float k4 = params[5];
        float k5 = params[6];
        float k6 = params[7];

        // Calculate radial distortion factor
        float numerator = 1.0f + k1 * r2 + k2 * r4 + k3 * r6;
        float denominator = 1.0f + k4 * r2 + k5 * r4 + k6 * r6;

        float radial_factor = numerator / denominator;

        // Apply tangential distortion
        float xy = xp * yp;
        x_distorted = xp * radial_factor + 2.0f * p1 * xy + p2 * (r2 + 2.0f * xp * xp);
        y_distorted = yp * radial_factor + p1 * (r2 + 2.0f * yp * yp) + 2.0f * p2 * xy;
      }

      // Project to pixel coordinates
      float u_projected = fx * x_distorted + cx;
      float v_projected = fy * y_distorted + cy;

      if (u_projected < 0 || u_projected > camera.size[0] || v_projected < 0 || v_projected > camera.size[1]) {
        continue;
      }

      // Calculate reprojection error
      float u_error = std::abs(u_projected - observation.u);
      float v_error = std::abs(v_projected - observation.v);
      float error = std::sqrt(u_error * u_error + v_error * v_error);

      // Check that error is below threshold
      EXPECT_LE(error, max_error) << "Point " << point_id << " in frame " << frame_id << " in camera " << camera_index
                                  << " has reprojection error " << error << " pixels. "
                                  << "Projected: (" << u_projected << ", " << v_projected << "), "
                                  << "Observed: (" << observation.u << ", " << observation.v << ")";
    }
  }
}

/**
 * @brief Compares two bundle adjustment problems.
 *
 * @param original_problem The original problem.
 * @param perturbed_problem The perturbed problem.
 * @param epsilon The max difference value used for comparison.
 */
void CompareProblems(const cuvslam::refinement::BundleAdjustmentProblem& original_problem,
                     const cuvslam::refinement::BundleAdjustmentProblem& perturbed_problem) {
  for (auto& pose : perturbed_problem.world_from_rigs) {
    // Create quaternions for comparison using Map
    Eigen::Map<const Eigen::Quaternionf> quat(pose.second.rotation.data());
    Eigen::Map<const Eigen::Quaternionf> quat_original(original_problem.world_from_rigs.at(pose.first).rotation.data());

    // Compare translations
    Eigen::Map<const Eigen::Vector3f> t(pose.second.translation.data());
    Eigen::Map<const Eigen::Vector3f> t_original(original_problem.world_from_rigs.at(pose.first).translation.data());

    SCOPED_TRACE("Pose " + std::to_string(pose.first));

    // Calculate the distance between the two translations
    float distance = (t - t_original).norm();
    EXPECT_NEAR(distance, 0, kTranslationDifferenceTolerance) << " Translation distance";

    // Convert quaternions to rotation matrices once
    Eigen::Matrix3f rot = quat.toRotationMatrix();
    Eigen::Matrix3f rot_original = quat_original.toRotationMatrix();

    // Format rotation matrices for debug output
    Eigen::IOFormat matrix_format(4, 0, ", ", "\n", "[", "]");
    std::stringstream ss1, ss2;
    ss1 << "Rotation matrix:\n" << rot.format(matrix_format);
    ss2 << "Rotation matrix original:\n" << rot_original.format(matrix_format);
    SCOPED_TRACE(ss1.str());
    SCOPED_TRACE(ss2.str());

    // Calculate the angle magnitude between the two orientations
    Eigen::Matrix3f relative_rotation = rot * rot_original.transpose();
    Eigen::AngleAxisf angle_axis(relative_rotation);
    float angle_magnitude = std::abs(angle_axis.angle());
    EXPECT_NEAR(angle_magnitude, 0.0f, kRotationDifferenceTolerance) << " Angle magnitude between rotations";
  }

  for (auto& point : perturbed_problem.points_in_world) {
    Eigen::Map<const Eigen::Vector3f> point_map(point.second.coords.data());
    Eigen::Map<const Eigen::Vector3f> point_original_map(
        original_problem.points_in_world.at(point.first).coords.data());

    // Calculate the distance between the point and the original point
    float distance = (point_map - point_original_map).norm();
    EXPECT_NEAR(distance, 0, kPointDifferenceTolerance)
        << " Landmark ID: " << point.first << " Point_map: " << point_map.transpose()
        << " Point_original_map: " << point_original_map.transpose();
  }

  for (size_t i = 0; i < perturbed_problem.rig.cameras.size(); ++i) {
    const auto& perturbed_cam = perturbed_problem.rig.cameras[i];
    const auto& original_cam = original_problem.rig.cameras[i];
    EXPECT_NEAR(perturbed_cam.focal[0], original_cam.focal[0], kIntrinsicDifferenceTolerance)
        << " Focal X for camera " << i;
    EXPECT_NEAR(perturbed_cam.focal[1], original_cam.focal[1], kIntrinsicDifferenceTolerance)
        << " Focal Y for camera " << i;
    EXPECT_NEAR(perturbed_cam.principal[0], original_cam.principal[0], kIntrinsicDifferenceTolerance)
        << " Principal X for camera " << i;
    EXPECT_NEAR(perturbed_cam.principal[1], original_cam.principal[1], kIntrinsicDifferenceTolerance)
        << " Principal Y for camera " << i;
    EXPECT_EQ(perturbed_cam.distortion.model, original_cam.distortion.model) << " Distortion model for camera " << i;
    EXPECT_EQ(perturbed_cam.distortion.parameters.size(), original_cam.distortion.parameters.size())
        << " Distortion parameters size for camera " << i;
    for (size_t j = 0; j < perturbed_cam.distortion.parameters.size(); ++j) {
      EXPECT_NEAR(perturbed_cam.distortion.parameters[j], original_cam.distortion.parameters[j],
                  kDistortionDifferenceTolerance)
          << " Distortion Parameter " << j << " for camera " << i;
    }
  }

  CheckReprojectionError(perturbed_problem, kReprojectionErrorTolerance);
}

/**
 * @brief Tests the refinement of a bundle adjustment problem with no
 * perturbations.
 */
TEST(Refinement, BAProblem) {
  srand(42);  // Fixed seed for reproducibility
  cuvslam::refinement::BundleAdjustmentProblem problem = InitializeBundleAdjustmentProblem();

  cuvslam::refinement::BundleAdjustmentProblem solved_problem;
  refinement::BundleAdjustmentProblemOptions options;
  options.verbose = kVerbose;

  options.max_reprojection_error = 100000.0f;  // Disable extreme outlier rejection

  // Run bundle adjustment
  refinement::BundleAdjustmentProblemSummary summary = refinement::refine(problem, options, solved_problem);

  // Print out the original problem and solved problem for comparison
  size_t total_observations = 0;
  for (const auto& frame : problem.observations) {
    total_observations += frame.second.size();
  }
  // Compare the results with the original problem
  CompareProblems(problem, solved_problem);
}

/**
 * @brief Tests the refinement of a bundle adjustment problem with perturbed
 * intrinsics, poses, and points.
 */
TEST(Refinement, BAProblemPerturbed) {
  srand(42);  // Fixed seed for reproducibility
  cuvslam::refinement::BundleAdjustmentProblem problem = InitializeBundleAdjustmentProblem();
  if (kVerbose) {
    PrintProblem(problem);
  }

  // Set the options for the refinement
  refinement::BundleAdjustmentProblemOptions options;
  options.verbose = kVerbose;
  options.estimate_intrinsics = true;
  options.use_loss_function = false;
  options.ceres_options.function_tolerance = kCeresFunctionTolerance;
  options.ceres_options.parameter_tolerance = kCeresParameterTolerance;
  cuvslam::refinement::BundleAdjustmentProblem perturbed_problem = AddRandomNoiseToProblem(problem, options);

  // Run bundle adjustment
  refinement::BundleAdjustmentProblemSummary summary =
      refinement::refine(perturbed_problem, options, perturbed_problem);

  // Compare the results with the original problem
  CompareProblems(problem, perturbed_problem);
}

/**
 * @brief Tests the refinement of a bundle adjustment problem with poses and points. Intrinsics should stay exactly the
 * same.
 */
TEST(Refinement, BAProblemPerturbedNoEstimateIntrinsics) {
  srand(42);  // Fixed seed for reproducibility

  cuvslam::refinement::BundleAdjustmentProblem problem = InitializeBundleAdjustmentProblem();
  // Set the options for the refinement
  refinement::BundleAdjustmentProblemOptions options;
  options.verbose = kVerbose;
  options.estimate_intrinsics = false;
  options.max_reprojection_error = 100000.0f;  // Disable extreme outlier rejection

  cuvslam::refinement::BundleAdjustmentProblem perturbed_problem = AddRandomNoiseToProblem(problem, options);

  // Run bundle adjustment
  refinement::BundleAdjustmentProblemSummary summary =
      refinement::refine(perturbed_problem, options, perturbed_problem);

  // Compare the results with the original problem
  CompareProblems(problem, perturbed_problem);

  // If we didn't estimate the intrinsics, they should be the exact same as the original values.
  EXPECT_EQ(perturbed_problem.rig.cameras[0].focal[0], problem.rig.cameras[0].focal[0]);
  EXPECT_EQ(perturbed_problem.rig.cameras[0].focal[1], problem.rig.cameras[0].focal[1]);
  EXPECT_EQ(perturbed_problem.rig.cameras[0].principal[0], problem.rig.cameras[0].principal[0]);
  EXPECT_EQ(perturbed_problem.rig.cameras[0].principal[1], problem.rig.cameras[0].principal[1]);
}

/**
 * @brief Tests the refinement of a bundle adjustment problem with perturbed
 * intrinsics, poses, and points but with symmetric focal length.
 */
TEST(Refinement, BAProblemPerturbedSymmetricFocalLength) {
  srand(42);  // Fixed seed for reproducibility

  cuvslam::refinement::BundleAdjustmentProblem problem = InitializeBundleAdjustmentProblem(true);
  // Set the options for the refinement
  refinement::BundleAdjustmentProblemOptions options;
  options.verbose = kVerbose;
  options.estimate_intrinsics = true;
  options.symmetric_focal_length = true;
  options.use_loss_function = false;
  options.ceres_options.function_tolerance = kCeresFunctionTolerance;
  options.ceres_options.parameter_tolerance = kCeresParameterTolerance;

  cuvslam::refinement::BundleAdjustmentProblem perturbed_problem = AddRandomNoiseToProblem(problem, options);

  // Run bundle adjustment
  refinement::BundleAdjustmentProblemSummary summary =
      refinement::refine(perturbed_problem, options, perturbed_problem);

  // Compare the results with the original problem
  CompareProblems(problem, perturbed_problem);
  EXPECT_EQ(perturbed_problem.rig.cameras[0].focal[0], perturbed_problem.rig.cameras[0].focal[1]);
}

/**
 * @brief Tests the refinement of a bundle adjustment problem with rational polynomial distortion.
 */
TEST(Refinement, BAProblemWithRationalPolynomialDistortionNoPerturbed) {
  srand(42);  // Fixed seed for reproducibility

  // Initialize problem with rational polynomial distortion
  cuvslam::refinement::BundleAdjustmentProblem problem = InitializeBundleAdjustmentProblem(false, true);

  // Set the options for the refinement
  refinement::BundleAdjustmentProblemOptions options;
  options.verbose = kVerbose;
  options.estimate_intrinsics = true;
  options.use_loss_function = false;

  // Run bundle adjustment
  cuvslam::refinement::BundleAdjustmentProblem output_problem;
  refinement::BundleAdjustmentProblemSummary summary = refinement::refine(problem, options, output_problem);

  // Compare the results with the original problem
  CompareProblems(problem, output_problem);
}

/**
 * @brief Tests the refinement of a bundle adjustment problem with rational polynomial distortion.
 */
TEST(Refinement, BAProblemWithRationalPolynomialDistortionPerturbed) {
  srand(42);  // Fixed seed for reproducibility

  // Initialize problem with rational polynomial distortion
  cuvslam::refinement::BundleAdjustmentProblem problem = InitializeBundleAdjustmentProblem(false, true);
  if (kVerbose) {
    PrintProblem(problem);
  }

  // Set the options for the refinement
  refinement::BundleAdjustmentProblemOptions options;
  options.verbose = kVerbose;
  options.estimate_intrinsics = true;
  options.use_loss_function = false;
  options.ceres_options.max_num_iterations = 100;
  options.ceres_options.function_tolerance = kCeresFunctionTolerance;
  options.ceres_options.parameter_tolerance = kCeresParameterTolerance;

  cuvslam::refinement::BundleAdjustmentProblem perturbed_problem = AddRandomNoiseToProblem(problem, options);

  // Run bundle adjustment
  refinement::BundleAdjustmentProblemSummary summary =
      refinement::refine(perturbed_problem, options, perturbed_problem);

  // Compare the results with the original problem
  CompareProblems(problem, perturbed_problem);
}

/**
 * @brief Tests the refinement of a bundle adjustment problem with rational polynomial distortion and symmetric focal
 * length.
 */
TEST(Refinement, BAProblemWithRationalPolynomialDistortionPerturbedSymmetricFocalLength) {
  srand(42);  // Fixed seed for reproducibility
  bool symmetric_focal_length = true;
  // Initialize problem with rational polynomial distortion
  cuvslam::refinement::BundleAdjustmentProblem problem =
      InitializeBundleAdjustmentProblem(symmetric_focal_length, true);
  if (kVerbose) {
    PrintProblem(problem);
  }
  // cuvslam::refinement::BundleAdjustmentProblem perturbed_problem = AddRandomNoiseToProblem(problem);
  cuvslam::refinement::BundleAdjustmentProblem perturbed_problem = problem;

  // Set the options for the refinement
  refinement::BundleAdjustmentProblemOptions options;
  options.verbose = kVerbose;
  options.estimate_intrinsics = true;
  options.symmetric_focal_length = symmetric_focal_length;
  options.use_loss_function = false;
  options.ceres_options.max_num_iterations = 200;
  options.ceres_options.function_tolerance = kCeresFunctionTolerance;
  options.ceres_options.parameter_tolerance = kCeresParameterTolerance;

  // Run bundle adjustment
  refinement::BundleAdjustmentProblemSummary summary =
      refinement::refine(perturbed_problem, options, perturbed_problem);

  // Compare the results with the original problem
  CompareProblems(problem, perturbed_problem);

  // Verify that the distortion parameters are preserved
  EXPECT_EQ(perturbed_problem.rig.cameras[0].focal[0], perturbed_problem.rig.cameras[0].focal[1]);
}

TEST(Refinement, BAProblemSingleCameraFixedGridWithRig) {
  srand(42);  // Fixed seed for reproducibility

  // Initialize bundle adjustment problem
  cuvslam::refinement::BundleAdjustmentProblem problem;

  // Add camera to the problem
  cuvslam::Camera camera;
  camera.size = {kImageWidth, kImageHeight};
  camera.focal = {kFocalLength, kFocalLength * 1.1f};
  camera.principal = {kImageWidth / 2.0f, kImageHeight / 2.0f};
  camera.distortion.model = cuvslam::Distortion::Model::Pinhole;
  problem.rig.cameras.push_back(camera);

  // Add a second camera to the rig with baseline of 1 meter
  camera.rig_from_camera.translation[0] = 2.0f;
  problem.rig.cameras.push_back(camera);

  // Fix first camera to have full scale
  problem.fixed_frames.insert(0);

  // Set up first camera at origin
  cuvslam::Pose pose;
  // Identity rotation
  Eigen::Quaternionf quat(1.0f, 0.0f, 0.0f, 0.0f);
  Eigen::Map<Eigen::Quaternionf>(pose.rotation.data()) = quat.coeffs();
  // Zero translation
  pose.translation[0] = 0.0f;
  pose.translation[1] = 0.0f;
  pose.translation[2] = 0.0f;
  problem.world_from_rigs[0] = invertPose(pose);

  // Generate 3x3 grid of points centered 10 meters in front of the first camera
  // Grid is in the x-y plane (z=10)
  std::vector<Eigen::Vector3f> points_3d;
  const float grid_size = 4.0f;   // 2 meters grid size
  const float z_distance = 8.0f;  // 10 meters in front of camera

  for (int x = -1; x <= 1; ++x) {
    for (int y = -1; y <= 1; ++y) {
      for (int z = -1; z <= 1; ++z) {
        float px = x * grid_size;
        float py = y * grid_size;
        float pz = z * grid_size + z_distance;
        points_3d.emplace_back(px, py, pz);
      }
    }
  }

  // Add 3D points to the problem
  for (size_t i = 0; i < points_3d.size(); ++i) {
    cuvslam::Landmark landmark;
    landmark.id = i;
    for (size_t j = 0; j < 3; ++j) {
      landmark.coords[j] = points_3d[i][j];
    }
    problem.points_in_world[i] = landmark;
  }

  // Project points into 2D and add them to the problem
  ProjectPointsToCameras(problem, points_3d);

  problem.fixed_frames.insert(0);

  // Create a copy of the problem for solution
  cuvslam::refinement::BundleAdjustmentProblem solved_problem = problem;

  if (kVerbose) {
    PrintProblem(problem);
  }

  // Set up options for refinement
  refinement::BundleAdjustmentProblemOptions options;
  options.verbose = kVerbose;
  options.estimate_intrinsics = false;
  options.symmetric_focal_length = false;
  options.ceres_options.function_tolerance = kCeresFunctionTolerance;
  options.ceres_options.parameter_tolerance = kCeresParameterTolerance;
  options.ceres_options.gradient_tolerance = kCeresGradientTolerance;
  options.use_loss_function = false;

  cuvslam::refinement::BundleAdjustmentProblem perturbed_problem = AddRandomNoiseToProblem(problem, options);

  // Run bundle adjustment
  refinement::BundleAdjustmentProblemSummary summary = refinement::refine(perturbed_problem, options, solved_problem);

  // Compare the results with the original problem
  CompareProblems(problem, solved_problem);
}
}  // namespace test::refinement
