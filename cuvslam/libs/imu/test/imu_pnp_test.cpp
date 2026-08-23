
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

#warning imu_pnp_unit_test should be fixed
#if 0

#include <algorithm>
#include <list>
#include <random>

#include "common/environment.h"
#include "common/imu_measurement.h"
#include "common/include_gtest.h"
#include "common/isometry.h"
#include "common/log.h"
#include "common/rotation_utils.h"
#include "imu/imu_calibration.h"
#include "imu/imu_measurement.h"
#include "imu/soft_inertial_pnp.h"
#include "log/log_eigen.h"
#include "math/twist.h"
#include "odometry/increment_pose.h"
#include "sba/imu/imu_pnp.h"
#include "sba/imu/imu_pnp_types.h"
namespace {
using namespace cuvslam;

inline float rotationError(const Isometry3T& pose_error)
{
    const float a = (common::CalculateRotationFromSVD(pose_error.matrix()).trace() - 1) / 2;
    return acos(a < 1 ? (a > -1 ? a : -1) : 1);
}

inline float translationError(const Isometry3T& pose_error)
{
    return pose_error.translation().norm();
}

float DeltaT(int64_t ts1, int64_t ts2) {
    return static_cast<float>(ts2 - ts1) / static_cast<float>(1e9);
}

Vector3T Velocity(float dt, const Isometry3T& prev, const Isometry3T& next) {
    assert(dt > 0);
    const Vector3T dp{ next.translation() - prev.translation() };
    return dp / dt;
}

using AngleAxisT = Eigen::AngleAxis<float>;

AngleAxisT AngleAxis(const Isometry3T& prev, const Isometry3T& next) {
    const QuaternionT qp{ prev.linear() };
    const QuaternionT qn{ next.linear() };
    const QuaternionT dq{ qp.normalized().conjugate() * qn.normalized() };
    return AngleAxisT{ dq };
}

Vector3T AngularVelocity(float dt, const Isometry3T& prev,
                         const Isometry3T& next) {
    const AngleAxisT aa{ AngleAxis(prev, next) };
    assert(dt > 0);
    return (aa.angle() / dt) * aa.axis();
}

template <class FUNC>
Vector3T Velocity(int64_t time_ns, int64_t delta_time_ns, FUNC&& func, Isometry3T rig_from_imu = Isometry3T::Identity())
{
    // Velocity
    Isometry3T pose0, pose1;
    func(time_ns - delta_time_ns / 2, pose0);
    func(time_ns + delta_time_ns / 2, pose1);
    pose0 = pose0 * rig_from_imu;
    pose1 = pose1 * rig_from_imu;
    float delta_t = DeltaT(time_ns - delta_time_ns / 2, time_ns + delta_time_ns / 2);
    Vector3T start_velocity = Velocity(delta_t, pose0, pose1);
    return start_velocity;
}

template <class FUNC>
Vector3T AngularVelocity(int64_t time_ns, int64_t delta_time_ns, FUNC&& func, Isometry3T rig_from_imu = Isometry3T::Identity())
{
    // Velocity
    Isometry3T pose0, pose1;
    func(time_ns - delta_time_ns / 2, pose0);
    func(time_ns + delta_time_ns / 2, pose1);
    pose0 = pose0 * rig_from_imu;
    pose1 = pose1 * rig_from_imu;
    float delta_t = DeltaT(time_ns - delta_time_ns / 2, time_ns + delta_time_ns / 2);
    Vector3T angular_velocity = AngularVelocity(delta_t, pose0, pose1);
    return angular_velocity;
}

template <class FUNC>
Vector3T LinearAcceleration(int64_t time_ns, int64_t delta_time_ns, FUNC&& func, Isometry3T rig_from_imu = Isometry3T::Identity())
{
    Vector3T v0 = Velocity(time_ns - delta_time_ns / 2, delta_time_ns, func, rig_from_imu);
    Vector3T v1 = Velocity(time_ns + delta_time_ns / 2, delta_time_ns, func, rig_from_imu);

    float delta_t = DeltaT(time_ns - delta_time_ns / 2, time_ns + delta_time_ns / 2);
    Vector3T acceleration = (v1 - v0) / delta_t;
    return acceleration;
}

Isometry3T RandomPoseShift(std::mt19937& rng, const Matrix6T& covariance) {
    Eigen::LLT<Matrix6T> lltOfA(covariance); // compute the Cholesky decomposition of A
    Matrix6T chol = lltOfA.matrixL();

    std::normal_distribution<> nd(0, 1);
    Vector6T rn;

    for (int i = 0; i < 6; ++i) {
        rn[i] = (float)nd(rng);
    }

    Vector6T t = chol * rn;

    Isometry3T m;
    math::Exp(m, t);

    return m;
}

bool loadPoses(const std::string& file_name, std::vector<Isometry3T>& poses)
{
    FILE* fp = std::fopen(file_name.c_str(), "r");

    if (!fp)
    {
        return false;
    }

    poses.clear();
    while (!std::feof(fp))
    {
        Isometry3T pose;
        Matrix4T& m = pose.matrix();
        float fm[12];

        for (size_t k = 0; k < 12; ++k)
        {
            if (!std::fscanf(fp, "%f", &fm[k]))
            {
                assert(false);
                std::fclose(fp);
                return false;
            }
        }

        for (size_t i = 0; i < 3; ++i)
        {
            for (size_t j = 0; j < 4; ++j)
            {
                m(i, j) = fm[i * 4 + j];
            }
        }

        assert(m.allFinite());
        pose.makeAffine();
        poses.push_back(pose);
    }

    std::fclose(fp);

    poses.pop_back(); // current function somehow adds the last pose twice, get rid of it

    if (!poses[0].isApprox(Isometry3T::Identity(), 1e-3)) {
        Isometry3T start_pose_inverse = poses[0].inverse();
        for (auto& p: poses) {
            p = start_pose_inverse * p;
        }
    }
    return true;
}

struct PoseWithTime {
    Isometry3T pose;
    int64_t time_ns;
};

bool read_gt_poses(
    std::vector<PoseWithTime>& trajectory,
    float& duration_s,
    const std::string& dataset = "euroc",
    const std::string& sequence = "MH_01_easy",
    float cam_fps = 60.f) {

    std::string gt_path;
    if (!Environment::GetVar(Environment::CUVSLAM_DATASETS).empty())
    {
        std::string sequences_path = Environment::GetVar(Environment::CUVSLAM_DATASETS);
        if (!IsPathEndWithSlash(sequences_path)) {
            sequences_path += "/";
        }
        gt_path = sequences_path + dataset;
        if (!IsPathEndWithSlash(gt_path)) {
            gt_path += "/";
        }
    }
    gt_path += sequence;
    if (!IsPathEndWithSlash(gt_path)) {
        gt_path += "/";
    }
    gt_path += "gt.txt";

    std::vector<Isometry3T> gt_poses;

    bool ret = loadPoses(gt_path, gt_poses);
    if (!ret) {
        return false;
    }

    float delta_t_s = 1.f / cam_fps;
    duration_s = 0;
    trajectory.clear();

    for (auto& pose: gt_poses) {
        trajectory.push_back({
                                 pose,
                                 static_cast<int64_t>(duration_s * 1e9f)
                             });
        duration_s += delta_t_s;
    }

    return true;
}

bool interpolate_pose(const std::vector<PoseWithTime>& trajectory, int64_t time_ns, Isometry3T& pose) {
    if (time_ns < trajectory.front().time_ns || time_ns > trajectory.back().time_ns) {
        return false;
    }

    auto it = std::upper_bound(
        std::next(trajectory.begin()),
        trajectory.end(),
        time_ns,
        [](int64_t time, const PoseWithTime& p){
            return p.time_ns >= time;
        });

    if (it == trajectory.end()) {
        return false;
    }

    const Isometry3T& p1 = std::prev(it)->pose;
    int64_t t1 = std::prev(it)->time_ns;

    const Isometry3T& p2 = it->pose;
    int64_t t2 = it->time_ns;

    Vector6T twist;
    math::Log(twist, p2 * p1.inverse());

    float alpha = static_cast<float>(time_ns - t1) / static_cast<float>(t2 - t1);
    twist *= alpha;

    math::Exp(pose, twist);
    pose = pose * p1;
    return true;
}


struct Observation {
    Vector2T uv;
    int32_t point_id;
    int32_t pose_id;
    int8_t cam_id;

    Matrix2T info = Matrix2T::Identity() * 5e4;
};

void GeneratePoints(
    const std::vector<Isometry3T>& poses,
    const std::vector<int32_t>& pose_ids,
    const Isometry3T& left_from_right,
    std::vector<Vector3T>& points,
    std::vector<Observation>& obs,
    size_t num_points = 20
) {
    std::mt19937 gen(::testing::UnitTest::GetInstance()->random_seed());
    std::uniform_real_distribution<float> dist_z(5, 10.f);
    std::uniform_int_distribution<int> dist_i(0, poses.size() - 1);
    size_t point_counter = 0;

    while (point_counter < num_points) {

        float z = dist_z(gen);
        std::uniform_real_distribution<float> dist_xy(-z, z);

        Vector3T point_r = {dist_xy(gen), dist_xy(gen), z};
        Vector3T point_w = poses[dist_i(gen)] * point_r;

        std::vector<Vector3T> pc, pc_r;
        std::vector<Vector2T> uv, uv_r;
        bool ok = true;
        for (const Isometry3T& pose: poses) {
            Vector3T p_c = pose.inverse() * point_w;
            Vector3T p_c_r = (pose * left_from_right).inverse() * point_w;

            if (p_c.z() < 1.f || p_c_r.z() < 1.f) {
                ok = false;
                break;
            }

            Vector2T uv_ = p_c.topRows(2) / p_c.z();
            Vector2T uv_r_ = p_c_r.topRows(2) / p_c_r.z();

            if ((uv_.array() > 1.f).any() || (uv_.array() < -1.f).any()) {
                ok = false;
                break;
            }

            if ((uv_r_.array() > 1.f).any() || (uv_r_.array() < -1.f).any()) {
                ok = false;
                break;
            }

            if (!uv.empty()) {
                if (uv.back().isApprox(uv_, 1e-2)) {
                    ok = false;
                    break;
                }
            }

            uv.emplace_back(uv_);
            uv_r.emplace_back(uv_r_);

        }
        if (!ok) {
            continue;
        }

        int point_idx = points.size();
        points.push_back(point_w);

        for (int i = 0; i < static_cast<int>(poses.size()); i++) {
            obs.push_back({uv[i], point_idx, pose_ids[i], 0});
            obs.push_back({uv_r[i], point_idx, pose_ids[i], 1});
        }

        point_counter++;
    }
}

void AddNoise(std::mt19937& gen, sba_imu::Pose& pose) {
    Isometry3T shift = RandomPoseShift(gen, Matrix6T::Identity() * 1e-3);
    pose.w_from_imu = pose.w_from_imu * shift;
}

class PoseBuffer {
 public:
    size_t max_size_ = 10;
    imu::ImuCalibration calib_;
    Isometry3T left_from_right_;
    float fps_ = 10;

    explicit PoseBuffer(const Isometry3T& start_pose, const Isometry3T& left_from_right, float fps = 10):
        left_from_right_(left_from_right), fps_(fps) {
        sba_imu::Pose p;
        p.w_from_imu = start_pose * calib_.rig_from_imu();
        p.preintegration = sba_imu::IMUPreintegration();
        p.acc_bias.setZero();
        p.gyro_bias.setZero();
        p.velocity.setZero();
        poses_.push_back(p);
    }

    void add(const Isometry3T& w_from_left) {
        sba_imu::Pose imu_pose;
        imu_pose.w_from_imu = w_from_left * calib_.rig_from_imu();
        imu_pose.preintegration = sba_imu::IMUPreintegration();
        imu_pose.acc_bias.setZero();
        imu_pose.gyro_bias.setZero();
        imu_pose.velocity.setZero();

        // pose is fixed, velocities and biases as free;
        Eigen::Matrix<float, 15, 1> diag;
        diag << 1e6f, 1e6f, 1e6f, 1e6f, 1e6f, 1e6f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f;
        imu_pose.info = diag.asDiagonal();

        poses_.back().velocity = (imu_pose.w_from_imu.translation() - poses_.back().w_from_imu.translation()) / fps_;


        poses_.push_back(imu_pose);
        while (poses_.size() > max_size_) {
            poses_.pop_front();
        }
    }

    size_t size() const {
        return poses_.size();
    }

    void register_measumerent(const cuvslam::imu::ImuMeasurement& m) {
        poses_.back().preintegration.IntegrateNewMeasurement(calib_, m);
    }

    void get_poses(std::vector<sba_imu::Pose>& poses) const {
        poses.clear();
        poses.reserve(max_size_);
        for (const sba_imu::Pose& p: poses_) {
            poses.push_back(p);
        }
    }

    void build_pnp_input(sba_imu::StereoPnPInput& problem, const Vector3T& gravity) {
        problem.points.clear();
        problem.point_ids.clear();
        problem.observation_xys.clear();
        problem.observation_infos.clear();
        problem.camera_ids.clear();

        problem.rig.camera_from_rig[0] = Isometry3T::Identity();
        problem.rig.camera_from_rig[1] = left_from_right_.inverse();
        problem.rig.num_cameras = 2;

        problem.gravity = gravity; // in world frame
        problem.max_iterations = 7;
        problem.robustifier_scale = 20;
        problem.prior_velocity = 0;
        problem.prior_gyro = 1e6;
        problem.prior_acc = 1e6;

        std::vector<Observation> obs;
        {
            std::vector<Isometry3T> poses;
            std::vector<int32_t> pose_ids = {0, 1};

            poses.push_back(std::prev(poses_.end(), 2)->w_from_imu * calib_.rig_from_imu().inverse());
            poses.push_back(std::prev(poses_.end(), 1)->w_from_imu * calib_.rig_from_imu().inverse());

            GeneratePoints(
                poses, pose_ids,
                left_from_right_,
                problem.points,
                obs,
                500);
        }

        std::stable_sort(obs.begin(), obs.end(), [](const Observation& lhs, const Observation& rhs){
            return std::tie(lhs.pose_id, lhs.point_id) < std::tie(rhs.pose_id, rhs.point_id);
        });

        for (const Observation& o: obs) {
            if (o.pose_id == 1) {
                problem.observation_xys.push_back(o.uv);
                problem.observation_infos.push_back(o.info);
                problem.point_ids.push_back(o.point_id);
                problem.camera_ids.push_back(o.cam_id);
            }
        }

    }

    std::list<sba_imu::Pose> poses_;
};

template <class FUNC>
void FullIntegrate(
    FUNC&& func,
    int64_t max_time_ns,
    float g_norm = 9.81f,
    const Vector3T& gyro_mean_bias = Vector3T(0, 0, 0),
    const Vector3T& acc_mean_bias = Vector3T(0, 0, 0),
    float gyro_sigma = 1e-3f,
    float acc_sigma = 1e-3f,
    float vo_angle_sigma = 1e-3f,
    float vo_trans_sigma = 1e-1f)
{
    const Vector3T g(0, g_norm, 0);  // OpenCV world: +Y down, gravity along +Y

    Isometry3T left_from_right = Isometry3T::Identity();
    left_from_right.translation() << 0.1, 0, 0;

    // 1.000.000.000
    const int64_t second_ns = 1000000000;

    int64_t time_ns = 0.5 * second_ns;
    int64_t end_time_ns = max_time_ns - second_ns;
    const int64_t delta_time_ns = second_ns / 200;

    float fps = 10;
    const int64_t frame_duration_ns = second_ns / fps;
    int64_t frame_time_ns = time_ns;
    int frame_id = 0;

    Isometry3T start_pose = Isometry3T::Identity();
    func(time_ns, start_pose);

    PoseBuffer buffer(start_pose, left_from_right, fps);
    Isometry3T rig_from_imu = buffer.calib_.rig_from_imu();

    Isometry3T last_pose = start_pose;
    Matrix6T last_covariance;

    std::mt19937 gen(::testing::UnitTest::GetInstance()->random_seed());
    std::normal_distribution<float> gyro_normal_distribution(0, gyro_sigma);
    std::normal_distribution<float> acc_normal_distribution(0, acc_sigma);
    std::normal_distribution<float> vo_angle_normal_distribution(0, vo_angle_sigma);

    for (; time_ns < end_time_ns; time_ns += delta_time_ns) {

        Isometry3T original_pose;
        func(time_ns, original_pose);

        cuvslam::imu::ImuMeasurement m_world; // (world)
        cuvslam::imu::ImuMeasurement m_imu; //(left)


        m_world.time_ns = time_ns;
        m_imu.time_ns = time_ns;

        Isometry3T imu_pose_inv;
        {
            Isometry3T imu_pose = original_pose * rig_from_imu;
            imu_pose_inv = imu_pose.inverse();

            m_imu.angular_velocity = AngularVelocity(time_ns, delta_time_ns, func, rig_from_imu);
            m_world.linear_acceleration = LinearAcceleration(time_ns, delta_time_ns, func, rig_from_imu) - g;

            m_world.angular_velocity = common::CalculateRotationFromSVD(imu_pose.matrix()) * m_imu.angular_velocity;
            m_imu.linear_acceleration = common::CalculateRotationFromSVD(imu_pose_inv.matrix()) * m_world.linear_acceleration;
        }

        // add noise in imu frame
        Vector3T gyro_bias = gyro_mean_bias;
        Vector3T acc_bias = acc_mean_bias;

        m_imu.angular_velocity += gyro_bias;
        m_imu.linear_acceleration += acc_bias;

        // feed integrator
        buffer.register_measumerent(m_imu);

        // 1/60
        if (time_ns==0 || time_ns >= frame_time_ns + frame_duration_ns) {
            Isometry3T prev_original_pose;
            func(frame_time_ns, prev_original_pose);
            Isometry3T original_delta_pose = prev_original_pose.inverse() * original_pose;

            if (vo_angle_sigma < std::numeric_limits<float>::epsilon()) {
                vo_angle_sigma = std::numeric_limits<float>::epsilon();
            }
            if (vo_trans_sigma < std::numeric_limits<float>::epsilon()) {
                vo_trans_sigma = std::numeric_limits<float>::epsilon();
            }

            float angle_var = vo_angle_sigma * vo_angle_sigma;
            float trans_var = vo_trans_sigma * vo_trans_sigma;
            Matrix6T odometry_covariance = Matrix6T::Zero();
            {
                Vector6T diagonal;
                diagonal << angle_var, angle_var, angle_var, trans_var, trans_var, trans_var;
                odometry_covariance.diagonal() = diagonal;
            }
            frame_time_ns = time_ns;
            last_pose = odom::increment_pose(last_pose, original_delta_pose);

            buffer.add(last_pose);
            frame_id++;

            {
                if (buffer.size() >= 10) {
                    sba_imu::StereoPnPInput input;
                    buffer.build_pnp_input(input, g);

                    sba_imu::Pose pose_original = buffer.poses_.back();
                    sba_imu::Pose pose_noise = pose_original;
                    AddNoise(gen, pose_noise);
                    sba_imu::Pose prev_pose = *std::prev(buffer.poses_.end(), 2);
                    imu::ImuCalibration imu_calibration;
                    ASSERT_TRUE(SoftInertialPnP(imu_calibration, input, prev_pose, pose_noise));
                    float thesh = 0.10;
                    {
                        const Isometry3T& orig = pose_original.w_from_imu;
                        const Isometry3T& opt = pose_noise.w_from_imu;
                        Vector6T twist;
                        math::Log(twist, opt * orig.inverse());
                        if (twist.norm() >= thesh) {
                            std::cout << "twist.norm() = " << twist.norm() << std::endl;
                            std::cout << "orig = " << std::endl << orig.matrix() << std::endl;
                            std::cout << "opt = " << std::endl << opt.matrix() << std::endl;
                            std::cout << "frame = " << frame_id << std::endl;
                        }
                        ASSERT_TRUE(twist.norm() < thesh);
                    }
                    buffer.poses_.back().velocity = pose_noise.velocity;
                    buffer.poses_.back().gyro_bias = pose_noise.gyro_bias;
                    buffer.poses_.back().acc_bias = pose_noise.acc_bias;
                }
            }
        }
    }
}

void run_test(
    const std::string& dataset = "euroc",
    const std::string& sequence = "V2_03_difficult",
    const Vector3T& gyro_mean_bias = Vector3T{0, 0, 0},
    const Vector3T& acc_mean_bias = Vector3T{0, 0, 0},
    float gyro_sigma=1e-5f,
    float acc_sigma=1e-5f,
    float vo_angle_sigma = 1e-8f,
    float vo_trans_sigma = 1e-8f) {

    std::vector<PoseWithTime> trajectory;
    float duration = 0;
    float g_norm = 9.81;

    read_gt_poses(trajectory, duration, dataset, sequence, 20);

    FullIntegrate([&](int64_t time_ns, Isometry3T& m) {
                      interpolate_pose(trajectory, time_ns, m);
                  },static_cast<int64_t>(duration * 1e9f),
                  g_norm, gyro_mean_bias, acc_mean_bias,
                  gyro_sigma, acc_sigma,
                  vo_angle_sigma, vo_trans_sigma);
}

} // namespace

namespace test::sba_imu {

TEST(Imu, ImuPnPTest) {
    std::mt19937 gen(::testing::UnitTest::GetInstance()->random_seed());
    std::normal_distribution<float> nd_gyro(0, 1e-5);
    std::normal_distribution<float> nd_acc(0, 1e-5);

    for (int i = 0; i < 1; i++) {

        Vector3T gyro_bias = Vector3T::Zero();//{nd_gyro(gen),nd_gyro(gen),nd_gyro(gen)};
        Vector3T acc_bias = Vector3T::Zero();//{nd_acc(gen),nd_acc(gen),nd_acc(gen)};
        run_test("euroc", "V2_03_difficult",gyro_bias,acc_bias);
    }
}

} // namespace test::sba_imu

#endif
