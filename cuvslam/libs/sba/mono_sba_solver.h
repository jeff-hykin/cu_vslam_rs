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
#include <unordered_map>

#include "camera/observation.h"
#include "common/isometry_utils.h"
#include "common/statistic.h"
#include "common/types.h"
#include "common/unaligned_types.h"
#include "common/vector_2t.h"
#include "math/twist.h"

namespace cuvslam::sba {

using Stat2T = StatisticalBase<Vector2T>;
using StatT = StatisticalBase<float>;

// Code below is implementation of the HZ 2nd Edition p.613 Algorithm A6.4,
// i.e. it is Numerical implementation of Sparse Bundle Adjustment (SBA) using
// Exponential Mapping for Rotation.

using MatrixW = MatrixMN<float, 6, 3>;
using MatrixWT = MatrixMN<float, 3, 6>;

using CameraDerivs = math::TwistDerivativesT::CameraDerivs;
using PointDerivs = math::TwistDerivativesT::PointDerivs;

struct Frame {
  FrameId id;

  // only first frame of the pimpl->frames will be used in cuVSLAM as
  // constrained one -> may be replaced by 'int cframe'
  bool constrained;

  math::VectorTwistT aD;
  math::VectorTwistT a;   // this is actually 1/a, maps world coords to frame coords
  math::VectorTwistT da;  // (iii) of HZ, delta_a
  Vector6T errA, epsilon;

  Matrix6T Uo, Ua;

  // indexed by track index
  std::vector<MatrixW> Y, W;
  std::vector<MatrixWT> WT;
  std::vector<Vector2T> coord, err;

  // One weight per 2D residual.
  // Idea is that if even one of the components is too large,
  // we should downweight contribution of this observation.
  std::vector<float> weights;

  std::vector<math::TwistDerivativesT::CameraDerivs> Adir;
  std::vector<math::TwistDerivativesT::CameraDerivsTransposed> AdirT;
  std::vector<math::TwistDerivativesT::PointDerivs> Bdir;
  std::vector<math::TwistDerivativesT::PointDerivsTransposed> BdirT;
};

struct Track {
  TrackId id;

  bool skip;
  bool removed;

  Vector3T b, bD;
  Vector3T db;  // (iv) of HZ Algorithm A6.4 p.613
  Vector3T errB;
  Matrix3T V, invVa;

  Matrix2T infoMat;

  std::vector<size_t> idxFrames;
};

// Notations (A, B, U, V, W) exactly correspond to HZ ones
class MonoSBASolver {
public:
  MonoSBASolver(size_t maxFrames, size_t maxTracks, int iterations = 1000, float minResidual = 0.001f,
                float minResidualSpeed = 0.001f)
      : minResidual_(minResidual),
        minResidualSpeed_(minResidualSpeed),
        maxIterations_(iterations),
        residual_(-1),
        iteration_(-1),
        numTracks_(0),
        lambda_(0.001f) {
    frames_.reserve(maxFrames);
    tracks_.reserve(maxTracks);
  }

  void addFrame(FrameId id, bool constrained, const Isometry3T& a, const std::map<TrackId, bool>& filtered,
                const std::vector<camera::Observation>& trackCoords);

  void addTrack(TrackId id, const Vector3T& b);
  size_t numFrames() const;
  size_t numTracks() const;

  bool removeTrack(TrackId id);

  void resetState();

  void reset();

  int solve();

  float getResidual() const;
  float getResidualInitial() const;
  float getLambda() const;
  int getIterations() const;

  const std::vector<Track>& tracks() const;
  const std::vector<Frame>& frames() const;

  CameraMap getCameras() const;
  Tracks3DMap getTracks3d() const;

  bool getFailed() const;
  Matrix6T getCovariance() const;

private:
  float minResidual_, minResidualSpeed_;
  int maxIterations_;

  float residual_;
  float residualInitial_;  // for testing only
  int iteration_;
  bool failed_ = false;

  // solve() will set this threshold to an appropriate value
  float huberThreshold_ = 1;

  // number of tracks added via addTrack, which might be
  // different from mapTracks_.size()
  size_t numTracks_;

  std::vector<Frame> frames_;
  std::vector<Track> tracks_;
  std::unordered_map<FrameId, size_t> mapFrames_;
  std::unordered_map<TrackId, size_t> mapTracks_;

  // Transitory
  float lambda_;

  // used in calculateInformationMatrices, it's here so that
  // we don't have to create the vector every time we need it
  std::vector<Stat2T> covarMatDiag_;
  Matrix6T prepCovariance_;
  Matrix6T covariance_;
  std::vector<StatT> covarMatXY_;
  std::vector<StatT> covarMatX_;
  std::vector<StatT> covarMatY_;

  void filterHitherPoints();

  void update();
  void updateAdbD();
  void calculateInformationMatrices();

  void computeWeights();

  float calculateResidual();

  void calculateDerivatives();

  void calculateV();
  void calculateU();
  void calculateW();

  void stepUV();

  void calculateSMatrix(MatrixXT& S, VectorXT& b);
  void calculateEps();
  bool calculateAdjustments();

  bool invert3x3(const Matrix3T& i, Matrix3T& o) const;
};

}  // namespace cuvslam::sba
