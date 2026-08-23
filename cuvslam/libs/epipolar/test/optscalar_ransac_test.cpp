
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

#include "epipolar/test/camera_points_reconstruction_integ_test.h"

#include "epipolar/fundamental_matrix_utils.h"
#include "epipolar/optscalar_triangle_ransac.h"
#include "epipolar/point_reconstruction.h"

namespace test::epipolar {
using namespace cuvslam;

using OptimalScalarRansac = math::Ransac<OptimalScalarTriangle>;

class OptimalScalarRansacTest : public CameraPointsReconstructionIntegTest {
protected:
  virtual void SetUp() {
    CameraPointsReconstructionIntegTest::SetUp();
    // create relative transformation between frames 2&3 the same as between frames 1&2 rotation
    // wise but scale relative translation.
    scale_ = 7.0f;
    Isometry3T camera3InCamera2Space = absoluteCamera2_;
    camera3InCamera2Space.translation() *= scale_;
    const Isometry3T absoluteCamera3 = absoluteCamera2_ * camera3InCamera2Space;

    // scaled2Dpairs_ will have projection of our 3d points in canonical camera (Identity) in
    // pair.first Vector2T of the of scaled2Dpairs_ which correspond to 1st camera of 3, and
    // projection of the same points in 3rd camera in the pair.second Vector2T of scaled2Dpairs_.
    GeneratePairOf2DPointsFrom3DPoints(expected3DPoints_, absoluteCamera1_, absoluteCamera3, &scaled2Dpairs_);

    // replace pair.first of scaled2Dpairs_ by the projection of the 3d point in 2nd camera (of
    // frames 1-3) aperture, now scaled2Dpairs_ presents projections of 3d points in 2nd and
    // 3rd cameras apertures and add noise to third camera projections to simulate what will happen
    // in real life (= triangulated point using scaled2Dpairs_ will be not exactly on the ray from
    // second camera through 2d projection of 3d point in second camera aperture).
    Vector2TPairVectorCIt it2 = point2Dpairs_.cbegin();
    for_each(scaled2Dpairs_.begin(), scaled2Dpairs_.end(), [&it2](Vector2TPair& pp) {
      pp.first = it2->second;
      pp.second += Vector2T::Random() * 0.05f;
      it2++;
    });

    // now transfer expected3DPoints_ in second frame (of 3) camera local space, as second frame
    // will be our reference frame to estimate scale factor for both camera pairs: 1&2 and 2&3.
    // m_3DPointsInSecondCameraSpace are points triangulated between frames 1&2 in 2nd camera
    // space.
    const Isometry3T xform = expectedRelativeTransform_;
    points3dInSecondCameraSpace_.resize(expected3DPoints_.size());
    Vector3TVector::iterator it3 = points3dInSecondCameraSpace_.begin();
    for_each(expected3DPoints_.cbegin(), expected3DPoints_.cend(),
             [&it3, &xform](const Vector3T& pp) { *it3++ = xform * pp; });

    // Now if we triangulate using transformation between frames 1&2 using 2d tracks obtained by
    // transformation scaled by scale_ relative to transformation between frames 1&2 then our
    // triangulated points should be in z direction of frame 2 camera space scaled by 1/m_scale
    // relative to z in 2nd camera space of points triangulated based on motion between cameras
    // 1&2. If, by some magic insight, we would know our scale_ and scale
    // expectedRelativeTransform_ by it (T only), scaled3DPoints_ will be, up to some noise, in
    // the positions defined by m_3DPointsInSecondCameraSpace in 2nd camera space.
    Reconstruct3DPointsFrom2DPointsAndRelativeTransform(scaled2Dpairs_, expectedRelativeTransform_, scaled3DPoints_);
  }

protected:
  Isometry3T absoluteCamera3_;
  Vector3TVector scaled3DPoints_;
  Vector3TVector points3dInSecondCameraSpace_;
  Vector2TPairVector scaled2Dpairs_;

public:
  float scale_;
};

TEST_F(OptimalScalarRansacTest, SimilarTriangle) {
  std::vector<float> sampleSequence;

  Vector3TVector::const_iterator it = points3dInSecondCameraSpace_.cbegin();
  for_each(scaled3DPoints_.cbegin(), scaled3DPoints_.cend(), [&it, &sampleSequence](const Vector3T& v) {
    if (std::abs(v.z()) > epsilon()) {
      sampleSequence.push_back((*it).z() / v.z());
    }

    it++;
  });

  float scale = 1;

  OptimalScalarRansac st;
  st.setOptions(2.5f);
  size_t numIterations = st(scale, sampleSequence.cbegin(), sampleSequence.cend());
  EXPECT_NE(size_t(0), numIterations);

  EXPECT_LT(std::abs(scale_ - scale), st.getTheshold());
}

// TEST_F(OptimalScalarRansacTest, TriFocalTensor) {
//
// }

}  // namespace test::epipolar
