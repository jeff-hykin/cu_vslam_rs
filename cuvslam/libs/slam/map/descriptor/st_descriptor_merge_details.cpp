
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

#include "slam/map/descriptor/st_descriptor_merge_details.h"

#include "common/include_eigen.h"
#include "sof/gaussian_coefficients.h"
#include "sof/kernel_operator.h"

#include "slam/map/descriptor/feature_descriptor.h"
#include "slam/map/descriptor/st_descriptor_ops.h"

namespace {
using FeaturePatch = cuvslam::sof::STFeaturePatch;
}  // namespace

namespace cuvslam::slam::merge_details {

using SmallPatch = ImageMatrixPatch<float, 5, 5>;

SmallPatch make_x_patch() {
  SmallPatch patch;
  static_assert(SmallPatch::ColsAtCompileTime % 2 == 1, "");
  const int m = (SmallPatch::ColsAtCompileTime - 1) / 2;
  for (int y = 0; y < SmallPatch::RowsAtCompileTime; ++y) {
    for (int x = 0; x < SmallPatch::ColsAtCompileTime; ++x) {
      patch(y, x) = static_cast<float>(x - m);
    }
  }
  return patch;
}

SmallPatch make_y_patch() {
  SmallPatch patch;
  static_assert(SmallPatch::RowsAtCompileTime % 2 == 1, "");
  const int m = (SmallPatch::RowsAtCompileTime - 1) / 2;
  for (int y = 0; y < SmallPatch::RowsAtCompileTime; ++y) {
    for (int x = 0; x < SmallPatch::ColsAtCompileTime; ++x) {
      patch(y, x) = static_cast<float>(y - m);
    }
  }
  return patch;
}

const SmallPatch x_patch = make_x_patch();
const SmallPatch y_patch = make_y_patch();
const SmallPatch xx_patch = x_patch.cwiseProduct(x_patch);
const SmallPatch yy_patch = y_patch.cwiseProduct(y_patch);
const SmallPatch xy_patch = x_patch.cwiseProduct(y_patch);

template <int size>
struct Interpolation {
  Interpolation(float v) {
    static_assert(size % 2 == 1, "");
    float iv_;
    const float fv = std::modf(std::abs(v + (size - 1) / 2), &iv_);
    assert(iv_ >= 0);
    const int iv = static_cast<int>(iv_) % (2 * (size - 1));
    if (iv < size - 1) {
      i1 = iv;
      i2 = iv + 1;
    } else {
      i1 = 2 * (size - 1) - iv;
      i2 = 2 * (size - 1) - iv - 1;
    }
    c1 = 1 - fv;
    c2 = fv;
    assert(0 <= i1 && i1 < size);
    assert(0 <= i2 && i2 < size);
  }
  int i1;
  int i2;
  float c1;
  float c2;
};

float get_pixel(const FeaturePatch& patch, const Vector2T& xy) {
  const Interpolation<FeaturePatch::ColsAtCompileTime> x(xy.x());
  const Interpolation<FeaturePatch::RowsAtCompileTime> y(xy.y());
  return y.c1 * x.c1 * patch(y.i1, x.i1) + y.c2 * x.c1 * patch(y.i2, x.i1) + y.c1 * x.c2 * patch(y.i1, x.i2) +
         y.c2 * x.c2 * patch(y.i2, x.i2);
}

float get_scale(int source_level, int target_level) {
  int shift = target_level - source_level;
  if (shift > 0) {
    return 1.f / (1 << shift);
  }
  return 1 << (-shift);
}

Vector2T scale_xy(const Vector2T& xy, int source_level, int target_level) {
  return xy * get_scale(source_level, target_level);
}

bool has_level(const STDescriptor& fd, int level) {
  assert(level >= 0);
  return (fd.levels_mask & (1 << level)) != 0;
}
bool has_level(uint32_t levels_mask, int level) {
  assert(level >= 0);
  return (levels_mask & (1 << level)) != 0;
}

class DescriptorImage {
public:
  DescriptorImage(const STDescriptor& fd, int start_level)
      : levels_mask(fd.levels_mask),
        image_patches_size(fd.image_patches_size),
        image_patches(fd.image_patches),
        start_level(start_level) {}
  DescriptorImage(uint32_t levels_mask, uint32_t image_patches_size, sof::STFeaturePatch* image_patches,
                  int start_level)
      : levels_mask(levels_mask),
        image_patches_size(image_patches_size),
        image_patches(image_patches),
        start_level(start_level) {}

  // use x coord in [-x_mag(), x_mag()]
  float x_mag() const {
    constexpr float x_mag = (FeaturePatch::ColsAtCompileTime - 1) / 2;
    const int last_level = image_patches_size - 1;
    for (int level = last_level; level >= start_level; --level) {
      if (has_level(levels_mask, level)) {
        return get_scale(level, start_level) * x_mag;
      }
    }
    return 0;
  }

  // use y coord in [-y_mag(), y_mag()]
  float y_mag() const {
    constexpr float y_mag = (FeaturePatch::RowsAtCompileTime - 1) / 2;
    const int last_level = image_patches_size - 1;
    for (int level = last_level; level >= start_level; --level) {
      if (has_level(levels_mask, level)) {
        return get_scale(level, start_level) * y_mag;
      }
    }
    return 0;
  }

  float operator()(const Vector2T& xy) const {
    const int level = find_level(xy);
    // assert(0 <= level && level < static_cast<int>(fd.image_patches_size));
    return get_pixel(image_patches[level], scale_xy(xy, start_level, level));
  }

private:
  uint32_t levels_mask;
  uint32_t image_patches_size;
  const sof::STFeaturePatch* image_patches;
  const int start_level;

  int find_level(const Vector2T& xy) const {
    const float x_scale = std::abs(xy.x()) / x_mag();
    const float y_scale = std::abs(xy.y()) / y_mag();
    const float scale = std::max(x_scale, y_scale);

    const int size = image_patches_size;
    for (int level = start_level; level < size; ++level) {
      if (has_level(levels_mask, level) && scale * get_scale(start_level, level) <= 1) {
        return level;
      }
    }

    for (int level = image_patches_size - 1; level >= start_level; --level) {
      if (has_level(levels_mask, level)) {
        return level;
      }
    }

    assert(false);

    return 0;
  }
};

template <typename Patch>
float dot(const Patch& p0, const Patch& p1) {
  assert(p0.rows() == p1.rows());
  const int rows = p0.rows();
  assert(p0.cols() == p1.cols());
  const int cols = p0.cols();
  float sum = 0;
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      sum += p0(r, c) * p1(r, c);
    }
  }
  return sum;
}

void compute_patch(SmallPatch& patch, const FeaturePatch& feature_patch) {
  constexpr int scols = SmallPatch::ColsAtCompileTime;
  constexpr int srows = SmallPatch::RowsAtCompileTime;
  constexpr int fcols = FeaturePatch::ColsAtCompileTime;
  constexpr int frows = FeaturePatch::RowsAtCompileTime;

  static_assert(scols < fcols, "");
  static_assert(srows < frows, "");
  static_assert((fcols - scols) % 2 == 0, "");
  static_assert((frows - srows) % 2 == 0, "");

  constexpr int xs = (fcols - scols) / 2;
  constexpr int ys = (frows - srows) / 2;

  // TODO: is there more efficient way to retrieve subpatch?
  for (int y = 0; y < srows; ++y) {
    for (int x = 0; x < scols; ++x) {
      patch(y, x) = feature_patch(y + ys, x + xs);
    }
  }
}

bool compute_patch(SmallPatch& patch, const DescriptorImage& image, const Matrix2T& map, const Vector2T& xy) {
  constexpr int scols = SmallPatch::ColsAtCompileTime;
  constexpr int srows = SmallPatch::RowsAtCompileTime;
  constexpr int fcols = FeaturePatch::ColsAtCompileTime;
  constexpr int frows = FeaturePatch::RowsAtCompileTime;

  static_assert(scols <= fcols, "");
  static_assert(srows <= frows, "");
  static_assert((fcols - scols) % 2 == 0, "");
  static_assert((frows - srows) % 2 == 0, "");

  static_assert(scols % 2 == 1, "");
  constexpr int sx_mag = (scols - 1) / 2;

  static_assert(srows % 2 == 1, "");
  constexpr int sy_mag = (srows - 1) / 2;

  const Vector2T xy00 = xy + map * Vector2T(-sx_mag, -sy_mag);
  const Vector2T xy01 = xy + map * Vector2T(-sx_mag, sy_mag);
  const Vector2T xy10 = xy + map * Vector2T(sx_mag, -sy_mag);
  const Vector2T xy11 = xy + map * Vector2T(sx_mag, sy_mag);

  const std::pair<float, float> minmax_x = std::minmax<float>({xy00.x(), xy01.x(), xy10.x(), xy11.x()});
  const std::pair<float, float> minmax_y = std::minmax<float>({xy00.y(), xy01.y(), xy10.y(), xy11.y()});

  const float ix_mag = image.x_mag();
  const float iy_mag = image.y_mag();

  if (!(-ix_mag <= minmax_x.first && minmax_x.second <= ix_mag)) {
    return false;
  }
  if (!(-iy_mag <= minmax_y.first && minmax_y.second <= iy_mag)) {
    return false;
  }

  for (int y = 0; y < srows; ++y) {
    for (int x = 0; x < scols; ++x) {
      patch(y, x) = image(xy + map * Vector2T(x - sx_mag, y - sy_mag));
    }
  }

  return true;
}

void compute_gx(FeaturePatch& gx, const DescriptorImage& fi) {
  static_assert(FeaturePatch::ColsAtCompileTime % 2 == 1, "");
  static_assert(FeaturePatch::RowsAtCompileTime % 2 == 1, "");

  constexpr int xs = (FeaturePatch::ColsAtCompileTime - 1) / 2;
  constexpr int ys = (FeaturePatch::RowsAtCompileTime - 1) / 2;

  constexpr int n_coeffs = sizeof(sof::DSPDerivCoeffs) / sizeof(sof::DSPDerivCoeffs[0]);
  static_assert(n_coeffs % 2 == 1, "");
  constexpr int shift = (n_coeffs - 1) / 2;

  for (int y = 0; y < FeaturePatch::RowsAtCompileTime; ++y) {
    for (int x = 0; x < FeaturePatch::ColsAtCompileTime; ++x) {
      float sum = 0;
      for (int i = 0; i < n_coeffs; ++i) {
        sum += sof::DSPDerivCoeffs[i] * fi({x - xs - shift + i, y - ys});
      }
      gx(y, x) = sum;
    }
  }
}

void compute_gy(FeaturePatch& gy, const DescriptorImage& fi) {
  static_assert(FeaturePatch::ColsAtCompileTime % 2 == 1, "");
  static_assert(FeaturePatch::RowsAtCompileTime % 2 == 1, "");

  constexpr int xs = (FeaturePatch::ColsAtCompileTime - 1) / 2;
  constexpr int ys = (FeaturePatch::RowsAtCompileTime - 1) / 2;

  constexpr int n_coeffs = sizeof(sof::DSPDerivCoeffs) / sizeof(sof::DSPDerivCoeffs[0]);
  static_assert(n_coeffs % 2 == 1, "");
  constexpr int shift = (n_coeffs - 1) / 2;

  for (int y = 0; y < FeaturePatch::RowsAtCompileTime; ++y) {
    for (int x = 0; x < FeaturePatch::ColsAtCompileTime; ++x) {
      float sum = 0;
      for (int i = 0; i < n_coeffs; ++i) {
        sum += sof::DSPDerivCoeffs[i] * fi({x - xs, y - ys - shift + i});
      }
      gy(y, x) = sum;
    }
  }
}

void compute_gx(std::vector<FeaturePatch>& gx_image_patches, const STDescriptor& fd) {
  gx_image_patches = std::vector<FeaturePatch>(fd.image_patches_size);
  const int size = fd.image_patches_size;
  for (int level = 0; level < size; ++level) {
    if (has_level(fd, level)) {
      const DescriptorImage fi(fd, level);
      compute_gx(gx_image_patches[level], fi);
    }
  }
}

void compute_gy(std::vector<FeaturePatch>& gy_image_patches, const STDescriptor& fd) {
  gy_image_patches = std::vector<FeaturePatch>(fd.image_patches_size);
  const int size = fd.image_patches_size;
  for (int level = 0; level < size; ++level) {
    if (has_level(fd, level)) {
      const DescriptorImage fi(fd, level);
      compute_gy(gy_image_patches[level], fi);
    }
  }
}

void compute_residual(SmallPatch& residual, const SmallPatch& patch1, const SmallPatch& patch2) {
  const float delta = (patch1.sum() - patch2.sum()) / static_cast<float>(SmallPatch::SizeAtCompileTime);

  residual = patch1 - patch2 - SmallPatch::Constant(delta);
}

// Gaussian distribution integral for sigma=0.5
constexpr float GaussianWeightedFeatureCoeffs5[] = {0.1533884991456948, 0.22146091250452496, 0.2503011766995605,
                                                    0.22146091250452496, 0.1533884991456948};

const sof::KernelOperator<SmallPatch> gaussian_weights(
    (Eigen::Matrix<float, SmallPatch::RowsAtCompileTime, 1, Eigen::DontAlign>(GaussianWeightedFeatureCoeffs5) *
     Eigen::Matrix<float, 1, SmallPatch::ColsAtCompileTime>(GaussianWeightedFeatureCoeffs5))
        .eval());

bool refine_mapping(int n_shift_only_iterations, int n_full_mapping_iterations, const FeaturePatch& p1,
                    const DescriptorImage& p2, const DescriptorImage& gxp2, const DescriptorImage& gyp2, Matrix2T& map,
                    Vector2T& xy) {
  Matrix2T map_ = map;
  Vector2T xy_ = xy;

  const int n_iterations = n_shift_only_iterations + n_full_mapping_iterations;
  for (int i = 0; i < n_iterations; ++i) {
    const bool full_mapping = i >= n_shift_only_iterations;

    SmallPatch sp1;
    compute_patch(sp1, p1);

    SmallPatch sp2;
    if (!compute_patch(sp2, p2, map_, xy_)) {
      return false;
    }

    SmallPatch gx;
    if (!compute_patch(gx, gxp2, map_, xy_)) {
      return false;
    }

    SmallPatch gy;
    if (!compute_patch(gy, gyp2, map_, xy_)) {
      return false;
    }

    // Account for the fact that we subtract difference of means:
    // gx and gy are the entries of the jacobian matrix of the residual.
    gx.array() -= gx.sum() / static_cast<float>(gx.size());
    gy.array() -= gy.sum() / static_cast<float>(gy.size());

    const SmallPatch gxgx = gx.cwiseProduct(gx);
    const SmallPatch gxgy = gx.cwiseProduct(gy);
    const SmallPatch gygy = gy.cwiseProduct(gy);

    SmallPatch residual;
    compute_residual(residual, sp2, sp1);

    const float w__gxgx = gaussian_weights(gxgx);
    const float w__gxgy = gaussian_weights(gxgy);
    const float w__gygy = gaussian_weights(gygy);

    const float wr_gx = gaussian_weights(residual.cwiseProduct(gx));
    const float wr_gy = gaussian_weights(residual.cwiseProduct(gy));

    if (full_mapping) {
      const float wx_gxgx = gaussian_weights(x_patch.cwiseProduct(gxgx));
      const float wx_gxgy = gaussian_weights(x_patch.cwiseProduct(gxgy));
      const float wx_gygy = gaussian_weights(x_patch.cwiseProduct(gygy));

      const float wy_gxgx = gaussian_weights(y_patch.cwiseProduct(gxgx));
      const float wy_gxgy = gaussian_weights(y_patch.cwiseProduct(gxgy));
      const float wy_gygy = gaussian_weights(y_patch.cwiseProduct(gygy));

      const float wxxgxgx = gaussian_weights(xx_patch.cwiseProduct(gxgx));
      const float wxxgxgy = gaussian_weights(xx_patch.cwiseProduct(gxgy));
      const float wxxgygy = gaussian_weights(xx_patch.cwiseProduct(gygy));

      const float wxygxgx = gaussian_weights(xy_patch.cwiseProduct(gxgx));
      const float wxygxgy = gaussian_weights(xy_patch.cwiseProduct(gxgy));
      const float wxygygy = gaussian_weights(xy_patch.cwiseProduct(gygy));

      const float wyygxgx = gaussian_weights(yy_patch.cwiseProduct(gxgx));
      const float wyygxgy = gaussian_weights(yy_patch.cwiseProduct(gxgy));
      const float wyygygy = gaussian_weights(yy_patch.cwiseProduct(gygy));

      const float wrxgx = gaussian_weights(residual.cwiseProduct(x_patch.cwiseProduct(gx)));
      const float wrxgy = gaussian_weights(residual.cwiseProduct(x_patch.cwiseProduct(gy)));
      const float wrygx = gaussian_weights(residual.cwiseProduct(y_patch.cwiseProduct(gx)));
      const float wrygy = gaussian_weights(residual.cwiseProduct(y_patch.cwiseProduct(gy)));

      Matrix6T t;
      t << wxxgxgx, wxxgxgy, wxygxgx, wxygxgy, wx_gxgx, wx_gxgy, wxxgxgy, wxxgygy, wxygxgy, wxygygy, wx_gxgy, wx_gygy,
          wxygxgx, wxygxgy, wyygxgx, wyygxgy, wy_gxgx, wy_gxgy, wxygxgy, wxygygy, wyygxgy, wyygygy, wy_gxgy, wy_gygy,
          wx_gxgx, wx_gxgy, wy_gxgx, wy_gxgy, w__gxgx, w__gxgy, wx_gxgy, wx_gygy, wy_gxgy, wy_gygy, w__gxgy, w__gygy;

      Vector6T a;
      a << wrxgx, wrxgy, wrygx, wrygy, wr_gx, wr_gy;

      const Vector6T z = t.colPivHouseholderQr().solve(a);
      if (z.array().isNaN().any()) {
        return false;
      }

      const float& dxx = z(0);
      const float& dyx = z(1);
      const float& dxy = z(2);
      const float& dyy = z(3);
      const float& dx = z(4);
      const float& dy = z(5);

      Matrix2T z_map;
      z_map << dxx + 1.f, dxy, dyx, dyy + 1.f;

      const Vector2T z_xy(dx, dy);

      map_ *= z_map;

      xy_ += map_ * z_xy;
    } else {
      Matrix2T t;
      t << w__gxgx, w__gxgy, w__gxgy, w__gygy;

      const Vector2T a{wr_gx, wr_gy};

      const Vector2T z = t.colPivHouseholderQr().solve(a);
      if (z.array().isNaN().any()) {
        return false;
      }

      const Vector2T& z_xy = z;

      xy_ += map_ * z_xy;
    }
  }

  map = map_;
  xy = xy_;
  return true;
}  // refine_mapping

bool track(int n_shift_only_iterations, int n_full_mapping_iterations, const STDescriptor& fd0, const STDescriptor& fd1,
           Matrix2T& map, Vector2T& xy) {
  std::vector<FeaturePatch> gxfd1;
  compute_gx(gxfd1, fd1);

  std::vector<FeaturePatch> gyfd1;
  compute_gy(gyfd1, fd1);

  const int size = fd0.image_patches_size;

  Matrix2T map_ = map;
  Vector2T xy_ = scale_xy(xy, 0, size);

  for (int level = size - 1; level >= 0; --level) {
    xy_ = scale_xy(xy_, level + 1, level);
    if (has_level(fd0, level)) {
      const DescriptorImage i1(fd1, level);
      const DescriptorImage gxi1(fd1.levels_mask, fd1.image_patches_size, &gxfd1[0], level);
      const DescriptorImage gyi1(fd1.levels_mask, fd1.image_patches_size, &gyfd1[0], level);
      refine_mapping(n_shift_only_iterations, n_full_mapping_iterations, fd0.image_patches[level], i1, gxi1, gyi1, map_,
                     xy_);
    }
  }

  map = map_;
  xy = xy_;
  return true;
}

void center_patch(SmallPatch& patch) { patch -= SmallPatch::Constant(patch.sum() / patch.size()); }

float ncc_direct(const STDescriptor& fd0, const STDescriptor& fd1) {
  const int size = std::min(fd0.image_patches_size, fd1.image_patches_size);

  float sum00 = 0;
  float sum01 = 0;
  float sum11 = 0;
  for (int level = size - 1; level >= 0; --level) {
    if (has_level(fd0, level) && has_level(fd1, level)) {
      SmallPatch p0;
      compute_patch(p0, fd0.image_patches[level]);
      center_patch(p0);

      SmallPatch p1;
      compute_patch(p1, fd1.image_patches[level]);
      center_patch(p1);

      sum00 += dot(p0, p0);
      sum01 += dot(p0, p1);
      sum11 += dot(p1, p1);
    }
  }

  if (sum00 == 0 || sum11 == 0) {
    return 0;
  }

  // compute NCC
  return sum01 / std::sqrt(sum00 * sum11);
}

float ncc_shi_tomasi(int n_shift_only_iterations, int n_full_mapping_iterations, const STDescriptor& fd0,
                     const STDescriptor& fd1) {
  Matrix2T map = Matrix2T::Identity();
  Vector2T xy = Vector2T::Zero();
  if (!track(n_shift_only_iterations, n_full_mapping_iterations, fd0, fd1, map, xy)) {
    return -1;
  }

  const int size = fd0.image_patches_size;

  float sum00 = 0;
  float sum01 = 0;
  float sum11 = 0;
  for (int level = size - 1; level >= 0; --level) {
    if (has_level(fd0, level)) {
      SmallPatch p0;
      compute_patch(p0, fd0.image_patches[level]);
      center_patch(p0);

      const DescriptorImage i1(fd1, level);
      SmallPatch p1;
      if (compute_patch(p1, i1, map, scale_xy(xy, 0, level))) {
        center_patch(p1);

        sum00 += dot(p0, p0);
        sum01 += dot(p0, p1);
        sum11 += dot(p1, p1);
      }
    }
  }

  if (sum00 == 0 || sum11 == 0) {
    return 0;
  }

  // compute NCC
  return sum01 / std::sqrt(sum00 * sum11);
}

float ncc(int n_shift_only_iterations, int n_full_mapping_iterations, FeatureDescriptorRef fd0,
          FeatureDescriptorRef fd1) {
  const STDescriptor* st_fd0 = static_cast<const STDescriptor*>(fd0.get());
  if (st_fd0 == nullptr) {
    return -1;
  }

  const STDescriptor* st_fd1 = static_cast<const STDescriptor*>(fd1.get());
  if (st_fd1 == nullptr) {
    return -1;
  }

  return std::max(ncc_direct(*st_fd0, *st_fd1),
                  ncc_shi_tomasi(n_shift_only_iterations, n_full_mapping_iterations, *st_fd0, *st_fd1));
}

}  // namespace cuvslam::slam::merge_details
