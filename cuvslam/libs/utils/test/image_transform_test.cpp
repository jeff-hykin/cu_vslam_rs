
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

#include "utils/image_transform.h"
#include "common/include_gtest.h"

namespace test {

using namespace cuvslam;
using namespace cuvslam::utils;

template <ImageTransformType _Type>
class ImageTransformTest : public ::testing::Test {
protected:
  enum {
    TU14_LOW = 65,
    TU14_HIGH = 16383,  // also mask 0X3fff
    TU14_MASK = TU14_HIGH,
    TU8_LOW = 0,
    TU8_HIGH = 255,
  };

  using ImgTranU14 = _ImageTransform<float, _Type, 2, 255, TU14_LOW, TU14_HIGH>;
  using ImgTranU8 = _ImageTransform<float, _Type, 2, 255, TU8_LOW, TU8_HIGH>;
  using BaseU14 = std::uint16_t;
  using BaseU8 = std::uint8_t;

  struct SetU14Op {
    EIGEN_EMPTY_STRUCT_CTOR(SetU14Op)
    BaseU14 operator()(const BaseU14& a) const {
      const BaseU14 v = (a & TU14_MASK);
      return v < TU14_LOW ? v + TU14_LOW : v;
    }
  };

  std::vector<ImageMatrixT> data_;

  void SetUp() {
    ImageMatrix<BaseU8> img8bit(300, 300);
    img8bit.setRandom();
    assert(img8bit.minCoeff() == TU8_LOW && img8bit.maxCoeff() == TU8_HIGH);

    ImageMatrix<BaseU14> img14bit(3000, 3000);
    img14bit.noalias() = img14bit.setRandom().unaryExpr(SetU14Op());
    assert(img14bit.minCoeff() == TU14_LOW && img14bit.maxCoeff() == TU14_HIGH);

    data_.emplace_back(ImgTranU8::getSRGB(img8bit.template cast<float>()));
    data_.emplace_back(ImgTranU14::LinearToLog(img14bit.template cast<float>()));
  }
};

using ImageTransformTestGetSRGB = ImageTransformTest<ImageTransformType::getSRGB>;
using ImageTransformTestLinearToLog = ImageTransformTest<ImageTransformType::LinearToLog>;

TEST_F(ImageTransformTestGetSRGB, GetSRGBTest) {
  for (const auto& item : data_) {
    EXPECT_TRUE(item.minCoeff() >= 0.f && item.maxCoeff() <= 255.f);
  }
}

TEST_F(ImageTransformTestLinearToLog, DISABLED_LinearToLogTest) {
  for (const auto& item : data_) {
    EXPECT_TRUE(item.minCoeff() >= 0.f && item.maxCoeff() <= 255.f);
  }
}

}  // namespace test
