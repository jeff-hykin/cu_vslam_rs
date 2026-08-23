
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

#include "common/error.h"
#include "common/include_gtest.h"

using namespace cuvslam;

namespace Test {

TEST(TestcuVSLAMErrorCodeTests, ErrorCodeStaticCheck) {
  static_assert((std::is_same<ErrorCode::ErrorNumericType, int>::value),
                "Static check that int is a base type of ErrorNumericType");
  static_assert(sizeof(ErrorCode::ErrorNumericType) == 4, "Static check that size of ErrorNumericType is 32 bits");
}

TEST(TestcuVSLAMErrorCodeTests, ErrorCodeDefault) {
  ErrorCode err;
  ErrorCode::Enum eVal = err;
  EXPECT_EQ(eVal, ErrorCode::E_Unexpected);
}

TEST(TestcuVSLAMErrorCodeTests, ErrorCodeErrors) {
  const ErrorCode::Enum errors[] = {ErrorCode::E_Abort,       ErrorCode::E_AccessDenied,    ErrorCode::E_Bounds,
                                    ErrorCode::E_Failure,     ErrorCode::E_InvalidArgument, ErrorCode::E_NotImplemented,
                                    ErrorCode::E_OutOfMemory, ErrorCode::E_Pending,         ErrorCode::E_Pointer,
                                    ErrorCode::E_Unexpected};

  for (auto& e : errors) {
    ErrorCode err;
    err = e;
    ErrorCode::Enum eVal = err;
    EXPECT_EQ(eVal, e);
    EXPECT_EQ(err ? 1 : 2, 2);
  }
}

TEST(TestcuVSLAMErrorCodeTests, ErrorCodeSuccess) {
  const ErrorCode::Enum errors[] = {ErrorCode::S_True, ErrorCode::S_False};

  for (auto& e : errors) {
    ErrorCode err;
    err = e;
    ErrorCode::Enum eVal = err;
    EXPECT_EQ(eVal, e);
    EXPECT_EQ(err ? 1 : 2, 1);
  }
}

}  // namespace Test
