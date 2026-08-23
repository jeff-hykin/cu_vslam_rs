
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

#include <array>
#include <cassert>
#include <string>

#include "common/log.h"

namespace cuvslam {

inline bool IsPathEndWithSlash(const std::string& path) {
  size_t len = path.length();

  if (len <= 1) {
    return false;
  }

  return path[len - 1] == '/' || path[len - 1] == '\\';
}

class Environment {
public:
  enum EnumVar { CUVSLAM_DATASETS = 0, CUVSLAM_OUTPUT, _COUNT };

  static const std::string& GetVar(const EnumVar i) {
    static Environment e;
    return e.array_[i];
  }

private:
#define SET_ITEM(e) setItem(e, #e)

  Environment() {
    SET_ITEM(CUVSLAM_DATASETS);
    SET_ITEM(CUVSLAM_OUTPUT);
  }

#undef SET_ITEM

  void setItem(const EnumVar i, const char* pname, const bool isPath = true) {
    const char* pvar = std::getenv(pname);
    TraceErrorIf(pvar == nullptr, "Missing Environment variable: %s", pname);
    if (pvar == nullptr) {
      return;
    }
    array_[i] = pvar;
    const bool checkPathVar = !isPath || IsPathEndWithSlash(array_[i]);
    TraceErrorIf(!checkPathVar, "Missing end slash for Path Environment variable: %s", pname);
    assert(checkPathVar);
  }

  std::array<std::string, _COUNT> array_;
};

}  // namespace cuvslam
