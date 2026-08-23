
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

#include <iomanip>
#include <sstream>
#include <string>

#include "common/environment.h"
#include "common/frame_id.h"

namespace cuvslam::edex::filepath {

// return the base name of pathname path.
// Split the pathname path into a pair, (head, tail) where tail is the last pathname component and
// head is everything leading up to that.
inline std::string GetFilePath(const std::string& fileName) {
  return fileName.substr(0, fileName.find_last_of("/\\") + 1);
}

// return filename without extensions
inline std::string StripExt(const std::string& fileName) {
  size_t position = fileName.rfind(".");

  if (position != 0) {
    return fileName.substr(0, position);
  }

  return fileName;
}

// "/any/path/matchmove.ca038BgRaw.0004.jpg" ->
// prefix = "/any/path/matchmove.ca038BgRaw."
// frameId = 4
// ext = ".jpg"
inline bool SplitSequinceFileName(const std::string& fileName, std::string* pPrefix, FrameId* pFrameId,
                                  std::string* pExt, size_t* width = nullptr) {
  const size_t start_ext = fileName.find_last_of('.');

  if (start_ext == std::string::npos || start_ext == 0) {
    return false;
  }

  size_t start_frame_id = 0;
  const size_t optional_prefix_end = fileName.rfind('.', start_ext - 1);
  const size_t optional_folder_end = fileName.rfind('/', start_ext - 1);
  if (optional_prefix_end != std::string::npos) {
    start_frame_id = optional_prefix_end + 1;
  }
  if (optional_folder_end != std::string::npos) {
    start_frame_id = std::max(optional_folder_end + 1, start_frame_id);
  }

  const std::string frameIdStr = fileName.substr(start_frame_id, start_ext - start_frame_id);

  if (pFrameId != nullptr) {
    try {
      *pFrameId = stoi(frameIdStr);
    } catch (const std::invalid_argument&) {
      return false;
    }
  }
  if (width) {
    *width = start_ext - start_frame_id;
  }
  if (pPrefix != nullptr) {
    *pPrefix = fileName.substr(0, start_frame_id);
  }
  if (pExt != nullptr) {
    *pExt = fileName.substr(start_ext, std::string::npos);
  }
  return true;
}

inline bool ExtractFrameIdFromFileName(const std::string& fileName, FrameId& frameId) {
  return SplitSequinceFileName(fileName, NULL, &frameId, NULL);
}

inline std::string ZeroPadFrameId(const FrameId& frameId, size_t width = 4) {
  std::ostringstream ss;
  ss << std::setw(width) << std::setfill('0') << frameId;
  return ss.str();
}

inline bool ReplaceFrameIdInFileName(const std::string& fileName, const FrameId& replacedFrameId,
                                     std::string& replacedFileName) {
  std::string prefix;
  std::string ext;
  size_t width;

  if (!SplitSequinceFileName(fileName, &prefix, NULL, &ext, &width)) {
    return false;
  }

  replacedFileName = prefix + ZeroPadFrameId(replacedFrameId, width) + ext;
  return true;
}

}  // namespace cuvslam::edex::filepath
