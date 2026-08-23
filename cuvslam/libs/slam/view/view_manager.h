
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

#include <memory>
#include <mutex>
#include <vector>

namespace cuvslam::slam {

// VIEW class
//
template <class VIEW>
class ViewManager {
public:
  ViewManager();
  void init(int view_count, uint32_t view_arg);
  void reset();

  // VIEW class must have get_timestamp() method
  std::shared_ptr<VIEW> acquire_earliest();
  std::shared_ptr<VIEW> acquire_latest();

private:
  std::vector<std::shared_ptr<VIEW>> views_;
  std::mutex m_;
};

template <class VIEW>
ViewManager<VIEW>::ViewManager() {}

template <class VIEW>
void ViewManager<VIEW>::init(int view_count, uint32_t view_arg) {
  std::lock_guard<std::mutex> lock(m_);
  if (view_count != static_cast<int>(views_.size())) {
    views_.resize(view_count, nullptr);
    for (int i = 0; i < view_count; i++) {
      views_[i] = std::make_shared<VIEW>(view_arg);
    }
  }
}
template <class VIEW>
void ViewManager<VIEW>::reset() {
  std::lock_guard<std::mutex> lock(m_);
  views_.clear();
}

template <class VIEW>
std::shared_ptr<VIEW> ViewManager<VIEW>::acquire_earliest() {
  if (views_.empty()) {
    return nullptr;
  }
  std::lock_guard<std::mutex> lock(m_);
  std::shared_ptr<VIEW> min_view;
  uint64_t min_timestamp = 0;
  for (std::shared_ptr<VIEW> &x : views_) {
    if (x.use_count() == 1) {
      const uint64_t timestamp = x->get_timestamp();
      if (!min_view || timestamp < min_timestamp) {
        min_timestamp = timestamp;
        min_view = x;
      }
    }
  }
  return min_view;
}

template <class VIEW>
std::shared_ptr<VIEW> ViewManager<VIEW>::acquire_latest() {
  if (views_.empty()) {
    return nullptr;
  }
  std::lock_guard<std::mutex> lock(m_);
  std::shared_ptr<VIEW> max_view;
  uint64_t max_timestamp = 0;
  for (std::shared_ptr<VIEW> &x : views_) {
    if (x.use_count() == 1) {
      const uint64_t timestamp = x->get_timestamp();
      if (!max_view || timestamp > max_timestamp) {
        max_timestamp = timestamp;
        max_view = x;
      }
    }
  }
  return max_view;
}

}  // namespace cuvslam::slam
