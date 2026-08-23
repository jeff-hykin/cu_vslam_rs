
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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cuvslam {

class IImageDropper {
public:
  virtual std::unordered_set<uint32_t> GetDroppedImages(double drop_rate, uint32_t num) = 0;
};

// will drop between floor(drop_rate*num) and ceil(drop_rate*num) images per frame, drop_rate*num on average
template <typename URNG>
class SteadyImageDropper : public IImageDropper {
public:
  explicit SteadyImageDropper(URNG&& gen) : gen_{std::move(gen)} {}

  std::unordered_set<uint32_t> GetDroppedImages(double drop_rate, uint32_t num) override {
    drop_rate = std::max(0., std::min(drop_rate, 1.));
    uint32_t num_dropped = std::floor(drop_rate * num);
    if (std::bernoulli_distribution{drop_rate * num - num_dropped}(gen_)) {
      num_dropped++;
    }
    std::vector<uint32_t> images(num);
    std::iota(images.begin(), images.end(), 0);
    std::shuffle(images.begin(), images.end(), gen_);
    std::unordered_set<uint32_t> dropped{images.begin(), images.begin() + num_dropped};
    return dropped;
  };

private:
  URNG gen_;
};

// will drop each image independently with drop_rate probability
template <typename URNG>
class NormalImageDropper : public IImageDropper {
public:
  explicit NormalImageDropper(URNG&& gen) : gen_{std::move(gen)} {}

  std::unordered_set<uint32_t> GetDroppedImages(double drop_rate, uint32_t num) override {
    drop_rate = std::max(0., std::min(drop_rate, 1.));
    std::unordered_set<uint32_t> dropped;
    std::bernoulli_distribution distr{drop_rate};
    for (uint32_t i = 0; i < num; i++) {
      if (distr(gen_)) {
        dropped.insert(i);
      }
    }
    return dropped;
  }

private:
  URNG gen_;
};

// Will drop images in sequences, imitating a channel problem.
// Dropped sequences will have random length with poisson distribution.
// drop_rate approximates an average probability that any given image is dropped,
// but not the probability of sequences.
// Will work correctly ONLY if drop_rate & num are always constant.
template <typename URNG>
class StickyImageDropper : public IImageDropper {
public:
  explicit StickyImageDropper(URNG&& gen, double avg_seq_len = 30.) : gen_{std::move(gen)}, avg_seq_len_{avg_seq_len} {}

  std::unordered_set<uint32_t> GetDroppedImages(double drop_rate, uint32_t num) override {
    drop_rate = std::max(0., std::min(drop_rate, 1.));
    std::unordered_set<uint32_t> dropped;
    // dropped sequence probability is such that effective drop rate is close to a passed drop_rate
    const auto drop_start_rate = drop_rate / (avg_seq_len_ * (1 - drop_rate) + drop_rate);
    std::bernoulli_distribution start_distr{drop_start_rate};
    std::poisson_distribution len_distr{avg_seq_len_};
    dropped_sequences_.reserve(num);  // to avoid rehashing & iterator invalidation
    for (uint32_t i = 0; i < num; i++) {
      auto seq = dropped_sequences_.find(i);
      if (seq == dropped_sequences_.end() && start_distr(gen_)) {
        seq = dropped_sequences_.emplace(i, len_distr(gen_)).first;
      }
      if (seq != dropped_sequences_.end()) {
        if (seq->second > 0) {
          dropped.insert(i);
          seq->second--;
        } else {
          dropped_sequences_.erase(seq);
        }
      }
    }
    return dropped;
  }

private:
  URNG gen_;
  const double avg_seq_len_;
  std::unordered_map<uint32_t, uint32_t> dropped_sequences_;  // key: index, value: drops left
};

enum class ImageDropperType { Steady, Normal, Sticky };

inline ImageDropperType ParseImageDropperType(const std::string& name) {
  if (name == "steady") return ImageDropperType::Steady;
  if (name == "normal") return ImageDropperType::Normal;
  if (name == "sticky") return ImageDropperType::Sticky;
  throw std::invalid_argument{"Unknown image dropper type: " + name};
}

template <typename URNG>
std::shared_ptr<IImageDropper> CreateImageDropper(ImageDropperType type, URNG&& gen) {
  switch (type) {
    case ImageDropperType::Steady:
      return std::make_shared<SteadyImageDropper<URNG>>(std::forward<URNG>(gen));
    case ImageDropperType::Normal:
      return std::make_shared<NormalImageDropper<URNG>>(std::forward<URNG>(gen));
    case ImageDropperType::Sticky:
      return std::make_shared<StickyImageDropper<URNG>>(std::forward<URNG>(gen));
  }
  return {};
}

}  // namespace cuvslam
