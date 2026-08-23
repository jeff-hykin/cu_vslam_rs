
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

#include <thread>

#include "common/include_gtest.h"
#include "common/isometry.h"
#include "common/thread_safe_queue.h"

namespace test::slam {

TEST(SlamTest, thread_safe_queue) {
  cuvslam::ThreadSafeQueue<cuvslam::Isometry3T> queue;
  cuvslam::ThreadSafeQueue<cuvslam::Isometry3T> queue_return;

  cuvslam::Isometry3T m = cuvslam::Isometry3T::Identity();
  auto run = [&]() {
    cuvslam::Isometry3T t;
    while (queue.WaitPop(t)) {
      m = m * t;
      queue_return.Push(m);
    }
  };
  auto thread = std::thread(run);

  for (int i = 0; i < 10000; i++) {
    queue.Push(cuvslam::Isometry3T::Identity());

    cuvslam::Isometry3T ret;
    if (queue_return.TryPop(ret)) {
    }
  }

  // Stop queue
  queue.Abort();
  if (thread.joinable()) {
    thread.join();
  }
}

}  // namespace test::slam
