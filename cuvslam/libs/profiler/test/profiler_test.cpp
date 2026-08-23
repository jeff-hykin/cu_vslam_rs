
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

#include <chrono>
#include <thread>

#include "profiler/profiler_enable.h"

auto incapsulation_domain_helper = cuvslam::profiler::DefaultProfiler::DomainHelper("CheckTraceIncapsulationTest");

void checkTraceIncapsulation() {
  TRACE_EVENT main_event = incapsulation_domain_helper.trace_event("checkTraceIncapsulation()");
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  TRACE_EVENT loop_event = incapsulation_domain_helper.trace_event("Loop 3");
  for (int i = 0; i < 3; i++) {
    TRACE_EVENT ev = incapsulation_domain_helper.trace_event("InternalTraceEvent 1");
    {
      TRACE_EVENT ev1 = incapsulation_domain_helper.trace_event("SubTraceEvent 0000FF", 0x0000FF);
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    {
      TRACE_EVENT ev1 = incapsulation_domain_helper.trace_event("SubTraceEvent 00FF00", 0x00FF00);
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  }
  loop_event.Pop();

  TRACE_EVENT no_loop_event = incapsulation_domain_helper.trace_event("No Loop");
  {
    TRACE_EVENT ev = incapsulation_domain_helper.trace_event("InternalTraceEvent 2");
    {
      TRACE_EVENT ev2 = incapsulation_domain_helper.trace_event("InternalTraceEvent FF0000", 0xFF0000);
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    {
      TRACE_EVENT ev2 = incapsulation_domain_helper.trace_event("InternalTraceEvent RANDOM");
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
  }
  no_loop_event.Pop();

  main_event.Pop();
}

int main() {
  using namespace cuvslam;
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  checkTraceIncapsulation();

  auto domain_helper = profiler::DefaultProfiler::DomainHelper("DomainHelperTest");

  TRACE_EVENT id3 = domain_helper.trace_event_start("HelperTraceEventStart/End Test1");
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  domain_helper.trace_event_end(id3);

  {
    TRACE_EVENT ev = domain_helper.trace_event("HelperTraceEventTest");
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  domain_helper.mark("DomainMarkTest");

  TRACE_EVENT id1 = profiler::DefaultProfiler::TraceEventStart("TraceEventStart/End Test");
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  profiler::DefaultProfiler::TraceEventEnd(id1);

  {
    TRACE_EVENT ev = profiler::DefaultProfiler::TraceEvent("TraceEventTest");
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  profiler::DefaultProfiler::Mark("MarkTest");
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  auto domain = profiler::DefaultProfiler::Domain("DomainTest");

  TRACE_EVENT id2 =
      profiler::DefaultProfiler::DomainTraceEventStart(domain.get_handle(), "DomainTraceEventStart/End Test1");
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  profiler::DefaultProfiler::DomainTraceEventEnd(domain.get_handle(), id2);

  {
    TRACE_EVENT ev = profiler::DefaultProfiler::DomainTraceEvent(domain.get_handle(), "DomainTraceEventTest");
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  profiler::DefaultProfiler::DomainMark(domain.get_handle(), "DomainMarkTest");
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
}
