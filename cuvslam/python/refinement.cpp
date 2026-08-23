
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

#include "nanobind/nanobind.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/unordered_map.h"
#include "nanobind/stl/unordered_set.h"
#include "nanobind/stl/vector.h"

#include "refinement/bundle_adjustment_problem.h"
#include "refinement/refinement.h"

namespace nb = nanobind;

namespace cuvslam::refinement {

void init_refinement(nb::module_& m) {
  nb::class_<BundleAdjustmentProblem>(m, "BundleAdjustmentProblem")
      .def(nb::init<>())
      .def_rw("rig", &BundleAdjustmentProblem::rig)
      .def_rw("points_in_world", &BundleAdjustmentProblem::points_in_world)
      .def_rw("world_from_rigs", &BundleAdjustmentProblem::world_from_rigs)
      .def_rw("observations", &BundleAdjustmentProblem::observations)
      .def_rw("fixed_points", &BundleAdjustmentProblem::fixed_points)
      .def_rw("fixed_frames", &BundleAdjustmentProblem::fixed_frames);

  nb::class_<BundleAdjustmentProblemOptions>(m, "BundleAdjustmentProblemOptions")
      .def(nb::init<>())
      .def_rw("verbose", &BundleAdjustmentProblemOptions::verbose)
      .def_rw("export_iteration_state", &BundleAdjustmentProblemOptions::export_iteration_state)
      .def_rw("estimate_intrinsics", &BundleAdjustmentProblemOptions::estimate_intrinsics)
      .def_rw("use_loss_function", &BundleAdjustmentProblemOptions::use_loss_function)
      .def_rw("symmetric_focal_length", &BundleAdjustmentProblemOptions::symmetric_focal_length)
      .def_rw("max_reprojection_error", &BundleAdjustmentProblemOptions::max_reprojection_error)
      .def_rw("cauchy_loss_scale", &BundleAdjustmentProblemOptions::cauchy_loss_scale);
  // .def_rw("ceres_options", &BundleAdjustmentProblemOptions::ceres_options);

  nb::class_<BundleAdjustmentProblemSummary>(m, "BundleAdjustmentProblemSummary")
      .def(nb::init<>())
      .def("brief_report",
           [](const BundleAdjustmentProblemSummary& self) -> std::string { return self.ceres_summary.BriefReport(); })
      .def_rw("iteration_rigs_from_world", &BundleAdjustmentProblemSummary::iteration_rigs_from_world)
      .def_rw("iteration_points_in_world", &BundleAdjustmentProblemSummary::iteration_points_in_world)
      .def_rw("iteration_cameras", &BundleAdjustmentProblemSummary::iteration_cameras);

  m.def(
      "refine",
      [](const BundleAdjustmentProblem& problem, const BundleAdjustmentProblemOptions& options) {
        BundleAdjustmentProblem refined_problem;
        auto summary = refine(problem, options, refined_problem);
        return nb::make_tuple(refined_problem, summary);
      },
      nb::arg("problem"), nb::arg("options"),
      "Refines the bundle adjustment problem.\n\n"
      "Args:\n"
      "    problem: The bundle adjustment problem to refine\n"
      "    options: The options for the refinement\n"
      "Returns:\n"
      "    A tuple containing:\n"
      "        - The refined bundle adjustment problem\n"
      "        - The summary of the refinement");
}

}  // namespace cuvslam::refinement
