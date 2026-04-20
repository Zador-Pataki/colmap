#pragma once

#include "colmap/estimators/glomap/optimization_base_options.h"

namespace colmap::glomap {

// Fork's BundleAdjusterOptions renamed to BundleAdjustmentOptions for naming
// consistency with other Options structs. mpsfm callers use this name in §17.
struct BundleAdjustmentOptions {
  OptimizationBaseOptions solver_base;

  bool optimize_rotations = true;
  bool optimize_translation = true;
  bool optimize_intrinsics = true;
  bool optimize_points = true;

  int min_num_view_per_track = 3;

  double thres_loss_function = 1.0;

  BundleAdjustmentOptions() {
    // Fork ctor sets 200 iterations + thres=1.0.
    solver_base.solver_options.max_num_iterations = 200;
    solver_base.thres_loss_function = 1.0;
  }
};

}  // namespace colmap::glomap
