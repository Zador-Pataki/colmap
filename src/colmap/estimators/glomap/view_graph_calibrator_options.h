#pragma once

#include "colmap/estimators/glomap/optimization_base_options.h"

namespace colmap::glomap {

struct ViewGraphCalibratorOptions {
  OptimizationBaseOptions solver_base;

  // Min/max ratio between estimated focal and prior focal.
  double thres_lower_ratio = 0.1;
  double thres_higher_ratio = 10.0;

  // Two-view error threshold in the per-pair problem.
  double thres_two_view_error = 2.0;

  ViewGraphCalibratorOptions() {
    solver_base.thres_loss_function = 1e-2;
  }
};

}  // namespace colmap::glomap
