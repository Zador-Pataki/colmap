#pragma once

#include <thread>

#include <ceres/ceres.h>

namespace colmap::glomap {

// POD shared by GP/RA/BA/VGC option structs. Plan replaces fork's inheritance
// pattern (`struct X : public OptimizationBaseOptions`) with composition:
// downstream structs carry `OptimizationBaseOptions solver_base`.
struct OptimizationBaseOptions {
  // Threshold for the loss function (Huber / Cauchy scale, problem-dependent).
  double thres_loss_function = 1e-1;

  // Ceres solver options bundle.
  ceres::Solver::Options solver_options;

  OptimizationBaseOptions() {
    solver_options.num_threads = std::thread::hardware_concurrency();
    solver_options.max_num_iterations = 100;
    solver_options.minimizer_progress_to_stdout = false;
    solver_options.function_tolerance = 1e-5;
  }
};

}  // namespace colmap::glomap
