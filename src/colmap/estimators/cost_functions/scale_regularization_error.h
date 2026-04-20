// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.

#pragma once

#include <colmap/sensor/models.h>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/LU>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace colmap {

struct ScaleRegularizationError {
  ScaleRegularizationError(double weight) : sqrt_weight_(std::sqrt(weight)) {}

  template <typename T>
  bool operator()(const T* const log_scale, T* residuals) const {
    residuals[0] = T(sqrt_weight_) * log_scale[0];
    return true;
  }

  // Factory function to create the cost function object
  static ceres::CostFunction* Create(double weight) {
    // Residual: 1 dimension, Parameter block: log_scale (1 dimension)
    return new ceres::AutoDiffCostFunction<ScaleRegularizationError, 1, 1>(
        new ScaleRegularizationError(weight));
  }

 private:
  const double sqrt_weight_;  // Pre-compute and store sqrt(weight)
};

}  // namespace colmap
