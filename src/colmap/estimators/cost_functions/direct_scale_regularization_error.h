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

struct DirectScaleRegularizationError {
  DirectScaleRegularizationError(double sigma) : inv_sigma_(1.0 / sigma) {
    CHECK_GT(sigma, 0.0) << "Standard deviation must be positive";
  }

  template <typename T>
  bool operator()(const T* const scale, T* residuals) const {
    // Whitened residual: (scale - 1.0) / sigma
    residuals[0] = T(inv_sigma_) * (scale[0] - T(1.0));
    return true;
  }

  // Factory function to create the cost function object
  static ceres::CostFunction* Create(double sigma) {
    // Residual: 1 dimension, Parameter block: scale (1 dimension)
    return new ceres::
        AutoDiffCostFunction<DirectScaleRegularizationError, 1, 1>(
            new DirectScaleRegularizationError(sigma));
  }

 private:
  const double inv_sigma_;  // Pre-compute and store 1/sigma for efficiency
};

}  // namespace colmap
