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

struct ScalePriorError {
  ScalePriorError(double prior, double stddev)
      : prior_(prior), weight_(1.0 / std::max(1e-6, stddev)) {
    if (stddev <= 1e-9) {
      throw std::invalid_argument(
          "ScalePriorError: Standard deviation must be positive.");
    }
  }

  template <typename T>
  bool operator()(const T* const scale, T* residuals) const {
    residuals[0] = (scale[0] - T(prior_)) * T(weight_);
    return true;
  }

  static ceres::CostFunction* Create(double prior, double stddev) {
    if (stddev <= 1e-9) {
      LOG(ERROR) << "Cannot create ScalePriorError: Standard deviation must "
                    "be positive.";
      return nullptr;
    }
    // Residual: 1, Params: scale (1)
    return new ceres::AutoDiffCostFunction<ScalePriorError, 1, 1>(
        new ScalePriorError(prior, stddev));
  }

 private:
  const double prior_;
  const double weight_;  // 1 / stddev
};

}  // namespace colmap
