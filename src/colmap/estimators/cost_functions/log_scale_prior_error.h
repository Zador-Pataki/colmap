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

struct LogScalePriorError {
  LogScalePriorError(double sigma_log)
      : inv_sigma_log_(1.0 / std::max(1e-6, sigma_log)) {
    if (sigma_log <= 1e-9) {
      throw std::invalid_argument(
          "LogScalePriorError: Standard deviation must be positive.");
    }
  }

  template <typename T>
  bool operator()(const T* const log_scale, T* residuals) const {
    // Prior is log(1.0) = 0.0, so residual is just log_scale / sigma_log
    residuals[0] = log_scale[0] * T(inv_sigma_log_);
    return true;
  }

  static ceres::CostFunction* Create(double sigma_log) {
    if (sigma_log <= 1e-9) {
      LOG(ERROR) << "Cannot create LogScalePriorError: Standard deviation must "
                    "be positive.";
      return nullptr;
    }
    // Residual: 1, Params: log_scale (1)
    return new ceres::AutoDiffCostFunction<LogScalePriorError, 1, 1>(
        new LogScalePriorError(sigma_log));
  }

 private:
  const double inv_sigma_log_;  // 1 / sigma_log
};

}  // namespace colmap
