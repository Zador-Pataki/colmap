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

struct FocalLengthRandomWalkError {
  FocalLengthRandomWalkError(double variance_per_frame, int frame_gap)
      : inv_std_(1.0 / std::sqrt(variance_per_frame * frame_gap)) {}

  template <typename T>
  bool operator()(const T* const focal_prev,
                  const T* const focal_next,
                  T* residual) const {
    residual[0] = T(inv_std_) * (focal_next[0] - focal_prev[0]);
    return true;
  }

  static ceres::CostFunction* Create(double variance_per_frame, int frame_gap) {
    return new ceres::AutoDiffCostFunction<FocalLengthRandomWalkError, 1, 1, 1>(
        new FocalLengthRandomWalkError(variance_per_frame, frame_gap));
  }

 private:
  const double inv_std_;
};

}  // namespace colmap
