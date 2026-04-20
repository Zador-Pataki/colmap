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

struct GravError {
  GravError(const Eigen::Vector3d& grav_obs) : grav_obs_(grav_obs) {}

  template <typename T>
  bool operator()(const T* const gvec, T* residuals) const {
    Eigen::Matrix<T, 3, 1> grav_est;
    grav_est << gvec[0], gvec[1], gvec[2];

    for (int i = 0; i < 3; i++) {
      residuals[i] = grav_est[i] - grav_obs_[i];
    }

    return true;
  }

  // Factory function
  static ceres::CostFunction* CreateCost(const Eigen::Vector3d& grav_obs) {
    return (new ceres::AutoDiffCostFunction<GravError, 3, 3>(
        new GravError(grav_obs)));
  }

 private:
  const Eigen::Vector3d& grav_obs_;
};

}  // namespace colmap
