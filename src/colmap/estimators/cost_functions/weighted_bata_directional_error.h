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

struct WeightedBATADirectionalError {
  WeightedBATADirectionalError(const Eigen::Vector3d& translation_obs,
                               const Eigen::Quaterniond& rotation_in,
                               double sigma_x_in,
                               double sigma_y_in,
                               double sigma_z_in)
      : translation_obs_(translation_obs),
        rotation_(rotation_in),
        inv_sigma_x_(1.0 / sigma_x_in),
        inv_sigma_y_(1.0 / sigma_y_in),
        inv_sigma_z_(1.0 / sigma_z_in) {
    CHECK_GT(sigma_x_in, 0.0);
    CHECK_GT(sigma_y_in, 0.0);
    CHECK_GT(sigma_z_in, 0.0);
  }

  template <typename T>
  bool operator()(const T* position1,
                  const T* position2,
                  const T* scale,
                  T* residuals) const {
    using Vec3T = Eigen::Matrix<T, 3, 1>;

    // Unweighted world-frame residual
    Vec3T r_world = translation_obs_.cast<T>() -
                    scale[0] * (Eigen::Map<const Vec3T>(position2) -
                                Eigen::Map<const Vec3T>(position1));

    // Rotate to camera frame using constant rotation
    Vec3T r_cam = rotation_.cast<T>() * r_world;

    // Apply anisotropic weighting
    residuals[0] = T(inv_sigma_x_) * r_cam[0];
    residuals[1] = T(inv_sigma_y_) * r_cam[1];
    residuals[2] = T(inv_sigma_z_) * r_cam[2];
    return true;
  }

  static ceres::CostFunction* Create(const Eigen::Vector3d& translation_obs,
                                     const Eigen::Quaterniond& rotation,
                                     double sigma_x,
                                     double sigma_y,
                                     double sigma_z) {
    if (sigma_x <= 0.0 || sigma_y <= 0.0 || sigma_z <= 0.0) {
      LOG(ERROR) << "Invalid sigmas for WeightedBATADirectionalError: "
                 << sigma_x << ", " << sigma_y << ", " << sigma_z;
      return nullptr;
    }
    return (new ceres::
                AutoDiffCostFunction<WeightedBATADirectionalError, 3, 3, 3, 1>(
                    new WeightedBATADirectionalError(
                        translation_obs, rotation, sigma_x, sigma_y, sigma_z)));
  }

  const Eigen::Vector3d translation_obs_;
  const Eigen::Quaterniond rotation_;
  const double inv_sigma_x_;
  const double inv_sigma_y_;
  const double inv_sigma_z_;
};

}  // namespace colmap
