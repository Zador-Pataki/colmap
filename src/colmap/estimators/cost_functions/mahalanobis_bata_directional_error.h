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

struct MahalanobisBATADirectionalError {
  MahalanobisBATADirectionalError(const Eigen::Vector3d& translation_obs,
                                  const Eigen::Quaterniond& rotation_in,
                                  double L00_in,
                                  double L10_in,
                                  double L11_in,
                                  double sigma_z_in)
      : translation_obs_(translation_obs),
        rotation_(rotation_in),
        L00_(L00_in),
        L10_(L10_in),
        L11_(L11_in),
        inv_sigma_z_(1.0 / sigma_z_in) {
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

    // Apply Mahalanobis weighting for XY using lower triangular Cholesky factor
    // L @ r_xy where L = [[L00, 0], [L10, L11]]
    residuals[0] = T(L00_) * r_cam[0];
    residuals[1] = T(L10_) * r_cam[0] + T(L11_) * r_cam[1];

    // Apply diagonal weighting for Z
    residuals[2] = T(inv_sigma_z_) * r_cam[2];
    return true;
  }

  static ceres::CostFunction* Create(const Eigen::Vector3d& translation_obs,
                                     const Eigen::Quaterniond& rotation,
                                     double L00,
                                     double L10,
                                     double L11,
                                     double sigma_z) {
    if (sigma_z <= 0.0) {
      LOG(ERROR) << "Invalid sigma_z for MahalanobisBATADirectionalError: "
                 << sigma_z;
      return nullptr;
    }
    return (
        new ceres::
            AutoDiffCostFunction<MahalanobisBATADirectionalError, 3, 3, 3, 1>(
                new MahalanobisBATADirectionalError(
                    translation_obs, rotation, L00, L10, L11, sigma_z)));
  }

  const Eigen::Vector3d translation_obs_;
  const Eigen::Quaterniond rotation_;
  const double L00_;
  const double L10_;
  const double L11_;
  const double inv_sigma_z_;
};

}  // namespace colmap
