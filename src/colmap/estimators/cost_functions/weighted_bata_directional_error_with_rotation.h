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

struct WeightedBATADirectionalErrorWithRotation {
  WeightedBATADirectionalErrorWithRotation(
      const Eigen::Vector3d& bearing_cam,
      double sigma_x_in,
      double sigma_y_in,
      double sigma_z_in)
      : bearing_cam_(bearing_cam),
        inv_sigma_x_(1.0 / sigma_x_in),
        inv_sigma_y_(1.0 / sigma_y_in),
        inv_sigma_z_(1.0 / sigma_z_in) {
    CHECK_GT(sigma_x_in, 0.0);
    CHECK_GT(sigma_y_in, 0.0);
    CHECK_GT(sigma_z_in, 0.0);
  }

  template <typename T>
  bool operator()(const T* rotation,  // Quaternion (4 params, XYZW Eigen order)
                  const T* position1,
                  const T* position2,
                  const T* scale,
                  T* residuals) const {
    using Vec3T = Eigen::Matrix<T, 3, 1>;

    // Compute world-frame difference (track - camera_center)
    Vec3T world_diff = Eigen::Map<const Vec3T>(position2) -
                       Eigen::Map<const Vec3T>(position1);

    // Transform to camera frame using rotation PARAMETER
    Vec3T cam_diff = Eigen::Map<const Eigen::Quaternion<T>>(rotation) * world_diff;

    // Residual in camera frame: bearing_cam - scale * cam_diff
    Vec3T r_cam = bearing_cam_.cast<T>() - scale[0] * cam_diff;

    // Apply anisotropic weighting
    residuals[0] = T(inv_sigma_x_) * r_cam[0];
    residuals[1] = T(inv_sigma_y_) * r_cam[1];
    residuals[2] = T(inv_sigma_z_) * r_cam[2];
    return true;
  }

  static ceres::CostFunction* Create(const Eigen::Vector3d& bearing_cam,
                                     double sigma_x,
                                     double sigma_y,
                                     double sigma_z) {
    if (sigma_x <= 0.0 || sigma_y <= 0.0 || sigma_z <= 0.0) {
      LOG(ERROR) << "Invalid sigmas for WeightedBATADirectionalErrorWithRotation: "
                 << sigma_x << ", " << sigma_y << ", " << sigma_z;
      return nullptr;
    }
    // Params: rotation(4), position1(3), position2(3), scale(1)
    return new ceres::AutoDiffCostFunction<
        WeightedBATADirectionalErrorWithRotation, 3, 4, 3, 3, 1>(
        new WeightedBATADirectionalErrorWithRotation(
            bearing_cam, sigma_x, sigma_y, sigma_z));
  }

  const Eigen::Vector3d bearing_cam_;
  const double inv_sigma_x_;
  const double inv_sigma_y_;
  const double inv_sigma_z_;
};

}  // namespace colmap
