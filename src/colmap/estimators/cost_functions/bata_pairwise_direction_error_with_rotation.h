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

struct BATAPairwiseDirectionErrorWithRotation {
  BATAPairwiseDirectionErrorWithRotation(const Eigen::Vector3d& bearing_cam)
      : bearing_cam_(bearing_cam) {}

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

    residuals[0] = r_cam[0];
    residuals[1] = r_cam[1];
    residuals[2] = r_cam[2];
    return true;
  }

  static ceres::CostFunction* Create(const Eigen::Vector3d& bearing_cam) {
    // Params: rotation(4), position1(3), position2(3), scale(1)
    return new ceres::AutoDiffCostFunction<
        BATAPairwiseDirectionErrorWithRotation, 3, 4, 3, 3, 1>(
        new BATAPairwiseDirectionErrorWithRotation(bearing_cam));
  }

  const Eigen::Vector3d bearing_cam_;
};

}  // namespace colmap
