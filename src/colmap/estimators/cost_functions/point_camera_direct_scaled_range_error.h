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

struct PointCameraDirectScaledRangeError {
  PointCameraDirectScaledRangeError(
      const Eigen::Vector3d& v_ik_in,
      double depth_prior_in,
      double feature_undist_z_in)  // Z-component of v_cam (v_cam,z,ik)
      : v_ik_(v_ik_in),
        depth_prior_(depth_prior_in),
        v_cam_z_(feature_undist_z_in) {  // Store the z-component
    if (v_cam_z_ <= 1e-9) {
      throw std::invalid_argument(
          "feature_undist Z-component must be positive for range calculation.");
    }
  }

  template <typename T>
  bool operator()(const T* const c_i,  // Camera center (3 parameters)
                  const T* const X_k,  // 3D point (3 parameters)
                  const T* const d_i,  // Direct scale for image i (1 parameter)
                  T* residuals) const {
    // Calculate vector from camera center to 3D point: X_k - c_i
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> camera_center(c_i);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> point_3d(X_k);
    Eigen::Matrix<T, 3, 1> point_vec = point_3d - camera_center;  // X_k - c_i

    // Convert depth prior to range: m_ik / v_cam_z
    T range_prior = T(depth_prior_ / v_cam_z_);

    if (range_prior < T(1e-9)) {
      throw std::runtime_error("Range prior is near zero.");
    }

    // Calculate scale factor: d_i / range_prior
    T scale_factor = d_i[0] / range_prior;

    // Check for extreme values that might cause numerical issues
    if (ceres::abs(scale_factor) > T(1e6) ||
        ceres::abs(scale_factor) < T(1e-6)) {
      LOG(WARNING) << "Extreme scale factor: " << scale_factor
                   << " (d_i=" << d_i[0] << ", range_prior=" << range_prior
                   << ")";
    }

    // Calculate target scaled vector: (d_i / range_prior) * (X_k - c_i)
    Eigen::Matrix<T, 3, 1> target_scaled_vec = point_vec * scale_factor;

    // Residual = Observed_Direction - Target_Scaled_Vector
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals_vec(residuals);
    residuals_vec = v_ik_.cast<T>() - target_scaled_vec;

    return true;
  }

  static ceres::CostFunction* Create(
      const Eigen::Vector3d& v_ik,  // World view direction
      double depth_prior,           // Z-depth prior
      double feature_undist_z) {    // Z of camera view vector
    // Pre-check for validity before creating
    if (feature_undist_z <= 1e-9) {
      LOG(ERROR)
          << "Cannot create PointCameraDirectScaledRangeError: feature_undist "
             "Z-component is non-positive ("
          << feature_undist_z << ")";
      return nullptr;  // Return null if invalid input
    }
    return new ceres::
        AutoDiffCostFunction<PointCameraDirectScaledRangeError, 3, 3, 3, 1>(
            new PointCameraDirectScaledRangeError(
                v_ik, depth_prior, feature_undist_z));
  }

 private:
  const Eigen::Vector3d
      v_ik_;                  // Observed world viewing direction (unit vector)
  const double depth_prior_;  // Input depth prior (m_ik)
  const double v_cam_z_;      // Z-component of camera view vector (v_cam,z,ik)
};

}  // namespace colmap
