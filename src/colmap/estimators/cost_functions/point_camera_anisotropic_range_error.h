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

struct PointCameraAnisotropicRangeError {
  PointCameraAnisotropicRangeError(
      const Eigen::Vector3d& v_ik_in,  // World viewing direction (unit vector)
      double depth_prior_in,           // Z-depth prior (m_ik)
      double feature_undist_z_in,      // Z-component of v_cam (v_cam,z,ik)
      const Eigen::Quaterniond&
          rotation_in,    // Rotation R_W_C (constant, not optimized)
      double sigma_x_in,  // Std dev for X component in camera frame (angular)
      double sigma_y_in,  // Std dev for Y component in camera frame (angular)
      double sigma_z_in)  // Std dev for Z component (depth/radial)
      : v_ik_(v_ik_in),
        depth_prior_(depth_prior_in),
        v_cam_z_(feature_undist_z_in),
        rotation_(rotation_in),
        inv_sigma_x_(1.0 / sigma_x_in),
        inv_sigma_y_(1.0 / sigma_y_in),
        inv_sigma_z_(1.0 / sigma_z_in) {
    // Ensure v_cam_z_ is positive
    if (v_cam_z_ <= 1e-9) {
      throw std::invalid_argument(
          "PointCameraAnisotropicRangeError: feature_undist Z-component must "
          "be positive.");
    }
    // Ensure sigmas are positive
    if (sigma_x_in <= 1e-9 || sigma_y_in <= 1e-9 || sigma_z_in <= 1e-9) {
      throw std::invalid_argument(
          "PointCameraAnisotropicRangeError: Standard deviations must be "
          "positive.");
    }
  }

  template <typename T>
  bool operator()(const T* const c_i,  // Camera center (3 parameters)
                  const T* const X_k,  // 3D point (3 parameters)
                  T* residuals) const {
    // --- Unitless residual r = v - (X - c) / range ---
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> camera_center(c_i);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> point_3d(X_k);
    const Eigen::Matrix<T, 3, 1> point_vec_world = point_3d - camera_center;
    const T prior_range = T(depth_prior_ / v_cam_z_);
    if (prior_range < T(1e-9)) {
      throw std::runtime_error(
          "PointCameraAnisotropicRangeError: Calculated prior range is near "
          "zero.");
    }
    const T inv_prior_range = T(1.0) / prior_range;
    const Eigen::Matrix<T, 3, 1> target_scaled_vec =
        point_vec_world * inv_prior_range;
    const Eigen::Matrix<T, 3, 1> r_world = v_ik_.cast<T>() - target_scaled_vec;

    // Transform residual to camera frame (still unitless)
    const Eigen::Matrix<T, 3, 1> r_cam = rotation_.cast<T>() * r_world;

    // --- Apply anisotropic whitening in camera frame (unitless residual) ---
    // X/Y: use angular stddevs (radians) directly: 1 / sigma_angle
    // Z: use relative depth stddev sigma_z / range -> weight = range / sigma_z
    const T wx = T(inv_sigma_x_);
    const T wy = T(inv_sigma_y_);
    const T wz = T(inv_sigma_z_) * prior_range;

    residuals[0] = wx * r_cam[0];
    residuals[1] = wy * r_cam[1];
    residuals[2] = wz * r_cam[2];

    return true;
  }

  // Factory function to create the cost function object
  static ceres::CostFunction* Create(
      const Eigen::Vector3d& v_ik,  // World view direction
      double depth_prior,           // Z-depth prior
      double feature_undist_z,      // Z of camera view vector
      const Eigen::Quaterniond&
          rotation,    // Rotation (constant, passed to constructor)
      double sigma_x,  // Std dev for X component (angular)
      double sigma_y,  // Std dev for Y component (angular)
      double sigma_z)  // Std dev for Z component (depth)
  {
    // Pre-check for validity before creating
    if (feature_undist_z <= 1e-9) {
      LOG(ERROR) << "Cannot create PointCameraAnisotropicRangeError: "
                    "feature_undist Z-component is non-positive ("
                 << feature_undist_z << ")";
      return nullptr;
    }
    // Use slightly larger epsilon for sigma checks to avoid issues near zero
    if (sigma_x <= 1e-9 || sigma_y <= 1e-9 || sigma_z <= 1e-9) {
      LOG(ERROR) << "Cannot create PointCameraAnisotropicRangeError: Standard "
                    "deviations must be positive. sigma_x="
                 << sigma_x << ", sigma_y=" << sigma_y
                 << ", sigma_z=" << sigma_z;
      return nullptr;
    }

    // Residual: 3, Params: c_i (3), X_k (3)
    // Rotation is constant (stored in object), not a parameter block
    return new ceres::
        AutoDiffCostFunction<PointCameraAnisotropicRangeError, 3, 3, 3>(
            new PointCameraAnisotropicRangeError(v_ik,
                                                 depth_prior,
                                                 feature_undist_z,
                                                 rotation,
                                                 sigma_x,
                                                 sigma_y,
                                                 sigma_z));
  }

 private:
  const Eigen::Vector3d
      v_ik_;                  // Observed world viewing direction (unit vector)
  const double depth_prior_;  // Input Z-depth prior (m_ik)
  const double v_cam_z_;      // Z-component of feature_undist (v_cam,z,ik)
  const Eigen::Quaterniond
      rotation_;              // Rotation R_W_C (constant, not optimized)
  const double inv_sigma_x_;  // Pre-computed 1/sigma_x for X component
  const double inv_sigma_y_;  // Pre-computed 1/sigma_y for Y component
  const double inv_sigma_z_;  // Pre-computed 1/sigma_z for Z component (depth)
};

}  // namespace colmap
