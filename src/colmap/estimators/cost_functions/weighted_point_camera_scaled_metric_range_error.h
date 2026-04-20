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

struct WeightedPointCameraScaledMetricRangeError {
  WeightedPointCameraScaledMetricRangeError(
      const Eigen::Vector3d& v_ik_in,
      double depth_prior_in,
      double feature_undist_z_in,
      const Eigen::Quaterniond& rotation_in,
      double sigma_x_in,
      double sigma_y_in,
      double sigma_z_in)
      : v_ik_(v_ik_in),
        depth_prior_(depth_prior_in),
        v_cam_z_(feature_undist_z_in),
        rotation_(rotation_in),
        inv_sigma_x_(1.0 / sigma_x_in),
        inv_sigma_y_(1.0 / sigma_y_in),
        inv_sigma_z_(1.0 / sigma_z_in) {
    CHECK_GT(v_cam_z_, 1e-9);
    CHECK_GT(sigma_x_in, 0.0);
    CHECK_GT(sigma_y_in, 0.0);
    CHECK_GT(sigma_z_in, 0.0);
  }

  template <typename T>
  bool operator()(const T* const c_i,
                  const T* const X_k,
                  const T* const d_ik,
                  T* residuals) const {
    using Vec3T = Eigen::Matrix<T, 3, 1>;
    const Vec3T camera_center(c_i[0], c_i[1], c_i[2]);
    const Vec3T point_3d(X_k[0], X_k[1], X_k[2]);
    const Vec3T point_vec = point_3d - camera_center;  // X_k - c_i
    const T range_prior = T(depth_prior_ / v_cam_z_);
    if (range_prior < T(1e-9)) {
      return false;
    }
    const T scale_factor = d_ik[0] / range_prior;  // d_ik / r_ik
    const Vec3T target_scaled_vec = point_vec * scale_factor;
    const Vec3T r_world = v_ik_.cast<T>() - target_scaled_vec;
    const Vec3T r_cam = rotation_.cast<T>() * r_world;

    residuals[0] = T(inv_sigma_x_) * r_cam[0];
    residuals[1] = T(inv_sigma_y_) * r_cam[1];
    residuals[2] = T(inv_sigma_z_) * r_cam[2];
    return true;
  }

  static ceres::CostFunction* Create(const Eigen::Vector3d& v_ik,
                                     double depth_prior,
                                     double feature_undist_z,
                                     const Eigen::Quaterniond& rotation,
                                     double sigma_x,
                                     double sigma_y,
                                     double sigma_z) {
    if (feature_undist_z <= 1e-9) {
      LOG(ERROR) << "WeightedPointCameraScaledMetricRangeError: bad z";
      return nullptr;
    }
    if (sigma_x <= 0.0 || sigma_y <= 0.0 || sigma_z <= 0.0) {
      LOG(ERROR) << "WeightedPointCameraScaledMetricRangeError: bad sigmas";
      return nullptr;
    }
    return new ceres::AutoDiffCostFunction<
        WeightedPointCameraScaledMetricRangeError,
        3,
        3,
        3,
        1>(new WeightedPointCameraScaledMetricRangeError(v_ik,
                                                         depth_prior,
                                                         feature_undist_z,
                                                         rotation,
                                                         sigma_x,
                                                         sigma_y,
                                                         sigma_z));
  }

 private:
  const Eigen::Vector3d v_ik_;
  const double depth_prior_;
  const double v_cam_z_;
  const Eigen::Quaterniond rotation_;
  const double inv_sigma_x_;
  const double inv_sigma_y_;
  const double inv_sigma_z_;
};

}  // namespace colmap
