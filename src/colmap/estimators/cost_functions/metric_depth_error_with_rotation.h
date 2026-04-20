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

struct MetricDepthErrorWithRotation {
  MetricDepthErrorWithRotation(
      double depth_prior_in,
      double sigma_depth_in,
      bool use_log_scale_in = false,
      bool use_log_residual_in = false,
      bool zero_residual_behind_in = false,
      bool smooth_transition_in = false,
      double threshold_in = 0.1)
      : depth_prior_(depth_prior_in),
        sigma_depth_(sigma_depth_in),
        use_log_scale_(use_log_scale_in),
        use_log_residual_(use_log_residual_in),
        zero_residual_behind_(zero_residual_behind_in),
        smooth_transition_(smooth_transition_in),
        threshold_(threshold_in) {
    if (sigma_depth_in <= 1e-9) {
      throw std::invalid_argument(
          "MetricDepthErrorWithRotation: Standard deviation must be positive.");
    }
  }

  template <typename T>
  bool operator()(
      const T* const rotation,    // Quaternion (4 params, XYZW Eigen order)
      const T* const c_i,         // Camera center (3 params)
      const T* const X_k,         // 3D point (3 params)
      const T* const dmap_scale,  // Image scale (1 param)
      T* residuals) const {
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> camera_center(c_i);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> point_3d(X_k);
    Eigen::Matrix<T, 3, 1> point_vec_world = point_3d - camera_center;

    // Transform vector to Camera Frame using rotation PARAMETER
    Eigen::Matrix<T, 3, 1> point_vec_cam =
        Eigen::Map<const Eigen::Quaternion<T>>(rotation) * point_vec_world;

    T z_est = point_vec_cam[2];

    // Convert scale: if log-space, exp(log_scale), else use directly
    T scale = use_log_scale_ ? ceres::exp(dmap_scale[0]) : dmap_scale[0];

    T scaled_prior = scale * T(depth_prior_);
    T scaled_sigma = scale * T(sigma_depth_);

    // Handle points behind camera
    if (zero_residual_behind_ && z_est <= T(0.0)) {
      residuals[0] = T(0.0);
      return true;
    }

    T r_depth;
    T weight;

    if (use_log_residual_) {
      T depth_prior_safe = std::max(T(depth_prior_), T(1e-6));
      T sigma_log = T(sigma_depth_) / depth_prior_safe;
      T weight_log = T(1.0) / std::max(T(1e-6), sigma_log);

      if (smooth_transition_) {
        T thresh = T(threshold_);
        if (z_est > thresh) {
          T z_est_safe = std::max(z_est, T(1e-6));
          T scaled_prior_safe = std::max(scaled_prior, T(1e-6));
          r_depth = ceres::log(z_est_safe / scaled_prior_safe);
          weight = weight_log;
        } else {
          T scaled_prior_safe = std::max(scaled_prior, T(1e-6));
          T r_at_threshold = ceres::log(thresh / scaled_prior_safe);
          r_depth = r_at_threshold + (z_est - thresh);
          weight = weight_log;
        }
      } else if (z_est > T(0.0)) {
        T z_est_safe = std::max(z_est, T(1e-6));
        T scaled_prior_safe = std::max(scaled_prior, T(1e-6));
        r_depth = ceres::log(z_est_safe / scaled_prior_safe);
        weight = weight_log;
      } else {
        r_depth = z_est - scaled_prior;
        weight = T(1.0) / std::max(T(1e-6), scaled_sigma);
      }
    } else {
      r_depth = z_est - scaled_prior;
      weight = T(1.0) / std::max(T(1e-6), scaled_sigma);
    }

    residuals[0] = weight * r_depth;
    return true;
  }

  static ceres::CostFunction* Create(
      double depth_prior,
      double sigma_depth,
      bool use_log_scale = false,
      bool use_log_residual = false,
      bool zero_residual_behind = false,
      bool smooth_transition = false,
      double threshold = 0.1) {
    if (sigma_depth <= 1e-9) {
      LOG(ERROR) << "Cannot create MetricDepthErrorWithRotation: "
                    "Standard deviation must be positive.";
      return nullptr;
    }
    // Residual: 1, Params: rotation(4), c_i(3), X_k(3), dmap_scale(1)
    return new ceres::AutoDiffCostFunction<MetricDepthErrorWithRotation,
                                           1, 4, 3, 3, 1>(
        new MetricDepthErrorWithRotation(depth_prior,
                                         sigma_depth,
                                         use_log_scale,
                                         use_log_residual,
                                         zero_residual_behind,
                                         smooth_transition,
                                         threshold));
  }

 private:
  const double depth_prior_;
  const double sigma_depth_;
  const bool use_log_scale_;
  const bool use_log_residual_;
  const bool zero_residual_behind_;
  const bool smooth_transition_;
  const double threshold_;
};

}  // namespace colmap
