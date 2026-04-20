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

struct MetricDepthError {
  MetricDepthError(
      const Eigen::Quaterniond& rotation_in,  // R_W_C (fixed)
      double depth_prior_in,                  // m_ik (meters)
      double sigma_depth_in,                  // sigma_m (meters)
      bool use_log_scale_in = false,  // If true, dmap_scale is log_scale
      bool use_log_residual_in =
          false,  // If true, use log-space residual for points in front
      bool zero_residual_behind_in =
          false,  // If true, set residual to 0 for points behind camera
      bool smooth_transition_in =
          false,  // If true, smooth log-to-linear transition at threshold
      double threshold_in = 0.1)  // Threshold for smooth transition
      : rotation_(rotation_in),
        depth_prior_(depth_prior_in),
        sigma_depth_(sigma_depth_in),
        use_log_scale_(use_log_scale_in),
        use_log_residual_(use_log_residual_in),
        zero_residual_behind_(zero_residual_behind_in),
        smooth_transition_(smooth_transition_in),
        threshold_(threshold_in) {
    if (sigma_depth_in <= 1e-9) {
      throw std::invalid_argument(
          "MetricDepthError: Standard deviation must be positive.");
    }
  }

  template <typename T>
  bool operator()(
      const T* const c_i,         // Camera center (3 params)
      const T* const X_k,         // 3D point (3 params)
      const T* const dmap_scale,  // Image scale (1 param) - linear or log_scale
      T* residuals) const {
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> camera_center(c_i);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> point_3d(X_k);
    Eigen::Matrix<T, 3, 1> point_vec_world =
        point_3d - camera_center;  // X_k - c_i
    // Transform vector to Camera Frame
    Eigen::Matrix<T, 3, 1> point_vec_cam =
        rotation_.cast<T>() * point_vec_world;
    // Get Estimated Z-Depth
    T z_est = point_vec_cam[2];

    // Convert scale: if log-space, exp(log_scale), else use directly
    T scale = use_log_scale_ ? ceres::exp(dmap_scale[0]) : dmap_scale[0];

    // Get Scaled Metric Prior
    T scaled_prior = scale * T(depth_prior_);

    // Scale the standard deviation by scale to maintain relative uncertainty
    T scaled_sigma = scale * T(sigma_depth_);

    // Handle points behind camera: set residual to zero if enabled
    if (zero_residual_behind_ && z_est <= T(0.0)) {
      residuals[0] = T(0.0);
      return true;
    }

    // Compute residual: use log-space for points in front if enabled, else
    // linear
    T r_depth;
    T weight;

    if (use_log_residual_) {
      // Compute relative uncertainty (used for log-space weighting)
      T depth_prior_safe = std::max(T(depth_prior_), T(1e-6));
      T sigma_log = T(sigma_depth_) / depth_prior_safe;
      T weight_log = T(1.0) / std::max(T(1e-6), sigma_log);

      if (smooth_transition_) {
        // Smooth log-to-linear transition at threshold
        static bool first_smooth_eval = true;
        if (first_smooth_eval && std::is_same<T, double>::value) {
          first_smooth_eval = false;
          LOG(INFO) << "MetricDepthError: smooth_log_linear_transition ACTIVE "
                    << "(threshold=" << threshold_ << ")";
        }
        T thresh = T(threshold_);
        if (z_est > thresh) {
          // Above threshold: use log-space residual
          T z_est_safe = std::max(z_est, T(1e-6));
          T scaled_prior_safe = std::max(scaled_prior, T(1e-6));
          r_depth = ceres::log(z_est_safe / scaled_prior_safe);
          weight = weight_log;
        } else {
          // At or below threshold: linear residual with gradient=1, continuous
          // at threshold Value at threshold: log(threshold / scaled_prior)
          // Linear continuation: r = log(threshold / scaled_prior) + (z -
          // threshold)
          T scaled_prior_safe = std::max(scaled_prior, T(1e-6));
          T r_at_threshold = ceres::log(thresh / scaled_prior_safe);
          r_depth = r_at_threshold + (z_est - thresh);
          weight = weight_log;  // Keep same weighting for continuity
        }
      } else if (z_est > T(0.0)) {
        // Original behavior: log-space for z > 0, linear for z <= 0
        T z_est_safe = std::max(z_est, T(1e-6));  // Numerical safety
        T scaled_prior_safe = std::max(scaled_prior, T(1e-6));
        T r_relative = ceres::log(z_est_safe / scaled_prior_safe);
        weight = weight_log;
        r_depth = r_relative;
      } else {
        // Point behind camera: use linear residual
        r_depth = z_est - scaled_prior;
        weight = T(1.0) / std::max(T(1e-6), scaled_sigma);
      }
    } else {
      // Log-space disabled: always use linear residual
      r_depth = z_est - scaled_prior;
      weight = T(1.0) / std::max(T(1e-6), scaled_sigma);
    }

    residuals[0] = weight * r_depth;  // Residual is 1D
    return true;
  }

  static ceres::CostFunction* Create(
      const Eigen::Quaterniond& rotation,  // Fixed R_W_C
      double depth_prior,                  // m_ik (meters)
      double sigma_depth,                  // sigma_m (meters)
      bool use_log_scale = false,          // If true, dmap_scale is log_scale
      bool use_log_residual =
          false,  // If true, use log-space residual for points in front
      bool zero_residual_behind =
          false,  // If true, set residual to 0 for points behind camera
      bool smooth_transition =
          false,  // If true, smooth log-to-linear transition at threshold
      double threshold = 0.1)  // Threshold for smooth transition
  {
    if (sigma_depth <= 1e-9) {
      LOG(ERROR) << "Cannot create MetricDepthError: Standard deviation must "
                    "be positive.";
      return nullptr;
    }
    // Residual: 1, Params: c_i (3), X_k (3), dmap_scale (1)
    return new ceres::AutoDiffCostFunction<MetricDepthError, 1, 3, 3, 1>(
        new MetricDepthError(rotation,
                             depth_prior,
                             sigma_depth,
                             use_log_scale,
                             use_log_residual,
                             zero_residual_behind,
                             smooth_transition,
                             threshold));
  }

 private:
  const Eigen::Quaterniond rotation_;  // R_W_C (fixed)
  const double depth_prior_;
  const double sigma_depth_;  // sigma_m (meters) - unscaled std dev
  const bool use_log_scale_;  // If true, dmap_scale parameter is log_scale
  const bool
      use_log_residual_;  // If true, use log-space residual for points in front
  const bool zero_residual_behind_;  // If true, set residual to 0 for points
                                     // behind camera
  const bool smooth_transition_;     // If true, smooth log-to-linear transition
  const double threshold_;           // Threshold for smooth transition
};

}  // namespace colmap
