#pragma once

#include "colmap/util/logging.h"

#include <algorithm>
#include <stdexcept>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ceres/ceres.h>

namespace colmap {

enum class MetricDepthResidualType { kLinear, kLog, kLogLinear };

struct MetricDepthOptions {
  bool use_log_scale = false;
  MetricDepthResidualType residual_type = MetricDepthResidualType::kLinear;
  bool zero_residual_behind = false;
  double log_linear_threshold = 0.1;

  bool IsValid() const {
    return residual_type != MetricDepthResidualType::kLogLinear ||
           log_linear_threshold > 0.0;
  }
};

// 1-D residual: camera-frame z-depth vs scaled depth prior (s_i * m_ik).
// AutoDiff<1, 3, 3, 1> over (frame_center, point3D, dmap_scale).
struct MetricDepthError {
  MetricDepthError(const Eigen::Quaterniond& rotation,
                   double depth_prior,
                   double sigma_depth,
                   const MetricDepthOptions& options = MetricDepthOptions())
      : rotation_(rotation),
        depth_prior_(depth_prior),
        sigma_depth_(sigma_depth),
        options_(options) {
    if (sigma_depth <= 1e-9) {
      throw std::invalid_argument(
          "MetricDepthError: Standard deviation must be positive.");
    }
    if (!options.IsValid()) {
      throw std::invalid_argument(
          "MetricDepthError: log-linear threshold must be positive.");
    }
  }

  template <typename T>
  bool operator()(const T* const c_i,
                  const T* const X_k,
                  const T* const dmap_scale,
                  T* residuals) const {
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> camera_center(c_i);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> point_3d(X_k);
    const Eigen::Matrix<T, 3, 1> point_vec_world = point_3d - camera_center;
    const Eigen::Matrix<T, 3, 1> point_vec_cam =
        rotation_.cast<T>() * point_vec_world;
    const T z_est = point_vec_cam[2];

    const T scale =
        options_.use_log_scale ? ceres::exp(dmap_scale[0]) : dmap_scale[0];
    const T scaled_prior = scale * T(depth_prior_);
    const T scaled_sigma = scale * T(sigma_depth_);

    if (options_.zero_residual_behind && z_est <= T(0.0)) {
      residuals[0] = T(0.0);
      return true;
    }

    T r_depth;
    T weight;

    if (options_.residual_type != MetricDepthResidualType::kLinear) {
      const T depth_prior_safe = std::max(T(depth_prior_), T(1e-6));
      const T sigma_log = T(sigma_depth_) / depth_prior_safe;
      const T weight_log = T(1.0) / std::max(T(1e-6), sigma_log);

      if (options_.residual_type == MetricDepthResidualType::kLogLinear) {
        const T thresh = T(options_.log_linear_threshold);
        const T scaled_prior_safe = std::max(scaled_prior, T(1e-6));
        if (z_est > thresh) {
          const T z_est_safe = std::max(z_est, T(1e-6));
          r_depth = ceres::log(z_est_safe / scaled_prior_safe);
        } else {
          // Linear continuation below threshold (C¹ at boundary):
          // d/dz log(z/p) = 1/z, so slope at threshold is 1/threshold.
          const T r_at_threshold = ceres::log(thresh / scaled_prior_safe);
          r_depth = r_at_threshold + (z_est - thresh) / thresh;
        }
        weight = weight_log;
      } else if (z_est > T(0.0)) {
        const T z_est_safe = std::max(z_est, T(1e-6));
        const T scaled_prior_safe = std::max(scaled_prior, T(1e-6));
        r_depth = ceres::log(z_est_safe / scaled_prior_safe);
        weight = weight_log;
      } else {
        // Point behind camera: linear residual (log undefined).
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
      const Eigen::Quaterniond& rotation,
      double depth_prior,
      double sigma_depth,
      const MetricDepthOptions& options = MetricDepthOptions()) {
    if (sigma_depth <= 1e-9) {
      LOG(ERROR) << "Cannot create MetricDepthError: Standard deviation must "
                    "be positive.";
      return nullptr;
    }
    if (!options.IsValid()) {
      LOG(ERROR)
          << "Cannot create MetricDepthError: log-linear threshold must be "
             "positive.";
      return nullptr;
    }
    return new ceres::AutoDiffCostFunction<MetricDepthError, 1, 3, 3, 1>(
        new MetricDepthError(rotation, depth_prior, sigma_depth, options));
  }

 private:
  const Eigen::Quaterniond rotation_;
  const double depth_prior_;
  const double sigma_depth_;
  const MetricDepthOptions options_;
};

// ScalePriorError + LogScalePriorError replaced by native
// CovarianceWeightedCostFunctor<NormalPriorCostFunctor<1>> (see
// global_positioning.cc): prior=1.0 (linear) or 0.0 (log), cov=stddev^2.

}  // namespace colmap
