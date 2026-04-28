// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include "colmap/util/logging.h"

#include <stdexcept>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ceres/ceres.h>

namespace colmap {

// 1-D residual cost functor comparing camera-frame z-depth of a
// world point against ``s_i * m_ik`` where ``s_i`` is a per-image
// optimization variable (linear or log-space) and ``m_ik`` is a depth prior.
//
// Five toggles control residual shape (NOT mutually orthogonal —
// ``smooth_transition`` only takes effect inside the
// ``use_log_residual=true`` branch; reachable combinations: 12, not 32):
//   - ``use_log_scale`` — parameter is ``log(s)`` instead of linear ``s``.
//   - ``use_log_residual`` — residual is ``log(z/sp) / sigma_log`` instead
//     of ``(z - sp) / (s * sigma_m)`` for points in front.
//   - ``zero_residual_behind`` — force r=0 when ``z <= 0``.
//   - ``smooth_transition`` — C¹ blend between linear-below and log-above
//     ``threshold``. Only active when ``use_log_residual=true``.
//   - ``threshold`` — blend point for ``smooth_transition``.
//
// AutoDiff signature: ``<1, 3, 3, 1>`` on ``(c_i, X_k, dmap_scale)``.
// Rotation is baked in (constant ``R_W_C``).
struct MetricDepthError {
  MetricDepthError(const Eigen::Quaterniond& rotation,
                   double depth_prior,
                   double sigma_depth,
                   bool use_log_scale = false,
                   bool use_log_residual = false,
                   bool zero_residual_behind = false,
                   bool smooth_transition = false,
                   double threshold = 0.1)
      : rotation_(rotation),
        depth_prior_(depth_prior),
        sigma_depth_(sigma_depth),
        use_log_scale_(use_log_scale),
        use_log_residual_(use_log_residual),
        zero_residual_behind_(zero_residual_behind),
        smooth_transition_(smooth_transition),
        threshold_(threshold) {
    if (sigma_depth <= 1e-9) {
      throw std::invalid_argument(
          "MetricDepthError: Standard deviation must be positive.");
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

    const T scale = use_log_scale_ ? ceres::exp(dmap_scale[0]) : dmap_scale[0];
    const T scaled_prior = scale * T(depth_prior_);
    const T scaled_sigma = scale * T(sigma_depth_);

    if (zero_residual_behind_ && z_est <= T(0.0)) {
      residuals[0] = T(0.0);
      return true;
    }

    T r_depth;
    T weight;

    if (use_log_residual_) {
      const T depth_prior_safe = std::max(T(depth_prior_), T(1e-6));
      const T sigma_log = T(sigma_depth_) / depth_prior_safe;
      const T weight_log = T(1.0) / std::max(T(1e-6), sigma_log);

      if (smooth_transition_) {
        const T thresh = T(threshold_);
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

  static ceres::CostFunction* Create(const Eigen::Quaterniond& rotation,
                                     double depth_prior,
                                     double sigma_depth,
                                     bool use_log_scale = false,
                                     bool use_log_residual = false,
                                     bool zero_residual_behind = false,
                                     bool smooth_transition = false,
                                     double threshold = 0.1) {
    if (sigma_depth <= 1e-9) {
      LOG(ERROR) << "Cannot create MetricDepthError: Standard deviation must "
                    "be positive.";
      return nullptr;
    }
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
  const Eigen::Quaterniond rotation_;
  const double depth_prior_;
  const double sigma_depth_;
  const bool use_log_scale_;
  const bool use_log_residual_;
  const bool zero_residual_behind_;
  const bool smooth_transition_;
  const double threshold_;
};

// ScalePriorError + LogScalePriorError were dropped in favor of native
// CovarianceWeightedCostFunctor<NormalPriorCostFunctor<1>>::Create(cov, prior)
// in cost_functions/utils.h. See the call site in global_positioning.cc
// where the linear/log branch collapses to a single Create() with
// prior=1.0 (linear) or 0.0 (log) and cov=stddev^2.

}  // namespace colmap
