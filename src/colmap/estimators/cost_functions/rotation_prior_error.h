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

struct RotationPriorError {
  RotationPriorError(const Eigen::Quaterniond& prior_rotation,
                     double sigma_rotation)
      : prior_rotation_(prior_rotation),
        inv_sigma_(1.0 / std::max(1e-9, sigma_rotation)) {
    if (sigma_rotation <= 1e-9) {
      throw std::invalid_argument(
          "RotationPriorError: Standard deviation must be positive.");
    }
  }

  template <typename T>
  bool operator()(const T* rotation, T* residuals) const {
    // Compute rotation difference: q_diff = q * q_prior^{-1}
    Eigen::Quaternion<T> q_current = Eigen::Map<const Eigen::Quaternion<T>>(rotation);
    Eigen::Quaternion<T> q_prior = prior_rotation_.cast<T>();
    Eigen::Quaternion<T> q_diff = q_current * q_prior.inverse();

    // Convert to angle-axis for residual
    T angle_axis[3];
    ceres::QuaternionToAngleAxis(q_diff.coeffs().data(), angle_axis);

    // Note: Ceres QuaternionToAngleAxis expects WXYZ order but Eigen uses XYZW
    // We need to reorder. Actually, let's use Ceres rotation functions directly.
    // The quaternion from EigenQuaternionMap is in XYZW (Eigen) order.
    // Ceres expects WXYZ. So we need to handle this.

    // Alternative: compute angle-axis manually for correctness
    // angle = 2 * acos(w), axis = (x,y,z) / sin(angle/2)
    // For small angles, angle_axis ≈ 2 * (x, y, z)
    T w = q_diff.w();
    T x = q_diff.x();
    T y = q_diff.y();
    T z = q_diff.z();

    // Normalize to ensure valid quaternion
    T norm = ceres::sqrt(w * w + x * x + y * y + z * z);
    w /= norm;
    x /= norm;
    y /= norm;
    z /= norm;

    // For small rotations: angle_axis ≈ 2 * (x, y, z)
    // This is correct for small angles and numerically stable
    T sin_half_angle_sq = x * x + y * y + z * z;

    // Use Taylor expansion for small angles to avoid numerical issues
    T scale;
    if (sin_half_angle_sq < T(1e-10)) {
      // Small angle: angle_axis = 2 * (x, y, z)
      scale = T(2.0);
    } else {
      T sin_half_angle = ceres::sqrt(sin_half_angle_sq);
      T cos_half_angle = w;
      // angle = 2 * atan2(sin_half_angle, cos_half_angle)
      T angle = T(2.0) * ceres::atan2(sin_half_angle, cos_half_angle);
      scale = angle / sin_half_angle;
    }

    residuals[0] = T(inv_sigma_) * scale * x;
    residuals[1] = T(inv_sigma_) * scale * y;
    residuals[2] = T(inv_sigma_) * scale * z;
    return true;
  }

  static ceres::CostFunction* Create(const Eigen::Quaterniond& prior_rotation,
                                     double sigma_rotation) {
    if (sigma_rotation <= 1e-9) {
      LOG(ERROR) << "Cannot create RotationPriorError: "
                    "Standard deviation must be positive.";
      return nullptr;
    }
    // Residual: 3 (angle-axis), Params: rotation(4)
    return new ceres::AutoDiffCostFunction<RotationPriorError, 3, 4>(
        new RotationPriorError(prior_rotation, sigma_rotation));
  }

 private:
  const Eigen::Quaterniond prior_rotation_;
  const double inv_sigma_;
};

}  // namespace colmap
