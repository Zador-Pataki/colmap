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

struct RelativeTranslationError {
  // Constructor now takes the *original* covariance matrix
  RelativeTranslationError(const Eigen::Matrix3d& R_w1,
                           const Eigen::Vector3d& t_12_metric,
                           const Eigen::Matrix3d& cov_t)  // <-- CHANGED
      : R_w1_(R_w1),
        t_12_metric_(t_12_metric),
        Sigma_local_inv_(cov_t.inverse())  // <-- DO INVERSION HERE, ONCE
  {}

  // The operator() is IDENTICAL to before
  template <typename T>
  bool operator()(const T* const c1, const T* const c2, T* residual) const {
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> c1_vec(c1);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> c2_vec(c2);

    Eigen::Matrix<T, 3, 1> r_LC =
        (c2_vec - c1_vec) -
        R_w1_.template cast<T>() * t_12_metric_.template cast<T>();

    // Use the pre-inverted member variable
    Eigen::Matrix<T, 3, 3> Sigma_global_inv =
        R_w1_.template cast<T>() * Sigma_local_inv_.template cast<T>() *
        R_w1_.transpose().template cast<T>();

    Eigen::LLT<Eigen::Matrix<T, 3, 3>> llt(Sigma_global_inv);
    if (llt.info() == Eigen::NumericalIssue) {
      Eigen::Map<Eigen::Matrix<T, 3, 1>> residual_vec(residual);
      residual_vec.setZero();
      return true;
    }

    Eigen::Map<Eigen::Matrix<T, 3, 1>> residual_vec(residual);
    residual_vec = llt.matrixL().transpose() * r_LC;

    return true;
  }

  // Factory function also takes the original covariance matrix
  static ceres::CostFunction* Create(
      const Eigen::Matrix3d& R_w1,
      const Eigen::Vector3d& t_12_metric,
      const Eigen::Matrix3d& cov_t) {  // <-- CHANGED
    return (new ceres::AutoDiffCostFunction<RelativeTranslationError, 3, 3, 3>(
        // Pass cov_t directly to the constructor
        new RelativeTranslationError(R_w1, t_12_metric, cov_t)));
  }

 private:
  const Eigen::Matrix3d R_w1_;
  const Eigen::Vector3d t_12_metric_;
  // This now stores the *inverse*
  const Eigen::Matrix3d Sigma_local_inv_;
};

}  // namespace colmap
