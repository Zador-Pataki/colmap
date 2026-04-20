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

class FetzerFocalLengthCost {
 public:
  FetzerFocalLengthCost(const Eigen::Matrix3d& i1_F_i0,
                        const Eigen::Vector2d& principal_point0,
                        const Eigen::Vector2d& principal_point1) {
    Eigen::Matrix3d K0 = Eigen::Matrix3d::Identity(3, 3);
    K0(0, 2) = principal_point0(0);
    K0(1, 2) = principal_point0(1);

    Eigen::Matrix3d K1 = Eigen::Matrix3d::Identity(3, 3);
    K1(0, 2) = principal_point1(0);
    K1(1, 2) = principal_point1(1);

    const Eigen::Matrix3d i1_G_i0 = K1.transpose() * i1_F_i0 * K0;

    const std::array<Eigen::Vector4d, 3> ds = fetzer_ds(i1_G_i0);

    d_01 = ds[0];
    d_02 = ds[1];
    d_12 = ds[2];
  }

  static ceres::CostFunction* Create(const Eigen::Matrix3d i1_F_i0,
                                     const Eigen::Vector2d& principal_point0,
                                     const Eigen::Vector2d& principal_point1) {
    return (new ceres::AutoDiffCostFunction<FetzerFocalLengthCost, 2, 1, 1>(
        new FetzerFocalLengthCost(
            i1_F_i0, principal_point0, principal_point1)));
  }

  template <typename T>
  bool operator()(const T* const fi_, const T* const fj_, T* residuals) const {
    const Eigen::Vector<T, 4> d_01_ = d_01.cast<T>();
    const Eigen::Vector<T, 4> d_12_ = d_12.cast<T>();

    const T fi = fi_[0];
    const T fj = fj_[0];

    T di = (fj * fj * d_01_(0) + d_01_(1));
    T dj = (fi * fi * d_12_(0) + d_12_(2));
    di = di == T(0) ? T(1e-6) : di;
    dj = dj == T(0) ? T(1e-6) : dj;

    const T K0_01 = -(fj * fj * d_01_(2) + d_01_(3)) / di;
    const T K1_12 = -(fi * fi * d_12_(1) + d_12_(3)) / dj;

    residuals[0] = (fi * fi - K0_01) / (fi * fi);
    residuals[1] = (fj * fj - K1_12) / (fj * fj);

    return true;
  }

 private:
  Eigen::Vector4d d_01;
  Eigen::Vector4d d_02;
  Eigen::Vector4d d_12;
};

}  // namespace colmap
