// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.

#pragma once

#include <array>

#include <Eigen/Core>
#include <Eigen/SVD>

namespace colmap {

// Numerical epsilon used by the ported §06 cost functors. Matches fork's
// `glomap::EPS`.
constexpr double EPS = 1e-12;

// Helpers for Fetzer focal length costs (ported from fork
// install/glomap/glomap/estimators/cost_function.h:441-494).
inline Eigen::Vector4d fetzer_d(const Eigen::Vector3d& ai,
                                const Eigen::Vector3d& bi,
                                const Eigen::Vector3d& aj,
                                const Eigen::Vector3d& bj,
                                const int u,
                                const int v) {
  Eigen::Vector4d d;
  d.setZero();
  d(0) = ai(u) * aj(v) - ai(v) * aj(u);
  d(1) = ai(u) * bj(v) - ai(v) * bj(u);
  d(2) = bi(u) * aj(v) - bi(v) * aj(u);
  d(3) = bi(u) * bj(v) - bi(v) * bj(u);
  return d;
}

inline std::array<Eigen::Vector4d, 3> fetzer_ds(
    const Eigen::Matrix3d& i1_G_i0) {
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(
      i1_G_i0, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Vector3d s = svd.singularValues();

  Eigen::Vector3d v_0 = svd.matrixV().col(0);
  Eigen::Vector3d v_1 = svd.matrixV().col(1);

  Eigen::Vector3d u_0 = svd.matrixU().col(0);
  Eigen::Vector3d u_1 = svd.matrixU().col(1);

  Eigen::Vector3d ai =
      Eigen::Vector3d(s(0) * s(0) * (v_0(0) * v_0(0) + v_0(1) * v_0(1)),
                      s(0) * s(1) * (v_0(0) * v_1(0) + v_0(1) * v_1(1)),
                      s(1) * s(1) * (v_1(0) * v_1(0) + v_1(1) * v_1(1)));

  Eigen::Vector3d aj = Eigen::Vector3d(u_1(0) * u_1(0) + u_1(1) * u_1(1),
                                       -(u_0(0) * u_1(0) + u_0(1) * u_1(1)),
                                       u_0(0) * u_0(0) + u_0(1) * u_0(1));

  Eigen::Vector3d bi = Eigen::Vector3d(s(0) * s(0) * v_0(2) * v_0(2),
                                       s(0) * s(1) * v_0(2) * v_1(2),
                                       s(1) * s(1) * v_1(2) * v_1(2));

  Eigen::Vector3d bj =
      Eigen::Vector3d(u_1(2) * u_1(2), -(u_0(2) * u_1(2)), u_0(2) * u_0(2));

  Eigen::Vector4d d_01 = fetzer_d(ai, bi, aj, bj, 1, 0);
  Eigen::Vector4d d_02 = fetzer_d(ai, bi, aj, bj, 0, 2);
  Eigen::Vector4d d_12 = fetzer_d(ai, bi, aj, bj, 2, 1);

  std::array<Eigen::Vector4d, 3> ds;
  ds[0] = d_01;
  ds[1] = d_02;
  ds[2] = d_12;
  return ds;
}

}  // namespace colmap
