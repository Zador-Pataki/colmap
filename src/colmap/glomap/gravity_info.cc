#include "colmap/glomap/gravity_info.h"

#include <Eigen/QR>

namespace colmap::glomap {

Eigen::Matrix3d GetAlignRot(const Eigen::Vector3d& gravity) {
  Eigen::Matrix3d R;
  const Eigen::Vector3d v = gravity.normalized();
  R.col(1) = v;

  const Eigen::Matrix3d Q = v.householderQr().householderQ();
  const Eigen::Matrix<double, 3, 2> N = Q.rightCols(2);
  R.col(0) = N.col(0);
  R.col(2) = N.col(1);
  if (R.determinant() < 0) {
    R.col(2) = -R.col(2);
  }
  return R;
}

void GravityInfo::SetGravity(const Eigen::Vector3d& g) {
  gravity = g;
  R_align = GetAlignRot(g);
  has_gravity = true;
}

}  // namespace colmap::glomap
