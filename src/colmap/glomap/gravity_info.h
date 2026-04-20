#pragma once

#include <Eigen/Core>

namespace colmap::glomap {

// Alignment rotation (second column = gravity direction) computed via
// Householder QR decomposition.
Eigen::Matrix3d GetAlignRot(const Eigen::Vector3d& gravity);

struct GravityInfo {
  // Whether the gravity information is available.
  bool has_gravity = false;

  const Eigen::Matrix3d& GetRAlign() const { return R_align; }

  void SetGravity(const Eigen::Vector3d& g);

  Eigen::Vector3d GetGravity() const { return gravity; }

 private:
  // Direction of the gravity.
  Eigen::Vector3d gravity;

  // Alignment matrix; the second column is the gravity direction.
  Eigen::Matrix3d R_align;
};

}  // namespace colmap::glomap
