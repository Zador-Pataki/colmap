#pragma once

#include <colmap/scene/camera.h>

#include <Eigen/Core>

namespace colmap::glomap {

// Composition (NOT inheritance). Callers reach the underlying colmap::Camera
// via the `.camera` field directly — no delegating getters.
struct Camera {
  Camera() = default;
  explicit Camera(const colmap::Camera& c) : camera(c) {}

  colmap::Camera camera;
  bool has_refined_focal_length = false;

  double Focal() const;
  Eigen::Vector2d PrincipalPoint() const;
  Eigen::Matrix3d GetK() const;
};

}  // namespace colmap::glomap
