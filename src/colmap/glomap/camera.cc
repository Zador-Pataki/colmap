#include "colmap/glomap/camera.h"

namespace colmap::glomap {

double Camera::Focal() const {
  return (camera.FocalLengthX() + camera.FocalLengthY()) / 2.0;
}

Eigen::Vector2d Camera::PrincipalPoint() const {
  return Eigen::Vector2d(camera.PrincipalPointX(), camera.PrincipalPointY());
}

Eigen::Matrix3d Camera::GetK() const {
  Eigen::Matrix3d K;
  K << camera.FocalLengthX(), 0, camera.PrincipalPointX(),
      0, camera.FocalLengthY(), camera.PrincipalPointY(),
      0, 0, 1;
  return K;
}

}  // namespace colmap::glomap
