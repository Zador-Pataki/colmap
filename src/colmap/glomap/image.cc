#include "colmap/glomap/image.h"

namespace colmap::glomap {

Eigen::Vector3d Image::Center() const {
  return cam_from_world.rotation().inverse() * -cam_from_world.translation();
}

}  // namespace colmap::glomap
