#pragma once

#include "colmap/glomap/types.h"

#include <utility>
#include <vector>

#include <Eigen/Core>

namespace colmap::glomap {

using Observation = std::pair<image_t, feature_t>;

struct Track {
  // Track id.
  track_t track_id;

  // Estimated 3D point.
  Eigen::Vector3d xyz = Eigen::Vector3d::Zero();

  // RGB color (currently unused).
  Eigen::Matrix<uint8_t, 3, 1> color =
      Eigen::Matrix<uint8_t, 3, 1>::Zero();

  // Whether the 3D point has been estimated.
  bool is_initialized = false;

  // (image_id, feature_id) pairs where the track is observed.
  std::vector<Observation> observations;

  // Observations that came specifically from loop closures.
  std::vector<Observation> lc_observations;
};

}  // namespace colmap::glomap
