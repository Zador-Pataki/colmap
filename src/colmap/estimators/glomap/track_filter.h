#pragma once

#include "colmap/glomap/camera.h"
#include "colmap/glomap/image.h"
#include "colmap/glomap/image_pair.h"
#include "colmap/glomap/track.h"
#include "colmap/glomap/view_graph.h"

namespace colmap::glomap {

struct TrackFilter {
  static int FilterTracksByReprojection(
      const ViewGraph& view_graph,
      const std::unordered_map<camera_t, Camera>& cameras,
      const std::unordered_map<image_t, Image>& images,
      std::unordered_map<track_t, Track>& tracks,
      double max_reprojection_error = 1e-2,
      bool in_normalized_image = true);

  static int FilterTracksByAngle(
      const ViewGraph& view_graph,
      const std::unordered_map<camera_t, Camera>& cameras,
      const std::unordered_map<image_t, Image>& images,
      std::unordered_map<track_t, Track>& tracks,
      double max_angle_error = 1.);

  static int FilterTrackTriangulationAngle(
      const ViewGraph& view_graph,
      const std::unordered_map<image_t, Image>& images,
      std::unordered_map<track_t, Track>& tracks,
      double min_angle = 1.);
};

}  // namespace colmap::glomap
