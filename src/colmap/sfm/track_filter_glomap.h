#pragma once

#include "colmap/scene/camera.h"
#include "colmap/scene/correspondence_graph.h"
#include "colmap/scene/image.h"
#include "colmap/sfm/track_establishment_glomap.h"  // Track + track_t + feature_t + Observation
#include "colmap/util/types.h"  // image_t, camera_t, image_pair_t
#include <unordered_map>

namespace colmap {
namespace glomap_ra {

using ViewGraph = colmap::CorrespondenceGraph;
using ImagePair = colmap::CorrespondenceGraph::ImagePair;


struct TrackFilter {
  static int FilterTracksByReprojection(
      ViewGraph& view_graph,
      const std::unordered_map<camera_t, Camera>& cameras,
      const std::unordered_map<image_t, Image>& images,
      std::unordered_map<track_t, Track>& tracks,
      double max_reprojection_error = 1e-2,
      bool in_normalized_image = true);

  static int FilterTracksByAngle(
      ViewGraph& view_graph,
      const std::unordered_map<camera_t, Camera>& cameras,
      const std::unordered_map<image_t, Image>& images,
      std::unordered_map<track_t, Track>& tracks,
      double max_angle_error = 1.);

  static int FilterTrackTriangulationAngle(
      ViewGraph& view_graph,
      const std::unordered_map<image_t, Image>& images,
      std::unordered_map<track_t, Track>& tracks,
      double min_angle = 1.);
};

}  // namespace glomap_ra
}  // namespace colmap
