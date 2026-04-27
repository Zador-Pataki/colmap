#pragma once

#include "colmap/scene/camera.h"
#include "colmap/scene/correspondence_graph.h"
#include "colmap/scene/image.h"
#include "colmap/scene/point3d.h"
#include "colmap/sfm/track_establishment_glomap.h"  // track_t + feature_t typedefs
#include "colmap/util/types.h"  // image_t, camera_t, image_pair_t
#include <unordered_map>

namespace colmap {
namespace sfm_ext {

using ViewGraph = colmap::CorrespondenceGraph;
using ImagePair = colmap::CorrespondenceGraph::ImagePair;


struct TrackFilter {
  static int FilterTracksByAngle(
      ViewGraph& view_graph,
      const std::unordered_map<camera_t, Camera>& cameras,
      const std::unordered_map<image_t, Image>& images,
      std::unordered_map<track_t, Point3D>& tracks,
      double max_angle_error = 1.);

  static int FilterTrackTriangulationAngle(
      ViewGraph& view_graph,
      const std::unordered_map<image_t, Image>& images,
      std::unordered_map<track_t, Point3D>& tracks,
      double min_angle = 1.);
};

}  // namespace sfm_ext
}  // namespace colmap
