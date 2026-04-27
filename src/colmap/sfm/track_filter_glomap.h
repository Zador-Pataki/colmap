#pragma once

#include "colmap/scene/camera.h"
#include "colmap/scene/correspondence_graph.h"
#include "colmap/scene/image.h"
#include "colmap/sfm/track_establishment_glomap.h"  // Track + track_t + feature_t + Observation
#include "colmap/util/types.h"  // image_t, camera_t, image_pair_t
#include <unordered_map>

// TODO(dedup-glomap-vs-colmap4): FilterTrackTriangulationAngle has the
// same math as ObservationManager::FilterPoints3DWithSmallTriangulationAngle
// (colmap/scene/observation_manager.h:125) but operates on
// glomap_ra::Track pre-Point3D state. After the Track-shape collapse
// (track_establishment_glomap), this whole TU can call native directly.
// See .claude/notes/glomap_audit/audit_glomap_files_vs_colmap4.md.

namespace colmap {
namespace glomap_ra {

using ViewGraph = colmap::CorrespondenceGraph;
using ImagePair = colmap::CorrespondenceGraph::ImagePair;


struct TrackFilter {
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
