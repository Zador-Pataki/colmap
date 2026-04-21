// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
// (BSD-3-Clause license, see LICENSE)

#pragma once

#include "colmap/scene/pose_graph.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/types.h"

#include <unordered_set>

namespace colmap {

// Establishes tracks from a pose graph and image set.
// Ported from the glomap fork's TrackEngine, adapted to upstream colmap types.
class TrackEngine {
 public:
  TrackEngine() = default;

  // Limit subsequent method scope to the given image IDs.
  // Empty set means "all images" (default).
  void SetImageIdsToProcess(
      const std::unordered_set<image_t>& image_ids) {
    image_ids_to_process_ = image_ids;
  }

  // For every LC edge in pose_graph (is_LC == true), append LC observations
  // to the owning Point3D's track.lc_elements_ via Track::AddLcElement.
  // Orphan pairs (no existing Point3D on either side) get a stub Point3D.
  // Skips pairs where either image is absent from image_ids_to_process_
  // (when the set is non-empty).
  void ProcessLoopClosurePairs(const PoseGraph& pose_graph,
                               Reconstruction& reconstruction);

 private:
  std::unordered_set<image_t> image_ids_to_process_;
};

}  // namespace colmap
