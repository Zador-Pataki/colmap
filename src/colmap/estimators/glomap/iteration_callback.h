#pragma once

#include "colmap/glomap/image.h"
#include "colmap/glomap/image_pair.h"
#include "colmap/glomap/view_graph.h"
#include "colmap/glomap/track.h"


#include <ceres/ceres.h>

#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace colmap::glomap {

// Info about a loop closure edge for visualization
struct LCEdgeInfo {
  image_t image_id1;
  image_t image_id2;
  int shared_track_count;  // tracks with observations in both images
};

// Data passed to the iteration callback
struct IterationData {
  int iteration;
  double cost;
  double cost_change;
  double gradient_norm;

  // Camera poses: image_id -> cam_from_world (quaternion wxyz + translation)
  // For GP: converted from camera centers back to cam_from_world on-the-fly
  // For BA: already in cam_from_world form
  std::unordered_map<image_t, Image>* images;

  // 3D points: track_id -> xyz
  std::unordered_map<track_t, Track>* tracks;

  // LC edge info (static, computed once at construction)
  const std::vector<LCEdgeInfo>* lc_edges;
};

using IterationCallbackFn = std::function<void(const IterationData&)>;

// Ceres iteration callback that snapshots current state and invokes a
// user-provided function. Designed for rendering optimization progress.
//
// For GlobalPositioner: during optimization, image.cam_from_world.translation
// stores the camera CENTER (world position), not the standard t. The callback
// temporarily converts to cam_from_world, calls the user function, then
// restores the camera-center parameterization.
class SfMIterationCallback : public ceres::IterationCallback {
 public:
  // gp_mode: if true, translations are camera centers during optimization
  //          and need conversion for the callback
  SfMIterationCallback(IterationCallbackFn callback,
                       std::unordered_map<image_t, Image>& images,
                       std::unordered_map<track_t, Track>& tracks,
                       const ViewGraph& view_graph,
                       bool gp_mode)
      : callback_(std::move(callback)),
        images_(images),
        tracks_(tracks),
        gp_mode_(gp_mode) {
    ComputeLCEdges(view_graph);
  }

  ceres::CallbackReturnType operator()(
      const ceres::IterationSummary& summary) override {
    if (!callback_) return ceres::SOLVER_CONTINUE;

    IterationData data;
    data.iteration = summary.iteration;
    data.cost = summary.cost;
    data.cost_change = summary.cost_change;
    data.gradient_norm = summary.gradient_max_norm;
    data.images = &images_;
    data.tracks = &tracks_;
    data.lc_edges = &lc_edges_;

    if (gp_mode_) {
      // Convert camera centers to cam_from_world for the callback
      for (auto& [image_id, image] : images_) {
        image.cam_from_world.translation() =  // TODO(rigid3d-sweep): manual review needed
            -(image.cam_from_world.rotation() * image.cam_from_world.translation());
      }
    }

    callback_(data);

    if (gp_mode_) {
      // Restore camera centers for continued optimization
      for (auto& [image_id, image] : images_) {
        image.cam_from_world.translation() =  // TODO(rigid3d-sweep): manual review needed
            -(image.cam_from_world.rotation().inverse() *
              image.cam_from_world.translation());
      }
    }

    return ceres::SOLVER_CONTINUE;
  }

 private:
  // Pre-compute LC edges and shared track counts from view graph
  void ComputeLCEdges(const ViewGraph& view_graph) {
    // Build image_id -> set of track_ids for shared track counting
    std::unordered_map<image_t, std::unordered_set<track_t>>
        image_to_tracks;
    for (const auto& [track_id, track] : tracks_) {
      for (const auto& [image_id, feature_id] : track.observations) {
        image_to_tracks[image_id].insert(track_id);
      }
      for (const auto& [image_id, feature_id] : track.lc_observations) {
        image_to_tracks[image_id].insert(track_id);
      }
    }

    // Find LC pairs and count shared tracks
    for (const auto& [pair_id, pair] : view_graph.image_pairs) {
      if (!pair.is_LC) continue;
      if (!pair.is_valid) continue;

      int shared = 0;
      auto it1 = image_to_tracks.find(pair.image_id1);
      auto it2 = image_to_tracks.find(pair.image_id2);
      if (it1 != image_to_tracks.end() && it2 != image_to_tracks.end()) {
        const auto& tracks1 = it1->second;
        const auto& tracks2 = it2->second;
        // Iterate over the smaller set
        const auto& smaller =
            tracks1.size() < tracks2.size() ? tracks1 : tracks2;
        const auto& larger =
            tracks1.size() < tracks2.size() ? tracks2 : tracks1;
        for (track_t tid : smaller) {
          if (larger.count(tid)) ++shared;
        }
      }

      lc_edges_.push_back(
          LCEdgeInfo{pair.image_id1, pair.image_id2, shared});
    }
  }

  IterationCallbackFn callback_;
  std::unordered_map<image_t, Image>& images_;
  std::unordered_map<track_t, Track>& tracks_;
  bool gp_mode_;
  std::vector<LCEdgeInfo> lc_edges_;
};

}  // namespace colmap::glomap
