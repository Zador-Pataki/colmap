// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
// (BSD-3-Clause license, see LICENSE)

#pragma once

#include "colmap/geometry/rigid3.h"
#include "colmap/scene/frame.h"
#include "colmap/scene/image.h"
#include "colmap/scene/pose_graph.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/types.h"

#include <ceres/ceres.h>

#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace colmap::glomap_ext {

// Info about a loop closure edge for visualization.
struct LCEdgeInfo {
  image_t image_id1;
  image_t image_id2;
  int shared_track_count;  // tracks with observations in both images
};

// Data passed to the iteration callback.
struct IterationData {
  int iteration;
  double cost;
  double cost_change;
  double gradient_norm;

  // Non-owning pointer to the reconstruction's live image map.
  // For GP: translations temporarily converted from camera centers to
  // cam_from_world during the callback, then restored.
  // For BA: already in cam_from_world form.
  std::unordered_map<image_t, colmap::Image>* images;

  // Non-owning pointer to the reconstruction's live Point3D map.
  // LC observations are in point3d.track.LcElements().
  const std::unordered_map<point3D_t, colmap::Point3D>* points3D;

  // Pointer to the callback object's own lc_edges_ vector; lifetime is
  // the enclosing Solve scope.
  const std::vector<LCEdgeInfo>* lc_edges;
};

using IterationCallbackFn = std::function<void(const IterationData&)>;

// Ceres iteration callback that snapshots current state and invokes a
// user-provided function. Designed for rendering optimization progress.
//
// gp_mode: when true, image Frame poses store camera CENTERS in translation
// (world position = -R^T * t_standard). The callback temporarily converts
// each posed image to standard cam_from_world form before invoking the user
// callback_, then restores the camera-center parameterization so the ceres
// parameter blocks are unchanged post-callback.
//
// Thread-safety: ceres invokes IterationCallback on the main solver thread
// (the thread that called Solve). The gp_mode translation flip is therefore
// safe without additional locking. If upstream adds multi-threaded iteration
// callbacks, revisit this section.
class SfMIterationCallback : public ceres::IterationCallback {
 public:
  SfMIterationCallback(IterationCallbackFn callback,
                       std::unordered_map<image_t, colmap::Image>& images,
                       colmap::Reconstruction& reconstruction,
                       const colmap::PoseGraph& pose_graph,
                       bool gp_mode)
      : callback_(std::move(callback)),
        images_(images),
        reconstruction_(reconstruction),
        gp_mode_(gp_mode) {
    ComputeLCEdges(pose_graph);
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
    data.points3D = &reconstruction_.Points3D();
    data.lc_edges = &lc_edges_;

    if (gp_mode_) {
      // Convert camera centers to standard cam_from_world for the callback.
      for (auto& [image_id, image] : images_) {
        if (!image.HasPose()) continue;
        Rigid3d& pose = image.FramePtr()->RigFromWorld();
        pose.translation() = -(pose.rotation() * pose.translation());
      }
    }

    callback_(data);

    if (gp_mode_) {
      // Restore camera-center parameterization for continued optimization.
      for (auto& [image_id, image] : images_) {
        if (!image.HasPose()) continue;
        Rigid3d& pose = image.FramePtr()->RigFromWorld();
        pose.translation() =
            -(pose.rotation().inverse() * pose.translation());
      }
    }

    return ceres::SOLVER_CONTINUE;
  }

 private:
  void ComputeLCEdges(const colmap::PoseGraph& pose_graph) {
    // Build image_id -> set of point3D_ids for shared-track counting.
    std::unordered_map<image_t, std::unordered_set<point3D_t>> image_to_p3ds;
    for (const auto& [pid, p3d] : reconstruction_.Points3D()) {
      for (const auto& el : p3d.track.Elements()) {
        image_to_p3ds[el.image_id].insert(pid);
      }
      for (const auto& el : p3d.track.LcElements()) {
        image_to_p3ds[el.image_id].insert(pid);
      }
    }

    // Find LC pairs and count shared tracks.
    for (const auto& [pair_id, edge] : pose_graph.Edges()) {
      if (!edge.is_LC) continue;
      if (!edge.valid) continue;

      const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);

      int shared = 0;
      auto it1 = image_to_p3ds.find(image_id1);
      auto it2 = image_to_p3ds.find(image_id2);
      if (it1 != image_to_p3ds.end() && it2 != image_to_p3ds.end()) {
        const auto& s1 = it1->second;
        const auto& s2 = it2->second;
        const auto& smaller = s1.size() < s2.size() ? s1 : s2;
        const auto& larger  = s1.size() < s2.size() ? s2 : s1;
        for (point3D_t pid : smaller) {
          if (larger.count(pid)) ++shared;
        }
      }

      lc_edges_.push_back(LCEdgeInfo{image_id1, image_id2, shared});
    }
  }

  IterationCallbackFn callback_;
  std::unordered_map<image_t, colmap::Image>& images_;
  colmap::Reconstruction& reconstruction_;
  bool gp_mode_;
  std::vector<LCEdgeInfo> lc_edges_;
};

}  // namespace colmap::glomap_ext
