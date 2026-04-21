// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
// (BSD-3-Clause license, see LICENSE)

#pragma once

#include "colmap/scene/camera.h"
#include "colmap/scene/image.h"
#include "colmap/scene/pose_graph.h"

#include <unordered_map>

namespace colmap {

// Decompose the two-view geometry (E/F/H + config) into cam2_from_cam1 for
// each valid edge whose cameras have has_prior_focal_length == true.
// Writes back cam2_from_cam1 and updates edge.config.
// Port of fork's ViewGraphManipulater::DecomposeRelPose.
void DecomposeRelPose(PoseGraph& pose_graph,
                      const std::unordered_map<camera_t, Camera>& cameras,
                      const std::unordered_map<image_t, Image>& images);

// Re-derive per-pair TwoViewGeometry config after camera intrinsics change.
// For UNCALIBRATED pairs where both cameras have has_prior_focal_length,
// upgrades config to CALIBRATED and recomputes F from cam2_from_cam1.
// Port of fork's ViewGraphManipulater::UpdateImagePairsConfig.
void UpdateImagePairsConfig(PoseGraph& pose_graph,
                             const std::unordered_map<camera_t, Camera>& cameras,
                             const std::unordered_map<image_t, Image>& images);

// Populate Image::features_undist for each image (CamFromImg on each point).
// Skips images where features_undist.size() == NumPoints2D() unless
// clean_points == true (force re-undistortion).
// Port of fork's glomap::UndistortImages.
void UndistortImages(const std::unordered_map<camera_t, Camera>& cameras,
                     std::unordered_map<image_t, Image>& images,
                     bool clean_points = false);

}  // namespace colmap
