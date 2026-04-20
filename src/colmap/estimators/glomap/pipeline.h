// §12 pipeline: 8 `run_*` free functions wrapping §09-§11 estimator classes.
#pragma once

#include "colmap/estimators/glomap/bundle_adjustment_options.h"
#include "colmap/estimators/glomap/global_positioner_options.h"
#include "colmap/estimators/glomap/inlier_threshold_options.h"
#include "colmap/estimators/glomap/relpose_estimation.h"  // RelativePoseEstimationOptions
#include "colmap/estimators/glomap/rotation_estimator_options.h"
#include "colmap/estimators/glomap/track_establishment_options.h"
#include "colmap/estimators/glomap/triangulator_options.h"
#include "colmap/estimators/glomap/view_graph_calibrator_options.h"
#include "colmap/glomap/camera.h"
#include "colmap/glomap/image.h"
#include "colmap/glomap/track.h"
#include "colmap/glomap/view_graph.h"

#include <unordered_map>

namespace colmap::glomap {

bool run_rotation_averaging(
    const RotationEstimatorOptions& options,
    ViewGraph& view_graph,
    std::unordered_map<image_t, Image>& images);

bool run_global_positioning(
    const GlobalPositionerOptions& options,
    const ViewGraph& view_graph,
    std::unordered_map<camera_t, Camera>& cameras,
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<track_t, Track>& tracks);

void run_relative_pose_estimation(
    const RelativePoseEstimationOptions& options,
    ViewGraph& view_graph,
    std::unordered_map<camera_t, Camera>& cameras,
    std::unordered_map<image_t, Image>& images);

bool run_view_graph_calibration(
    const ViewGraphCalibratorOptions& options,
    ViewGraph& view_graph,
    std::unordered_map<camera_t, Camera>& cameras,
    std::unordered_map<image_t, Image>& images);

size_t run_track_establishment(
    const TrackEstablishmentOptions& options,
    const ViewGraph& view_graph,
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<track_t, Track>& tracks);

int run_track_filter(
    const InlierThresholdOptions& options,
    const ViewGraph& view_graph,
    const std::unordered_map<camera_t, Camera>& cameras,
    const std::unordered_map<image_t, Image>& images,
    std::unordered_map<track_t, Track>& tracks);

bool run_bundle_adjustment(
    const BundleAdjustmentOptions& options,
    const ViewGraph& view_graph,
    std::unordered_map<camera_t, Camera>& cameras,
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<track_t, Track>& tracks);

// run_track_retriangulation is deferred — its implementation
// (track_retriangulation.{h,cc}) is not yet wired into the build due to
// multiple colmap4 API divergences in colmap_converter.cc.

}  // namespace colmap::glomap
