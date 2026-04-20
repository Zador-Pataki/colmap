#include "colmap/estimators/glomap/pipeline.h"

#include "colmap/estimators/glomap/bundle_adjustment.h"
#include "colmap/estimators/glomap/global_positioner.h"
#include "colmap/estimators/glomap/rotation_averaging.h"
#include "colmap/estimators/glomap/track_establishment.h"
#include "colmap/estimators/glomap/track_filter.h"
#include "colmap/estimators/glomap/view_graph_calibration.h"

namespace colmap::glomap {

bool run_rotation_averaging(
    const RotationEstimatorOptions& options,
    ViewGraph& view_graph,
    std::unordered_map<image_t, Image>& images) {
  RotationEstimator estimator(options);
  auto [ok, _weights] = estimator.EstimateRotations(view_graph, images);
  return ok;
}

bool run_global_positioning(
    const GlobalPositionerOptions& options,
    const ViewGraph& view_graph,
    std::unordered_map<camera_t, Camera>& cameras,
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<track_t, Track>& tracks) {
  GlobalPositioner positioner(options);
  return positioner.Solve(view_graph, cameras, images, tracks);
}

void run_relative_pose_estimation(
    const RelativePoseEstimationOptions& options,
    ViewGraph& view_graph,
    std::unordered_map<camera_t, Camera>& cameras,
    std::unordered_map<image_t, Image>& images) {
  EstimateRelativePoses(view_graph, cameras, images, options);
}

bool run_view_graph_calibration(
    const ViewGraphCalibratorOptions& options,
    ViewGraph& view_graph,
    std::unordered_map<camera_t, Camera>& cameras,
    std::unordered_map<image_t, Image>& images) {
  ViewGraphCalibrator calibrator(options);
  return calibrator.Solve(view_graph, cameras, images);
}

size_t run_track_establishment(
    const TrackEstablishmentOptions& options,
    const ViewGraph& view_graph,
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<track_t, Track>& tracks) {
  TrackEngine engine(view_graph, images, options);
  return engine.EstablishFullTracks(tracks);
}

int run_track_filter(
    const InlierThresholdOptions& options,
    const ViewGraph& view_graph,
    const std::unordered_map<camera_t, Camera>& cameras,
    const std::unordered_map<image_t, Image>& images,
    std::unordered_map<track_t, Track>& tracks) {
  return TrackFilter::FilterTracksByReprojection(
      view_graph, cameras, images, tracks, options.max_reprojection_error,
      /*in_normalized_image=*/true);
}

bool run_bundle_adjustment(
    const BundleAdjustmentOptions& options,
    const ViewGraph& view_graph,
    std::unordered_map<camera_t, Camera>& cameras,
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<track_t, Track>& tracks) {
  BundleAdjuster adjuster(options);
  return adjuster.Solve(view_graph, cameras, images, tracks);
}

}  // namespace colmap::glomap
