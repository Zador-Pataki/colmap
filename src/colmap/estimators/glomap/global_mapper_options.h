#pragma once

#include "colmap/estimators/glomap/bundle_adjustment_options.h"
#include "colmap/estimators/glomap/global_positioner_options.h"
#include "colmap/estimators/glomap/inlier_threshold_options.h"
#include "colmap/estimators/glomap/rotation_estimator_options.h"
#include "colmap/estimators/glomap/track_establishment_options.h"
#include "colmap/estimators/glomap/triangulator_options.h"
#include "colmap/estimators/glomap/view_graph_calibrator_options.h"

namespace colmap::glomap {

// Top-level pipeline options. Mirrors fork's glomap::GlobalMapperOptions but
// omits the PoseLib-dependent RelativePoseEstimationOptions — that lives in
// a separate §11 header (relpose_estimation_options.h) to avoid leaking
// poselib through the generic options export.
struct GlobalMapperOptions {
  ViewGraphCalibratorOptions opt_vgcalib;
  RotationEstimatorOptions opt_ra;
  TrackEstablishmentOptions opt_track;
  GlobalPositionerOptions opt_gp;
  BundleAdjustmentOptions opt_ba;
  TriangulatorOptions opt_triangulator;

  InlierThresholdOptions inlier_thresholds;

  // Pipeline flow control.
  int num_iteration_bundle_adjustment = 3;
  int num_iteration_retriangulation = 1;

  bool skip_preprocessing = false;
  bool skip_view_graph_calibration = false;
  bool skip_relative_pose_estimation = false;
  bool skip_rotation_averaging = false;
  bool skip_track_establishment = false;
  bool skip_global_positioning = false;
  bool skip_bundle_adjustment = false;
  bool skip_retriangulation = false;
  bool skip_pruning = true;
};

}  // namespace colmap::glomap
