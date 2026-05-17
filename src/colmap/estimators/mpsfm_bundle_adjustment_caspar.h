// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.

#pragma once

#include "colmap/estimators/bundle_adjustment.h"
#include "colmap/estimators/bundle_adjustment_caspar.h"
#include "colmap/scene/reconstruction.h"

#include <array>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace colmap {

enum class MpsfmCasparLossType { TRIVIAL, SOFT_L1, CAUCHY, HUBER };

struct MpsfmCasparReprojectionOptions {
  MpsfmCasparLossType loss_type = MpsfmCasparLossType::TRIVIAL;
  double loss_scale = 1.0;
  double keypoint_std = 1.0;
  double residual_weight = 1.0;
  bool use_keypoint_covariances = false;
};

struct MpsfmCasparDepthObservation {
  image_t image_id = kInvalidImageId;
  point3D_t point3D_id = kInvalidPoint3DId;
  double depth = 0.0;
  double sqrt_information = 1.0;
  MpsfmCasparLossType loss_type = MpsfmCasparLossType::TRIVIAL;
  double loss_scale = 1.0;
  bool gross_outlier = false;
  bool risky = false;
};

struct MpsfmCasparIntrinsicsPrior {
  camera_t camera_id = kInvalidCameraId;
  std::array<double, 4> prior = {0.0, 0.0, 0.0, 0.0};
  std::array<double, 4> std_dev = {1.0, 1.0, 1.0, 1.0};
};

struct MpsfmCasparIntrinsicsRandomWalk {
  camera_t prev_camera_id = kInvalidCameraId;
  camera_t next_camera_id = kInvalidCameraId;
  std::array<double, 4> variance_per_frame = {1.0, 1.0, 1.0, 1.0};
  double frame_gap = 1.0;
};

struct MpsfmCasparScalePrior {
  image_t image_id = kInvalidImageId;
  double std_dev = 1.0;
  MpsfmCasparLossType loss_type = MpsfmCasparLossType::TRIVIAL;
  double loss_scale = 1.0;
  double magnitude = 1.0;
};

struct MpsfmCasparBundleAdjustmentProblem {
  MpsfmCasparReprojectionOptions reprojection;
  std::vector<MpsfmCasparDepthObservation> depth_observations;
  std::vector<MpsfmCasparIntrinsicsPrior> intrinsics_priors;
  std::vector<MpsfmCasparIntrinsicsRandomWalk> intrinsics_random_walks;
  std::vector<MpsfmCasparScalePrior> scale_priors;
  std::unordered_map<image_t, std::array<double, 2>> shift_scale;
  bool tie_focal = false;
  bool log_depth = true;
};

struct MpsfmCasparBundleAdjustmentSummary : public BundleAdjustmentSummary {
  int num_reprojection_factors = 0;
  int num_depth_factors = 0;
  int num_intrinsics_prior_factors = 0;
  int num_intrinsics_random_walk_factors = 0;
  int num_scale_prior_factors = 0;
  double construction_time = 0.0;
  double solve_time = 0.0;
  double initial_score = 0.0;
  double final_score = 0.0;
  int iteration_count = 0;
  std::string backend_message;

  std::string BriefReport() const override;
};

std::shared_ptr<MpsfmCasparBundleAdjustmentSummary>
SolveMpsfmCasparBundleAdjustment(
    const BundleAdjustmentOptions& options,
    const BundleAdjustmentConfig& config,
    Reconstruction& reconstruction,
    MpsfmCasparBundleAdjustmentProblem& problem);

}  // namespace colmap
