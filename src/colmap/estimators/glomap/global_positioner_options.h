#pragma once

#include "colmap/estimators/glomap/optimization_base_options.h"

#include <memory>
#include <string>

#include <ceres/loss_function.h>

namespace colmap::glomap {

struct GlobalPositionerOptions {
  // Nested LossFunctionConfig (POD; bound as nested py::class_, not dict
  // round-trip).
  struct LossFunctionConfig {
    std::string name = "trivial";
    double scale = 1.0;
    double weight = 1.0;

    LossFunctionConfig() = default;
    LossFunctionConfig(const std::string& n, double s, double w)
        : name(n), scale(s), weight(w) {}
  };

  enum class ConstraintType : int {
    ONLY_POINTS = 0,
  };

  enum class PointConstraintType : int {
    GEOMETRY_ONLY = 0,
    SPLIT_METRIC_DEPTH = 1,
  };

  // Embedded base options (composition, not inheritance).
  OptimizationBaseOptions solver_base;

  // Initialization flags.
  bool generate_random_positions = true;
  bool generate_random_points = true;
  bool generate_scales = true;
  double random_init_scale = 100.0;
  bool use_init = false;

  // What to optimize.
  bool optimize_positions = true;
  bool optimize_points = true;
  bool optimize_scales = true;
  bool optimize_depth_map_scales = true;
  bool optimize_rotations = false;

  // Rotation regularization (only used when optimize_rotations=true).
  bool regularize_rotations = true;
  double rotation_prior_sigma = 0.01;
  LossFunctionConfig loss_rotation_prior = {"trivial", 1.0, 1.0};

  // Track / seed / constraint knobs.
  int min_num_view_per_track = 3;
  unsigned seed = 1;
  ConstraintType constraint_type = ConstraintType::ONLY_POINTS;  // DEAD (no live reader; kept for YAML compat).
  PointConstraintType point_constraint_type = PointConstraintType::GEOMETRY_ONLY;

  // Scale regularization.
  double scale_regularization_sigma = 1.0;  // DEAD (no live reader).
  double scale_prior_stddev = 0.1;
  bool regularize_depth_map_scales = true;
  bool use_log_scale_for_depth_map_scales = true;

  // Depth residual transforms.
  bool use_log_residual_for_depth = false;
  bool zero_residual_behind_camera = false;
  bool smooth_log_linear_transition = false;
  double log_linear_threshold = 0.1;

  // Outlier filtering.
  bool filter_depth_outliers = false;

  // DEAD (no live reader).
  bool consecutive_trivial = true;

  // Relative pose constraints (MDRP-derived).
  bool use_relative_pose_constraints = false;
  double relative_pose_default_stddev = 0.1;
  bool debug_only_relative_pose = false;

  // Threshold for top-level loss function (mirrors solver_base.thres_loss_function
  // but kept for legacy API compatibility).
  double thres_loss_function = 0.1;

  // Loss configs.
  LossFunctionConfig loss_normal_geometry = {"huber", 0.1, 1.0};
  LossFunctionConfig loss_normal_depth = {"huber", 0.1, 1.0};
  LossFunctionConfig loss_lc_geometry = {"huber", 0.1, 1.0};
  LossFunctionConfig loss_lc_depth = {"huber", 0.1, 1.0};
  LossFunctionConfig loss_scale_prior = {"huber", 0.1, 1.0};
  LossFunctionConfig loss_normal_geometry_inlier = {"trivial", 1.0, 1.0};
  LossFunctionConfig loss_normal_depth_inlier = {"trivial", 1.0, 1.0};
  LossFunctionConfig loss_normal_depth_outlier = {"huber", 1.0, 1.0};
  LossFunctionConfig loss_normal_geometry_trackstart = {"trivial", 1.0, 1.0};
  LossFunctionConfig loss_normal_depth_trackstart = {"huber", 1.0, 1.0};
  LossFunctionConfig loss_relative_pose = {"huber", 0.1, 1.0};
};

// Namespace-scope alias so ported fork code referencing unqualified
// `LossFunctionConfig` keeps compiling.
using LossFunctionConfig = GlobalPositionerOptions::LossFunctionConfig;

// Free-function helper (fork had it as a static method on GlobalPositionerOptions).
// Implementation lives in global_positioner.cc.
std::shared_ptr<ceres::LossFunction> CreateLossFromConfig(
    const LossFunctionConfig& config);

// Namespace-scope alias so ported fork code that references the unqualified
// name `LossFunctionConfig` keeps working.
using LossFunctionConfig = GlobalPositionerOptions::LossFunctionConfig;

}  // namespace colmap::glomap
