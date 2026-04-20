#pragma once

#include "colmap/estimators/glomap/optimization_base_options.h"

#include <string>

#include <Eigen/Core>

namespace colmap::glomap {

struct RotationEstimatorOptions {
  enum class SolverType : int {
    L1_IRLS = 0,
    CERES_GRAVITY = 1,
  };

  enum class WeightType : int {
    GEMAN_MCCLURE = 0,
    HALF_NORM = 1,
  };

  OptimizationBaseOptions solver_base;

  // Upstream L1/IRLS knobs.
  int max_num_l1_iterations = 5;
  double l1_step_convergence_threshold = 1e-3;
  int max_num_irls_iterations = 100;
  double irls_step_convergence_threshold = 1e-3;
  double irls_loss_parameter_sigma = 5.0;  // in degrees

  Eigen::Vector3d axis = Eigen::Vector3d(0, 1, 0);
  WeightType weight_type = WeightType::GEMAN_MCCLURE;

  // Init / precomputed-weights flags.
  bool skip_initialization = false;
  bool use_precomputed_weights = false;
  bool use_gravity = false;
  bool fix_non_lc_weights = false;
  double fixed_non_lc_weight = 138.0;
  bool skip_risky_LC_pairs = false;

  // Ceres-solver path (video + LC diff losses).
  SolverType solver_type = SolverType::L1_IRLS;
  bool use_video_constraints = false;
  double video_tracking_huber_scale = 0.1;
  double video_lc_cauchy_scale = 0.05;

  // Soft gravity prior for Ceres solver.
  bool use_gravity_prior = false;
  Eigen::Vector3d gravity_world = Eigen::Vector3d(0, -1, 0);
  double default_gravity_sigma = 0.035;
  double gravity_loss_scale = 0.05;
  std::string gravity_loss_type = "cauchy";

  // §09 extras used by pipeline wiring (loop closure weighting, video adjacency).
  int video_adjacent_frames = 0;
  double video_prior_sigma = 0.1;
  double gravity_prior_weight = 0.0;
  double loop_closure_weight = 1.0;
  int max_num_ceres_iterations = 100;
  unsigned random_seed = 42;
  int num_threads = -1;
  bool regularize_to_initial = false;
  double regularize_sigma = 0.1;
  bool update_rel_rotations = true;
  double min_weight = 1e-6;
};

}  // namespace colmap::glomap
