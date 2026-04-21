#pragma once

#include "colmap/estimators/iteration_callback.h"
#include "colmap/estimators/loss_function_config.h"
#include "colmap/scene/pose_graph.h"
#include "colmap/scene/reconstruction.h"

#include <map>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include <ceres/ceres.h>
#include <Eigen/Core>

namespace colmap {

enum class ConstraintType {
  ONLY_POINTS,
  ALL,  // reserved
};

enum class PointConstraintType {
  GEOMETRY_ONLY,       // viewing direction only, ignore depth priors
  SPLIT_METRIC_DEPTH,  // viewing direction + metric depth residual
};

struct GlobalPositionerOptions {
  // Whether to initialize the camera and track positions randomly.
  bool generate_random_positions = true;
  bool generate_random_points = true;
  // Whether to initialize the camera scales to a constant 1 or derive them from
  // the initialized camera and point positions.
  bool generate_scales = true;

  // Flags for which parameters to optimize
  bool optimize_positions = true;
  bool optimize_points = true;
  bool optimize_scales = true;

  bool use_gpu = true;
  std::string gpu_index = "-1";
  int min_num_images_gpu_solver = 50;

  // Constrain the minimum number of views per track
  int min_num_view_per_track = 3;

  // PRNG seed for random initialization.
  // If -1 (default), uses non-deterministic random_device seeding.
  // If >= 0, uses deterministic seeding with the given value.
  int random_seed = -1;

  // Scaling factor for the loss function
  double loss_function_scale = 0.1;

  // Whether to use custom parameter block ordering for Schur-based solvers.
  // Disable for deterministic behavior when using a fixed random seed.
  bool use_parameter_block_ordering = true;

  // The options for the solver
  ceres::Solver::Options solver_options;

  // --- Fork-only fields ---

  // Cube half-extent for random position/point initialization.
  double random_init_scale = 100.0;
  // If true, use existing positions/points for initialization instead of random.
  bool use_init = false;

  // Extension toggles
  bool optimize_depth_map_scales = true;
  bool optimize_rotations = false;
  bool regularize_rotations = true;
  double rotation_prior_sigma = 0.01;  // radians (~0.57 degrees)
  LossFunctionConfig loss_rotation_prior = {"trivial", 1.0, 1.0};

  // Constraint types
  ConstraintType constraint_type = ConstraintType::ONLY_POINTS;
  PointConstraintType point_constraint_type = PointConstraintType::GEOMETRY_ONLY;

  // Scale regularization
  double scale_regularization_sigma = 1.0;
  double scale_prior_stddev = 0.1;
  bool regularize_depth_map_scales = true;
  bool use_log_scale_for_depth_map_scales = true;

  // MDRP dispatch flags
  bool use_log_residual_for_depth = false;
  bool zero_residual_behind_camera = false;
  bool smooth_log_linear_transition = false;
  double log_linear_threshold = 0.1;
  bool filter_depth_outliers = false;
  bool consecutive_trivial = true;

  // Per-channel loss configs
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

  // Relative pose constraints
  bool use_relative_pose_constraints = false;
  double relative_pose_default_stddev = 0.1;
  bool debug_only_relative_pose = false;

  // Fork's own random seed (independent of upstream's random_seed).
  unsigned seed = 1;

  // Relative pose pair ids for use_relative_pose_constraints.
  // Moved from SolveExtended arg into options to fit upstream Solve() API.
  std::vector<image_pair_t> relative_pose_pair_ids = {};

  // --- New gate flags (all default OFF — upstream behavior unchanged) ---

  // Master gate: enable per-channel loss dispatch ladder.
  // When false, upstream calibrated/uncalibrated loss pair is used.
  // When true, dispatch picks from cached_loss_* per obs classification.
  // NOTE: when use_fork_loss_dispatch=true, uncalibrated downweighting (0.5)
  // is disabled — fork never learned about it.
  bool use_fork_loss_dispatch = false;

  // Master gate: enable MDRP depth residuals.
  // Also activates dmap_scales_ parameter, FilterDepthOutliers, scale priors.
  bool use_depth_priors = false;

  // Enable loop-closure observations from Track::LcElements().
  bool use_lc_observations = false;

  // Enable hard per-feature exclusion via image.is_excluded.
  bool use_fork_observation_exclusion = false;

  // Subset of images to optimize (empty = all).
  std::unordered_set<image_t> image_ids_to_optimize = {};

  // Per-image depth-map scale seeds (linear space). Consumed once in Solve().
  // Seeds with scale <= 0 are rejected. Converted to log-space if
  // use_log_scale_for_depth_map_scales=true.
  std::map<image_t, double> initial_dmap_scales = {};

  // Iteration callback invoked each Ceres iteration.
  // Default nullptr = inert.
  glomap_ext::IterationCallbackFn iteration_callback = nullptr;

  GlobalPositionerOptions() {
    solver_options.num_threads = -1;
    solver_options.max_num_iterations = 100;
    solver_options.function_tolerance = 1e-5;
  }

  std::shared_ptr<ceres::LossFunction> CreateLossFunction() {
    return loss_normal_geometry.Create();
  }
};

class GlobalPositioner {
 public:
  explicit GlobalPositioner(const GlobalPositionerOptions& options);
  virtual ~GlobalPositioner() = default;

  // Returns true if the optimization was a success, false if there was a
  // failure.
  // Assume tracks here are already filtered
  virtual bool Solve(const PoseGraph& pose_graph,
                     Reconstruction& reconstruction);

  GlobalPositionerOptions& GetOptions() { return options_; }

  // Accessors for fork state (valid after Solve).
  const std::vector<double>& GetScales() const { return scales_; }
  std::vector<double> GetDepthMapScales() const;
  std::map<image_t, double> GetDepthMapScaleMap() const;
  std::map<image_t, double> GetDepthMapScaleMapNested() const;
  int GetGeometryOnlyConstraints() const { return geometry_only_constraints_; }
  int GetDepthConstraints() const { return depth_constraints_; }

 protected:
  void SetupProblem(const PoseGraph& pose_graph,
                    const Reconstruction& reconstruction);

  // Initialize all cameras to be random.
  virtual void InitializeRandomPositions(const PoseGraph& pose_graph,
                                         Reconstruction& reconstruction);

  // Add tracks to the problem
  virtual void AddPointToCameraConstraints(Reconstruction& reconstruction);

  // Add a single point3D to the problem
  virtual void AddPoint3DToProblem(point3D_t point3D_id,
                                   Reconstruction& reconstruction);

  // Set the parameter groups
  void AddCamerasAndPointsToParameterGroups(Reconstruction& reconstruction);

  // Parameterize the variables, set some variables to be constant if desired
  void ParameterizeVariables(Reconstruction& reconstruction);

  // During the optimization, the camera translation is set to be the camera
  // center Convert the results back to camera poses
  virtual void ConvertBackResults(Reconstruction& reconstruction);

  // --- Fork helper methods ---
  void AddObservationToProblem(point3D_t point3D_id,
                               image_t image_id,
                               point2D_t point2D_idx,
                               bool is_lc_observation,
                               Reconstruction& reconstruction);
  void AddScalePriorConstraints();
  void AddRelativePoseConstraints(const PoseGraph& pose_graph,
                                  Reconstruction& reconstruction);
  void AddRotationPriors(Reconstruction& reconstruction);
  void FilterDepthOutliers(const Reconstruction& reconstruction);
  void InitializeDepthMapScalesFromObservations(
      const Reconstruction& reconstruction);

  GlobalPositionerOptions options_;

  std::unique_ptr<ceres::Problem> problem_;

  // Loss functions for reweighted terms.
  std::shared_ptr<ceres::LossFunction> loss_function_;
  std::shared_ptr<ceres::LossFunction> loss_function_ptcam_uncalibrated_;
  std::shared_ptr<ceres::LossFunction> loss_function_ptcam_calibrated_;

  // Auxiliary scale variables.
  std::vector<double> scales_;

  // Temporary storage for frame centers (world coordinates) during
  // optimization. This allows keeping RigFromWorld().translation() in
  // cam_from_world convention.
  std::unordered_map<frame_t, Eigen::Vector3d> frame_centers_;

  // Temporary storage for camera-in-rig positions when cam_from_rig is unknown
  // and needs to be estimated.
  std::unordered_map<sensor_t, Eigen::Vector3d> cams_in_rig_;

  // --- Fork state (only non-trivial when fork flags are enabled) ---

  // Per-image depth-map scale parameter blocks. std::map for pointer stability.
  std::map<image_t, double> dmap_scales_;
  std::map<image_t, int> dmap_scale_observation_counts_;

  // (image_id, point2D_idx) pairs classified as depth outliers.
  std::set<std::pair<image_t, point2D_t>> depth_outliers_;

  // Initial rotations captured at solve start for rotation-prior regularization.
  std::map<image_t, Eigen::Quaterniond> initial_rotations_;

  // Per-residual scale-prior loss owners (DO_NOT_TAKE_OWNERSHIP in Ceres).
  std::vector<std::shared_ptr<ceres::LossFunction>> scale_prior_losses_;

  // Cached per-channel loss functions (alive for Ceres lifetime).
  std::shared_ptr<ceres::LossFunction> cached_loss_normal_geometry_;
  std::shared_ptr<ceres::LossFunction> cached_loss_normal_depth_;
  std::shared_ptr<ceres::LossFunction> cached_loss_lc_geometry_;
  std::shared_ptr<ceres::LossFunction> cached_loss_lc_depth_;
  std::shared_ptr<ceres::LossFunction> cached_loss_scale_prior_;
  std::shared_ptr<ceres::LossFunction> cached_loss_outlier_depth_;
  std::shared_ptr<ceres::LossFunction> cached_loss_normal_geometry_inlier_;
  std::shared_ptr<ceres::LossFunction> cached_loss_normal_depth_inlier_;
  std::shared_ptr<ceres::LossFunction> cached_loss_normal_depth_outlier_;
  std::shared_ptr<ceres::LossFunction> cached_loss_normal_geometry_trackstart_;
  std::shared_ptr<ceres::LossFunction> cached_loss_normal_depth_trackstart_;
  std::shared_ptr<ceres::LossFunction> cached_loss_relative_pose_;
  std::shared_ptr<ceres::LossFunction> cached_loss_rotation_prior_;

  // Residual / constraint counters (always maintained, non-zero only when
  // corresponding fork feature is enabled).
  int geometry_only_constraints_ = 0;
  int depth_constraints_ = 0;
  int normal_residuals_ = 0;
  int lc_residuals_ = 0;
  int normal_inlier_residuals_ = 0;
  int normal_outlier_residuals_ = 0;
  int mdrp_depth_outlier_residuals_ = 0;
  int track_anchor_residuals_ = 0;
  int track_anchor_depth_residuals_ = 0;
  int mahalanobis_bata_residuals_ = 0;
  int diagonal_bata_residuals_ = 0;
  int unweighted_bata_residuals_ = 0;
  int relative_pose_constraints_ = 0;
  int rotation_prior_constraints_ = 0;
};

// Solve global positioning using point-to-camera constraints.
bool RunGlobalPositioning(const GlobalPositionerOptions& options,
                          const PoseGraph& pose_graph,
                          Reconstruction& reconstruction);

}  // namespace colmap
