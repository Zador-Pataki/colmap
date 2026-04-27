#pragma once

#include "colmap/estimators/loss_config.h"
#include "colmap/scene/pose_graph.h"
#include "colmap/scene/reconstruction.h"

#include <map>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>

#include <ceres/ceres.h>

namespace colmap {

// Selects which residual structure ``GlobalPositioner`` adds per
// observation. Mirrors the glomap-fork enum.
enum class PointConstraintType {
  // Standard BATA direction residual only (native colmap default).
  BATA = 0,
  // Two residuals per observation: BATA direction + 1-D ``MetricDepthError``
  // against per-image ``dmap_scale * depth_prior``. Requires
  // ``image.depth_prior_validity[fid]`` populated.
  SPLIT_METRIC_DEPTH = 1,
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

  // PRNG seed for random initialization. Default 1 to preserve byte-identity
  // across runs out-of-the-box; explicit ``-1`` falls back to GP_SEED env var
  // for one transition cycle (Q8), then to non-deterministic random_device.
  int random_seed = 1;

  // Scaling factor for the loss function
  double loss_function_scale = 0.1;

  // Whether to use custom parameter block ordering for Schur-based solvers.
  // Disable for deterministic behavior when using a fixed random seed.
  bool use_parameter_block_ordering = true;

  // The options for the solver
  ceres::Solver::Options solver_options;

  // --- glomap-fork additions ---

  // Selects per-observation residual structure (see PointConstraintType
  // enum). ``BATA`` keeps native pre-port behavior; ``SPLIT_METRIC_DEPTH``
  // emits an extra ``MetricDepthError`` residual when depth priors are
  // available.
  PointConstraintType point_constraint_type = PointConstraintType::BATA;

  // If true, skip random-init for both camera centers AND track xyz (collapses
  // ``generate_random_positions`` + ``generate_random_points`` short-circuit).
  // Used for GP2 (continues from GP1) and the
  // ``init_first_gp_from_mdrp`` path. Also gates
  // ``InitializeDepthMapScalesFromObservations``.
  bool use_init = false;

  // Cube size for random-init of camera centers / points, replaces fork's
  // hardcoded 100.0.
  double random_init_scale = 100.0;

  // --- Metric-depth path toggles (only consulted when
  //     point_constraint_type == SPLIT_METRIC_DEPTH) ---
  bool use_log_scale_for_depth_map_scales = false;
  bool use_log_residual_for_depth = false;
  bool zero_residual_behind = false;
  bool smooth_log_linear_transition = false;
  double log_linear_threshold = 1.0;
  double scale_prior_stddev = 1.0;

  // Pre-Solve depth-outlier filter (3-sigma log-space residual). Populates
  // a per-observation outlier set that switches the depth-loss cascade to a
  // hardcoded soft fallback.
  bool filter_depth_outliers = false;

  // Caller-supplied (image_id -> linear scale) seed for ``dmap_scales_``.
  // Used by GP2 to continue from GP1's solved scales. ``std::nullopt`` →
  // either ``InitializeDepthMapScalesFromObservations`` (when
  // ``use_init=true``) or constant init (1.0 linear / 0.0 log).
  std::optional<std::unordered_map<image_t, double>> initial_dmap_scales;

  // --- 10-bucket per-observation loss routing (mirrors glomap-fork field
  //     names verbatim — the 10 LossFunctionConfig fields exposed via
  //     glomap_ra port at colmap/sfm/global_positioning_glomap.h:140-167) ---
  LossFunctionConfig loss_normal_geometry;
  LossFunctionConfig loss_normal_depth;
  LossFunctionConfig loss_lc_geometry;
  LossFunctionConfig loss_lc_depth;
  LossFunctionConfig loss_normal_geometry_inlier;
  LossFunctionConfig loss_normal_depth_inlier;
  LossFunctionConfig loss_normal_depth_outlier;
  LossFunctionConfig loss_normal_geometry_trackstart;
  LossFunctionConfig loss_normal_depth_trackstart;
  LossFunctionConfig loss_scale_prior;

  GlobalPositionerOptions() {
    solver_options.num_threads = -1;
    solver_options.max_num_iterations = 100;
    solver_options.function_tolerance = 1e-5;
  }

  std::shared_ptr<ceres::LossFunction> CreateLossFunction() {
    return std::make_shared<ceres::HuberLoss>(loss_function_scale);
  }
};

class GlobalPositioner {
 public:
  explicit GlobalPositioner(const GlobalPositionerOptions& options);

  // Returns true if the optimization was a success, false if there was a
  // failure.
  // Assume tracks here are already filtered
  bool Solve(const PoseGraph& pose_graph, Reconstruction& reconstruction);

  GlobalPositionerOptions& GetOptions() { return options_; }

  // --- Glomap-fork accessor (M7) ---
  // Returns the per-image dmap_scales_ map after Solve(). Values are in
  // the parameterization the optimizer ran in (log-space when
  // options_.use_log_scale_for_depth_map_scales=true, linear otherwise).
  // Pycolmap binding converts to linear space before returning to Python.
  const std::map<image_t, double>& GetDmapScales() const {
    return dmap_scales_;
  }

 protected:
  void SetupProblem(const PoseGraph& pose_graph,
                    const Reconstruction& reconstruction);

  // Initialize all cameras to be random.
  void InitializeRandomPositions(const PoseGraph& pose_graph,
                                 Reconstruction& reconstruction);

  // Add tracks to the problem
  void AddPointToCameraConstraints(Reconstruction& reconstruction);

  // Add a single point3D to the problem
  void AddPoint3DToProblem(point3D_t point3D_id,
                           Reconstruction& reconstruction);

  // Add a single observation (regular or LC) for one point3D. Extracted as
  // a helper so AddPoint3DToProblem can iterate ``track.Elements()`` and
  // ``track.lc_elements`` separately and pass ``is_lc_observation``
  // through to the loss-routing cascade (M5).
  void AddObservationToProblem(point3D_t point3D_id,
                               const TrackElement& observation,
                               bool is_lc_observation,
                               bool random_initialization,
                               Reconstruction& reconstruction);

  // Glomap-fork pre-Solve depth-outlier filter (M6). When
  // ``options_.filter_depth_outliers=true`` (and SPLIT_METRIC_DEPTH active),
  // sweep both regular and LC observations per track. Flag observations whose
  // ``|log(z_est) - log(scale * depth_prior)|`` exceeds 3 sigma in log-space
  // by inserting ``(image_id, point2D_idx)`` into ``depth_outliers_``. The
  // M5 depth-loss cascade then routes flagged observations to the soft
  // fallback (or skips them entirely on LC pairs).
  void FilterDepthOutliers(const Reconstruction& reconstruction);

  // Set the parameter groups
  void AddCamerasAndPointsToParameterGroups(Reconstruction& reconstruction);

  // Parameterize the variables, set some variables to be constant if desired
  void ParameterizeVariables(Reconstruction& reconstruction);

  // During the optimization, the camera translation is set to be the camera
  // center Convert the results back to camera poses
  void ConvertBackResults(Reconstruction& reconstruction);

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

  // --- glomap-fork additions ---

  // Per-image depth-map scale parameter blocks. Only populated when
  // ``point_constraint_type == SPLIT_METRIC_DEPTH``. Lazily inserted on the
  // first valid depth-prior observation per image. Keyed by ``image_t``
  // (matches fork; trivial-rig case has ``image_t == frame_t`` numerically).
  // Uses ``std::map`` instead of ``unordered_map`` because Ceres residuals
  // store ``&dmap_scales_[image_id]`` data pointers; a hash-table rehash
  // during lazy-insert would invalidate them. ``std::map`` is a balanced BST
  // — pointers stay stable for the lifetime of the entry.
  std::map<image_t, double> dmap_scales_;

  // Per-image observation count for scale-prior weighting. Each observation
  // that contributes a ``MetricDepthError`` residual increments the count;
  // the per-image ``ScalePriorError`` block scales its loss by this count
  // so dense-depth images get proportionally stronger priors.
  std::unordered_map<image_t, int> dmap_scale_observation_counts_;

  // Pre-pass-flagged depth outliers (only populated when
  // ``filter_depth_outliers=true``). Switches the depth-loss cascade to a
  // hardcoded soft fallback for non-LC outliers, or skip-depth-residual
  // for LC outliers.
  std::set<std::pair<image_t, point2D_t>> depth_outliers_;

  // 10 cached loss buckets corresponding to the option struct's
  // ``loss_*`` fields. Pre-warmed at the start of
  // ``AddPointToCameraConstraints`` and reused per observation. Lifetime
  // mirrors the ``loss_function_*`` members above.
  std::shared_ptr<ceres::LossFunction> cached_loss_normal_geometry_;
  std::shared_ptr<ceres::LossFunction> cached_loss_normal_depth_;
  std::shared_ptr<ceres::LossFunction> cached_loss_lc_geometry_;
  std::shared_ptr<ceres::LossFunction> cached_loss_lc_depth_;
  std::shared_ptr<ceres::LossFunction> cached_loss_normal_geometry_inlier_;
  std::shared_ptr<ceres::LossFunction> cached_loss_normal_depth_inlier_;
  std::shared_ptr<ceres::LossFunction> cached_loss_normal_depth_outlier_;
  std::shared_ptr<ceres::LossFunction> cached_loss_normal_geometry_trackstart_;
  std::shared_ptr<ceres::LossFunction> cached_loss_normal_depth_trackstart_;
  std::shared_ptr<ceres::LossFunction> cached_loss_scale_prior_;

  // Hardcoded ScaledLoss(HuberLoss(1), 1) for non-LC depth outliers flagged
  // by the M6 filter pre-pass. Allocated on first use in
  // ``AddPointToCameraConstraints``.
  std::shared_ptr<ceres::LossFunction> soft_outlier_fallback_loss_;
};

// Solve global positioning using point-to-camera constraints.
bool RunGlobalPositioning(const GlobalPositionerOptions& options,
                          const PoseGraph& pose_graph,
                          Reconstruction& reconstruction);

}  // namespace colmap
