#pragma once

#include "colmap/estimators/ceres_loss.h"
#include "colmap/scene/pose_graph.h"
#include "colmap/scene/reconstruction.h"

#include <map>
#include <memory>
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

// (type, scale, weight) triple used by GlobalPositioner's 10-bucket
// per-observation loss cascade. The pycolmap binding accepts
// ``{name: str, scale, weight}`` dicts; videosfm maps the string name
// to the enum at the boundary (see _to_native_gp_options).
struct LossConfig {
  LossFunctionType type = LossFunctionType::TRIVIAL;
  double scale = 1.0;
  double weight = 1.0;

  // Wraps native ``CreateLossFunction(type, scale)`` with
  // ``ScaledLoss(weight)`` when weight != 1.
  std::shared_ptr<ceres::LossFunction> CreateLossFunction() const {
    auto loss = colmap::CreateLossFunction(type, scale);
    if (weight != 1.0) {
      loss.reset(new ceres::ScaledLoss(
          loss.release(), weight, ceres::TAKE_OWNERSHIP));
    }
    return std::shared_ptr<ceres::LossFunction>(loss.release());
  }
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

  // PRNG seed; -1 = non-deterministic random_device. The ctor also
  // honors a GP_SEED env var when this is -1.
  int random_seed = -1;

  // Top-level robust loss applied to the BATA direction residual.
  // Upstream colmap GP hardcoded ``HuberLoss(loss_function_scale)``;
  // this surface mirrors ``CeresBundleAdjustmentOptions`` so callers can
  // pick a different kernel without touching the GP body.
  LossFunctionType loss_function_type = LossFunctionType::HUBER;
  double loss_function_scale = 0.1;
  double loss_function_weight = 1.0;

  // Whether to use custom parameter block ordering for Schur-based solvers.
  // Disable for deterministic behavior when using a fixed random seed.
  bool use_parameter_block_ordering = true;

  // Whether to apply a 0.5x ScaledLoss to BATA residuals from cameras
  // whose focal length came from view-graph calibration rather than
  // an EXIF prior (``Camera::has_prior_focal_length == false``). The
  // heuristic downweights bearings whose direction was computed using
  // an estimated focal, since the bearing inherits the focal estimate's
  // uncertainty. Set false to treat all cameras at full weight.
  bool apply_uncalibrated_loss_downweight = true;

  // The options for the solver
  ceres::Solver::Options solver_options;

  // --- glomap-fork additions (default OFF — vanilla call = vanilla GP) ---

  // ``BATA`` = upstream behavior; ``SPLIT_METRIC_DEPTH`` adds a 1-D
  // ``MetricDepthError`` residual per observation with a valid depth prior.
  PointConstraintType point_constraint_type = PointConstraintType::BATA;

  // When true, observations with ``image.is_excluded[point2D_idx]`` are
  // skipped. The flag itself is fork-only (colmap/scene/image.h); this
  // gate keeps GP residual count unchanged when the flag is populated by
  // downstream code but not asked for.
  bool use_observation_exclusions = false;

  // When true, ``AddPoint3DToProblem`` also iterates
  // ``track.lc_elements`` (loop-closure observations). Vanilla colmap4
  // GP doesn't know about LC.
  bool use_lc_observations = false;

  // Skip random-init for both camera centers and track xyz. Used to
  // continue from a previous solve (e.g. GP1 -> GP2). Also gates
  // ``InitializeDepthMapScalesFromObservations``.
  bool use_init = false;

  // Cube size for random-init of camera centers / points.
  double random_init_scale = 100.0;

  // --- Metric-depth path toggles (only consulted when
  //     point_constraint_type == SPLIT_METRIC_DEPTH) ---
  bool use_log_scale_for_depth_map_scales = false;
  bool use_log_residual_for_depth = false;
  bool zero_residual_behind = false;
  bool smooth_log_linear_transition = false;
  double log_linear_threshold = 1.0;
  double scale_prior_stddev = 1.0;

  // Pre-Solve 3-sigma log-space depth-residual filter. Flagged
  // observations route through a hardcoded soft fallback in the
  // depth-loss cascade.
  bool filter_depth_outliers = false;

  // Optional caller-supplied seed for ``dmap_scales_`` (linear space).
  // ``std::nullopt`` → ``InitializeDepthMapScalesFromObservations``
  // when ``use_init=true``, else constant init (1.0 / 0.0 log).
  std::optional<std::unordered_map<image_t, double>> initial_dmap_scales;

  // 10-bucket per-observation loss routing.
  // NOTE: only consumed when ``point_constraint_type ==
  // SPLIT_METRIC_DEPTH``. In BATA mode the per-observation cascade
  // doesn't run; only the top-level ``loss_function_*`` fields apply
  // and these 10 buckets are silently ignored. ``loss_scale_prior`` is
  // always consumed.
  LossConfig loss_normal_geometry;
  LossConfig loss_normal_depth;
  LossConfig loss_lc_geometry;
  LossConfig loss_lc_depth;
  LossConfig loss_normal_geometry_inlier;
  LossConfig loss_normal_depth_inlier;
  LossConfig loss_normal_depth_outlier;
  LossConfig loss_normal_geometry_trackstart;
  LossConfig loss_normal_depth_trackstart;
  LossConfig loss_scale_prior;

  GlobalPositionerOptions() {
    solver_options.num_threads = -1;
    solver_options.max_num_iterations = 100;
    solver_options.function_tolerance = 1e-5;
  }

  std::shared_ptr<ceres::LossFunction> CreateLossFunction() const {
    auto loss = colmap::CreateLossFunction(loss_function_type,
                                           loss_function_scale);
    if (loss_function_weight != 1.0) {
      loss.reset(new ceres::ScaledLoss(
          loss.release(), loss_function_weight, ceres::TAKE_OWNERSHIP));
    }
    return std::shared_ptr<ceres::LossFunction>(loss.release());
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

  // Returns the per-image dmap_scales_ map after Solve(). Values are in
  // the parameterization the optimizer ran in (log-space when
  // options_.use_log_scale_for_depth_map_scales=true, linear otherwise).
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

  // Add a single observation (regular or LC) for one point3D. The
  // ``is_lc_observation`` flag selects which loss bucket the cascade
  // routes to.
  void AddObservationToProblem(point3D_t point3D_id,
                               const TrackElement& observation,
                               bool is_lc_observation,
                               bool random_initialization,
                               Reconstruction& reconstruction);

  // Sweep observations and flag those whose
  // ``|log(z_est) - log(scale * depth_prior)|`` exceeds 3 sigma into
  // ``depth_outliers_``. Gated by ``filter_depth_outliers`` +
  // ``SPLIT_METRIC_DEPTH``. The flagged set is consumed by the
  // depth-loss cascade in ``AddObservationToProblem``.
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

  // Per-image depth-map scale parameter blocks (lazily inserted on
  // first valid depth-prior observation; only populated when
  // SPLIT_METRIC_DEPTH active). Must be ``std::map`` not
  // ``unordered_map``: Ceres residuals store ``&dmap_scales_[image_id]``
  // and a hash rehash during insert would invalidate them.
  std::map<image_t, double> dmap_scales_;

  // Per-image count of MetricDepthError residuals. Used by the
  // per-image ScalePriorError to weight by observation density.
  std::unordered_map<image_t, int> dmap_scale_observation_counts_;

  // Observations flagged by ``FilterDepthOutliers``. Routed to
  // soft-fallback loss (non-LC) or skipped entirely (LC).
  std::set<std::pair<image_t, point2D_t>> depth_outliers_;

  // 10 cached loss buckets, one per ``options_.loss_*`` field.
  // Pre-warmed once in ``AddPointToCameraConstraints``.
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

  // ScaledLoss(HuberLoss(1), 1) used for non-LC depth outliers.
  // Lazily allocated.
  std::shared_ptr<ceres::LossFunction> soft_outlier_fallback_loss_;
};

// Solve global positioning using point-to-camera constraints.
bool RunGlobalPositioning(const GlobalPositionerOptions& options,
                          const PoseGraph& pose_graph,
                          Reconstruction& reconstruction);

}  // namespace colmap
