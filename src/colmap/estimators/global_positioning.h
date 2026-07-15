#pragma once

#include "colmap/estimators/ceres_loss.h"
#include "colmap/estimators/cost_functions/metric_depth.h"
#include "colmap/scene/pose_graph.h"
#include "colmap/scene/reconstruction.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <ceres/ceres.h>

namespace colmap {

struct GlobalPositioningPlaybackCapture {
  std::string phase;
  int iteration = -1;
  std::vector<uint64_t> image_ids;
  std::vector<double> image_centers;
  std::vector<uint64_t> point3D_ids;
  std::vector<double> points3D;
  std::vector<uint64_t> lc_pairs;
  std::vector<uint64_t> lc_support_count;
  std::vector<double> lc_raw_score;
};

struct GlobalPositioningPlaybackOptions {
  int snapshot_every_n_iterations = 1;
  std::function<void(const GlobalPositioningPlaybackCapture&)> callback;

  bool IsEnabled() const { return static_cast<bool>(callback); }
};

struct GlobalPositionerOptions {
  // Whether to initialize the camera and track positions randomly.
  bool generate_random_positions = true;
  bool generate_random_points = true;
  // Whether to initialize the camera scales to a constant 1 or derive them from
  // the initialized camera and point positions.
  bool generate_scales = true;
  // When generate_scales is false, derive per-observation BATA scales from the
  // current camera/point geometry even for warm-started solves. Disable only to
  // reproduce legacy behavior where warm-started solves left these scales at 1.
  bool initialize_warm_start_scales = true;

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

  // Robust loss for the BATA direction residual.
  LossConfig loss = {LossFunctionType::HUBER, 0.1, 1.0};

  // Whether to use custom parameter block ordering for Schur-based solvers.
  // Disable for deterministic behavior when using a fixed random seed.
  bool use_parameter_block_ordering = true;

  // Apply 0.5x ScaledLoss to BATA residuals from cameras without an EXIF
  // focal-length prior.
  bool apply_uncalibrated_loss_downweight = true;
  // Scale factor applied to the loss of uncalibrated cameras when
  // apply_uncalibrated_loss_downweight is true.
  double uncalibrated_loss_downweight = 0.5;

  // The options for the solver
  ceres::Solver::Options solver_options;

  // Optional in-memory snapshots for visualization. No files are written by
  // the native solver, and the disabled path preserves the original solve.
  GlobalPositioningPlaybackOptions playback;

  // Add per-observation MetricDepthError residual alongside BATA.
  bool use_metric_depth_constraint = false;

  // Include loop-closure observations in point3D problems.
  bool use_lc_observations = false;

  // Skip random initialization and reuse existing positions/points.
  bool use_init = false;

  // Cube half-extent for random initialization of positions and points.
  double random_init_scale = 100.0;

  // --- Metric-depth path toggles (only consulted when
  //     use_metric_depth_constraint == true) ---
  bool use_log_scale_for_depth_map_scales = false;
  MetricDepthResidualType metric_depth_residual_type =
      MetricDepthResidualType::kLinear;
  bool zero_residual_behind = false;
  double log_linear_threshold = 0.1;
  double scale_prior_stddev = 1.0;

  // Pre-Solve log-space depth-residual filter. Flagged observations
  // route through a soft fallback in the depth-loss cascade.
  bool filter_depth_outliers = false;
  // Number of sigma for the log-space depth-residual outlier threshold.
  double filter_depth_outlier_sigma = 3.0;

  // Caller-supplied initial dmap_scales (linear space). nullopt = auto.
  std::optional<std::unordered_map<image_t, double>> initial_dmap_scales;

  // Debug-only initialization hooks for parity testing. Empty maps are ignored.
  // BATA scale keys use "point3D_id:image_id:point2D_idx:lcflag".
  std::unordered_map<frame_t, Eigen::Vector3d> debug_initial_frame_centers;
  std::unordered_map<point3D_t, Eigen::Vector3d> debug_initial_point3D_xyz;
  std::unordered_map<std::string, double> debug_initial_bata_scales;

  // Soft fallback loss for depth outliers flagged by FilterDepthOutliers.
  LossConfig loss_soft_outlier_fallback = {LossFunctionType::HUBER, 1.0, 1.0};

  // Per-observation loss routing (only active when
  // use_metric_depth_constraint).
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
    return std::shared_ptr<ceres::LossFunction>(
        loss.CreateLossFunction().release());
  }
};

struct GlobalPositionerDiagnostics {
  int num_bata_residuals = 0;
  int num_metric_depth_residuals = 0;
  int num_scale_prior_residuals = 0;
  int num_regular_observations_used = 0;
  int num_lc_observations_used = 0;
  int num_bata_scales = 0;
  int num_dmap_scales = 0;
  int num_frame_centers = 0;
  int num_point3D_xyz = 0;
  int num_residual_blocks = 0;
  int num_parameter_blocks = 0;
  int num_parameters = 0;
  int num_iterations = 0;
  double initial_cost = 0.0;
  double final_cost = 0.0;
  int termination_type = 0;
};

class GlobalPositioner {
 public:
  explicit GlobalPositioner(const GlobalPositionerOptions& options);

  // Returns true if the optimization was a success, false if there was a
  // failure.
  // Assume tracks here are already filtered
  bool Solve(const PoseGraph& pose_graph, Reconstruction& reconstruction);

  GlobalPositionerOptions& GetOptions() { return options_; }

  // Per-image dmap_scales_ after Solve() (log or linear per options).
  const std::map<image_t, double>& GetDmapScales() const {
    return dmap_scales_;
  }
  const std::unordered_map<frame_t, Eigen::Vector3d>& GetInitialFrameCenters()
      const {
    return initial_frame_centers_;
  }
  const std::unordered_map<point3D_t, Eigen::Vector3d>& GetInitialPoint3DXYZ()
      const {
    return initial_point3D_xyz_;
  }
  const std::unordered_map<std::string, double>& GetInitialBataScales() const {
    return initial_bata_scales_;
  }
  std::unordered_map<std::string, double> GetFinalBataScales() const;
  const GlobalPositionerDiagnostics& GetDiagnostics() const {
    return diagnostics_;
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

  void AddObservationToProblem(point3D_t point3D_id,
                               const TrackElement& observation,
                               bool random_initialization,
                               Reconstruction& reconstruction,
                               bool is_lc_observation = false);

  // Add a MetricDepthError residual for a single observation.
  void AddMetricDepthResidual(point3D_t point3D_id,
                              const TrackElement& observation,
                              bool is_lc_observation,
                              Reconstruction& reconstruction);

  // Seed dmap_scales_ from per-image median z_est / depth_prior.
  void InitializeDepthMapScalesFromObservations(
      const Reconstruction& reconstruction);

  // Flag observations with depth residual exceeding filter_depth_outlier_sigma.
  void FilterDepthOutliers(const Reconstruction& reconstruction);

  // Set the parameter groups
  void AddCamerasAndPointsToParameterGroups(Reconstruction& reconstruction);

  // Parameterize the variables, set some variables to be constant if desired
  void ParameterizeVariables(Reconstruction& reconstruction);

  void RecordPlaybackObservation(const TrackElement& observation,
                                 bool is_lc_observation,
                                 ceres::ResidualBlockId residual_block_id,
                                 ceres::LossFunction* loss_function);
  void WritePlaybackCapture(const char* phase,
                            int iteration,
                            const Reconstruction& reconstruction);

  bool UseImageCenterBlocks() const;
  bool UseFrameInplaceCenterBlocks() const;
  Eigen::Vector3d& MutableCenterForImage(const Image& image);
  double* MutableCenterDataForImage(const Image& image);
  Eigen::Vector3d CenterForImage(const Image& image) const;

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
  std::unordered_map<std::string, size_t> bata_scale_indices_;

  // Temporary storage for frame centers (world coordinates) during
  // optimization. This allows keeping RigFromWorld().translation() in
  // cam_from_world convention.
  std::unordered_map<frame_t, Eigen::Vector3d> frame_centers_;
  std::unordered_map<image_t, Eigen::Vector3d> image_centers_;
  std::unordered_map<frame_t, Eigen::Vector3d> initial_frame_centers_;
  std::unordered_map<point3D_t, Eigen::Vector3d> initial_point3D_xyz_;
  std::unordered_map<std::string, double> initial_bata_scales_;
  uint64_t residual_order_hash_ = 1469598103934665603ULL;

  struct PlaybackObservation {
    image_t image_id;
    point2D_t point2D_idx;
    image_t anchor_image_id;
    point2D_t anchor_point2D_idx;
    ceres::ResidualBlockId residual_block_id;
    ceres::LossFunction* loss_function;
  };
  struct PlaybackEdge {
    image_t image_id1;
    image_t image_id2;
    std::vector<size_t> observation_indices;
    size_t support_count;
  };
  std::vector<PlaybackObservation> playback_observations_;
  std::vector<image_t> playback_image_ids_;
  std::vector<point3D_t> playback_point3D_ids_;
  std::vector<PlaybackEdge> playback_edges_;
  bool playback_topology_ready_ = false;

  // Temporary storage for camera-in-rig positions when cam_from_rig is unknown
  // and needs to be estimated.
  std::unordered_map<sensor_t, Eigen::Vector3d> cams_in_rig_;

  // --- Optional extensions ---

  // std::map (not unordered) — Ceres stores &dmap_scales_[id] pointers.
  std::map<image_t, double> dmap_scales_;
  std::unordered_map<image_t, int> dmap_scale_observation_counts_;
  std::set<std::pair<image_t, point2D_t>> depth_outliers_;

  // Cached loss buckets for the depth cascade.
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

  // Soft fallback loss for non-LC depth outliers. Lazily allocated from
  // options_.loss_soft_outlier_fallback.
  std::shared_ptr<ceres::LossFunction> soft_outlier_fallback_loss_;

  // Per-image ScaledLoss wrappers created in the scale-prior loop.
  // Owned here because problem_ uses DO_NOT_TAKE_OWNERSHIP.
  std::vector<std::unique_ptr<ceres::LossFunction>> per_image_scale_losses_;

  GlobalPositionerDiagnostics diagnostics_;
};

// Solve global positioning using point-to-camera constraints.
bool RunGlobalPositioning(const GlobalPositionerOptions& options,
                          const PoseGraph& pose_graph,
                          Reconstruction& reconstruction);

}  // namespace colmap
