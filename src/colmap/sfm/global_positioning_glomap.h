#pragma once

#include "colmap/sfm/optimization_base_glomap.h"
#include "colmap/scene/camera.h"
#include "colmap/scene/correspondence_graph.h"
#include "colmap/scene/image.h"
#include "colmap/sfm/track_establishment_glomap.h"  // Track + track_t + feature_t + Observation
#include "colmap/util/types.h"  // image_t, camera_t, image_pair_t
#include <unordered_map>
#include "colmap/util/logging.h"
#include "colmap/util/types.h"
#include <random>

#include <cmath>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace colmap {
namespace glomap_ra {

using ViewGraph = colmap::CorrespondenceGraph;
using ImagePair = colmap::CorrespondenceGraph::ImagePair;


// Configuration for a loss function
struct LossFunctionConfig {
  std::string name = "huber";  // "trivial", "huber", "cauchy", etc.
  double scale =
      1.0;  // Robustifier parameter (threshold for Huber, scale for Cauchy)
  double weight = 1.0;  // ScaledLoss multiplier

  LossFunctionConfig() = default;
  LossFunctionConfig(const std::string& n, double s, double w)
      : name(n), scale(s), weight(w) {}
};

struct GlobalPositionerOptions : public OptimizationBaseOptions {
  // Constraint type for global positioning
  enum ConstraintType {
    // only include camera to point constraints
    ONLY_POINTS,
  };

  // --- NEW: Enum for Point Constraint Source/Handling ---
  enum PointConstraintType {
    // Use only viewing direction (Optimize inverse range d_ik freely, ignore
    // depth priors)
    GEOMETRY_ONLY,
    // Use viewing direction + depth prior, assuming prior is metric (Fix scale

    // residual (camera Z vs m_ik). Skips observations without depth.
    SPLIT_METRIC_DEPTH,
    // Use viewing direction + depth prior with anisotropic weighting
    // Different weights for perpendicular (X,Y) vs radial (Z) components
    // ANISOTROPIC_METRIC_DEPTH
  };
  // --- END NEW ---

  // Whether initialize the reconstruction randomly
  bool generate_scales = true;  // Now using fixed 1 as initializaiton

  // Scale for random initialization (in meters if working in metric scale)
  // Random positions/points will be initialized in a cube from
  // [-random_init_scale, +random_init_scale] in each dimension
  // Default: 100.0 meters (200m x 200m x 200m cube)
  double random_init_scale = 100.0;

  // If true, use existing camera positions and track xyz values for
  // initialization instead of random initialization. If false, use random
  // initialization (or camera center for positions) as controlled by
  // generate_random_* flags.
  bool use_init = false;

  // Flags for which parameters to optimize
  bool optimize_depth_map_scales = true;

  // Constrain the minimum number of views per track
  int min_num_view_per_track = 3;

  // the type of global positioning
  ConstraintType constraint_type = ONLY_POINTS;

  PointConstraintType point_constraint_type = GEOMETRY_ONLY;
  // Standard deviation for scale regularization (uncertainty in d_ik prior)
  // Smaller values = stronger regularization (more trust in metric depth)
  // Typical values: 0.01 (strong trust) to 1.0 (weak trust)
  double scale_regularization_sigma = 1.0;

  // Standard deviation for depth map scale prior regularization
  // Pulls per-observation depth map scales toward 1.0
  // Smaller values = stronger regularization (more trust in initial scale=1.0)
  double scale_prior_stddev = 0.1;

  // If true, use log-space parameterization for depth map scales.
  // Optimizes log_scale where scale = exp(log_scale), making regularization
  // less aggressive for large scale errors. Default: true (recommended).
  bool use_log_scale_for_depth_map_scales = true;

  // If true, use log-space residual for depth errors when points are in front
  // of camera (z_est > 0). Points behind camera always use linear residual.
  // This treats relative depth errors equally regardless of absolute depth.
  // Default: false (backward compatible).
  bool use_log_residual_for_depth = false;

  // If true, set depth residuals to zero for points behind camera (z_est <= 0).
  // This effectively ignores depth constraints for points behind the camera,
  // which can help during initialization when many points start behind cameras.
  // Default: false (backward compatible).
  bool zero_residual_behind_camera = false;

  // If true, use smooth transition from log to linear residual at threshold.
  // For z > threshold: log-space residual
  // For z <= threshold: linear residual with gradient=1, continuous at threshold
  // This avoids gradient discontinuity at z=0 when use_log_residual_for_depth=true.
  // Default: false (backward compatible).
  bool smooth_log_linear_transition = false;

  // Threshold for smooth log-linear transition (in depth units, e.g., meters).
  // Only used if smooth_log_linear_transition=true.
  // Default: 0.1
  double log_linear_threshold = 0.1;

  // If true, pre-compute depth outliers before optimization and disable depth
  // constraints for them. Outliers are detected using log-space check:
  // |log(z_est) - log(depth_prior)| < 3 * log(1 + metric_std/depth_prior)
  // where metric_std = relative_stddev * depth_prior. Outliers will only use
  // geometry constraints. Default: false (backward compatible).
  bool filter_depth_outliers = false;

  // If true, use trivial loss functions for consecutive constraints
  // Otherwise use the same loss function as regular constraints
  bool consecutive_trivial = true;

  // Loss function configurations
  // These can be set from Python as dicts with keys: "name", "scale", "weight"
  LossFunctionConfig loss_normal_geometry =
      LossFunctionConfig("huber", 0.1, 1.0);
  LossFunctionConfig loss_normal_depth = LossFunctionConfig("huber", 0.1, 1.0);
  LossFunctionConfig loss_lc_geometry = LossFunctionConfig("huber", 0.1, 1.0);
  LossFunctionConfig loss_lc_depth = LossFunctionConfig("huber", 0.1, 1.0);
  LossFunctionConfig loss_scale_prior = LossFunctionConfig(
      "huber", 0.1, 1.0);  // For depth map scale prior regularization

  // Loss functions for inlier non-LC observations (used in second GP)
  // Default to trivial (no robust downweighting) to lock in good observations
  LossFunctionConfig loss_normal_geometry_inlier =
      LossFunctionConfig("trivial", 1.0, 1.0);
  LossFunctionConfig loss_normal_depth_inlier =
      LossFunctionConfig("trivial", 1.0, 1.0);

  // Loss function for MDRP depth outliers (observations with inconsistent
  // depths) Default to Huber with scale=1 for robust downweighting
  LossFunctionConfig loss_normal_depth_outlier =
      LossFunctionConfig("huber", 1.0, 1.0);

  // Loss function for track anchor geometry residuals (first observation in
  // track) Default to trivial (L2) to trust anchors fully
  LossFunctionConfig loss_normal_geometry_trackstart =
      LossFunctionConfig("trivial", 1.0, 1.0);

  // Loss function for track anchor depth residuals (first observation in track)
  // Default to huber to be robust for depth while trusting geometry
  LossFunctionConfig loss_normal_depth_trackstart =
      LossFunctionConfig("huber", 1.0, 1.0);

  // If true, add relative pose constraints for consecutive pairs
  // Uses MDRP-derived relative translations with covariance weighting
  bool use_relative_pose_constraints = false;

  // Standard deviation for relative pose constraints if cov_t is zero
  // Smaller values = stronger regularization
  double relative_pose_default_stddev = 0.1;

  // Loss function for relative pose constraints
  LossFunctionConfig loss_relative_pose =
      LossFunctionConfig("huber", 0.1, 1.0);

  // Debug: if true, skip all point-to-camera constraints (only use relative
  // pose)
  bool debug_only_relative_pose = false;

  GlobalPositionerOptions() : OptimizationBaseOptions() {
    thres_loss_function = 1e-1;
  }

  // Helper function to create a Ceres loss function from a config
  static std::shared_ptr<ceres::LossFunction> CreateLossFromConfig(
      const LossFunctionConfig& config);
};

class GlobalPositioner {
 public:
  GlobalPositioner(const GlobalPositionerOptions& options);

  // Returns true if the optimization was a success, false if there was a
  // failure.
  // Assume tracks here are already filtered
  // If image_ids_to_optimize is not empty, only optimize those images and
  // tracks/observations that reference them
  // If initial_dmap_scales is provided, use those values to initialize depth
  // map scales (assumed to be in linear space, will be converted to log-space
  // if use_log_scale_for_depth_map_scales is true)
  bool Solve(ViewGraph& view_graph,
             std::unordered_map<camera_t, Camera>& cameras,
             std::unordered_map<image_t, Image>& images,
             std::unordered_map<track_t, Track>& tracks,
             const std::unordered_set<image_t>& image_ids_to_optimize = {},
             const std::map<image_t, double>& initial_dmap_scales = {},
             const std::vector<image_pair_t>& consecutive_pair_ids = {});

  GlobalPositionerOptions& GetOptions() { return options_; }

  // Get the scales computed during optimization
  const std::vector<double>& GetScales() const { return scales_; }

  // Get the depth map scales computed during optimization
  // Converts from log-space to linear scale if log-space was used
  std::vector<double> GetDepthMapScales() const {
    std::vector<double> result;
    for (const auto& [image_id, scale_or_log] : dmap_scales_) {
      double scale = options_.use_log_scale_for_depth_map_scales
                         ? std::exp(scale_or_log)
                         : scale_or_log;
      result.push_back(scale);
    }
    return result;
  }

  // Get image_id -> depth map scale mapping
  // Converts from log-space to linear scale if log-space was used
  std::map<image_t, double> GetDepthMapScaleMap() const {
    std::map<image_t, double> result;
    for (const auto& [image_id, scale_or_log] : dmap_scales_) {
      double scale = options_.use_log_scale_for_depth_map_scales
                         ? std::exp(scale_or_log)
                         : scale_or_log;
      result[image_id] = scale;
    }
    return result;
  }

  // Get nested dict: image_id -> scale (more Python-friendly)
  // Converts from log-space to linear scale if log-space was used
  std::map<image_t, double> GetDepthMapScaleMapNested() const {
    return GetDepthMapScaleMap();
  }

 protected:
  void SetupProblem(ViewGraph& view_graph,
                    const std::unordered_map<track_t, Track>& tracks);

  // Initialize all cameras to be random.
  void InitializeRandomPositions(ViewGraph& view_graph,
                                 std::unordered_map<image_t, Image>& images,
                                 std::unordered_map<track_t, Track>& tracks);

  // Add tracks to the problem
  void AddPointToCameraConstraints(
      ViewGraph& view_graph,
      std::unordered_map<camera_t, Camera>& cameras,
      std::unordered_map<image_t, Image>& images,
      std::unordered_map<track_t, Track>& tracks);

  // Add a single track to the problem
  void AddTrackToProblem(track_t track_id,
                         ViewGraph& view_graph,
                         std::unordered_map<camera_t, Camera>& cameras,
                         std::unordered_map<image_t, Image>& images,
                         std::unordered_map<track_t, Track>& tracks);

  // Helper to process a single observation
  void AddObservationToProblem(track_t track_id,
                               image_t image_id,
                               feature_t feature_id,
                               ViewGraph& view_graph,
                               std::unordered_map<camera_t, Camera>& cameras,
                               std::unordered_map<image_t, Image>& images,
                               std::unordered_map<track_t, Track>& tracks,
                               bool is_lc_observation);

  // Set the parameter groups
  void AddCamerasAndPointsToParameterGroups(
      std::unordered_map<image_t, Image>& images,
      std::unordered_map<track_t, Track>& tracks);

  // Parameterize the variables, set some variables to be constant if desired
  void ParameterizeVariables(std::unordered_map<image_t, Image>& images,
                             std::unordered_map<track_t, Track>& tracks);

  // Initialize depth map scales from observed 3D points
  // Computes median scale for each image: median(z_est / depth_prior)
  // Only called when use_init=true and initial_dmap_scales is empty
  void InitializeDepthMapScalesFromObservations(
      const std::unordered_map<image_t, Image>& images,
      const std::unordered_map<track_t, Track>& tracks);

  // Pre-compute depth outliers before optimization
  // Checks which observations are outliers relative to their depth priors
  // and stores them in depth_outliers_ set
  void FilterDepthOutliers(const std::unordered_map<image_t, Image>& images,
                           const std::unordered_map<track_t, Track>& tracks);

  // Helper methods to get loss functions from current options
  // These are called whenever a loss function is needed, ensuring
  // that changes to options_ take effect even after SetupProblem()
  std::shared_ptr<ceres::LossFunction> GetLossNormalGeometry() const;
  std::shared_ptr<ceres::LossFunction> GetLossNormalDepth() const;
  std::shared_ptr<ceres::LossFunction> GetLossLCGeometry() const;
  std::shared_ptr<ceres::LossFunction> GetLossLCDepth() const;
  std::shared_ptr<ceres::LossFunction> GetLossScalePrior() const;
  // Soft loss for outlier depth constraints (default: Huber scale=1, weight=1)
  std::shared_ptr<ceres::LossFunction> GetLossOutlierDepth() const;
  // Loss for inlier non-LC observations (used in second GP)
  std::shared_ptr<ceres::LossFunction> GetLossNormalGeometryInlier() const;
  std::shared_ptr<ceres::LossFunction> GetLossNormalDepthInlier() const;
  // Loss for MDRP depth outlier observations (configurable)
  std::shared_ptr<ceres::LossFunction> GetLossNormalDepthOutlier() const;

  // During the optimization, the camera translation is set to be the camera
  // center Convert the results back to camera poses
  void ConvertResults(std::unordered_map<image_t, Image>& images);

  GlobalPositionerOptions options_;

  std::mt19937 random_generator_;
  std::unique_ptr<ceres::Problem> problem_;

  // Set of image IDs to optimize (empty means optimize all)
  std::unordered_set<image_t> image_ids_to_optimize_;

  // Cached loss functions - kept alive for Ceres (DO_NOT_TAKE_OWNERSHIP)
  // These are updated dynamically from options_ when accessed
  mutable std::shared_ptr<ceres::LossFunction> cached_loss_normal_geometry_;
  mutable std::shared_ptr<ceres::LossFunction> cached_loss_normal_depth_;
  mutable std::shared_ptr<ceres::LossFunction> cached_loss_lc_geometry_;
  mutable std::shared_ptr<ceres::LossFunction> cached_loss_lc_depth_;
  mutable std::shared_ptr<ceres::LossFunction> cached_loss_scale_prior_;
  // Soft loss for outlier depth constraints (non-LC observations only)
  // Default: Huber with scale=1, weight=1
  mutable std::shared_ptr<ceres::LossFunction> cached_loss_outlier_depth_;
  // Loss for inlier non-LC observations (used in second GP)
  mutable std::shared_ptr<ceres::LossFunction>
      cached_loss_normal_geometry_inlier_;
  mutable std::shared_ptr<ceres::LossFunction> cached_loss_normal_depth_inlier_;
  // Loss for MDRP depth outlier observations (configurable)
  mutable std::shared_ptr<ceres::LossFunction>
      cached_loss_normal_depth_outlier_;
  // Loss for track anchor geometry observations (first observation in track)
  mutable std::shared_ptr<ceres::LossFunction>
      cached_loss_normal_geometry_trackstart_;
  // Loss for track anchor depth observations (first observation in track)
  mutable std::shared_ptr<ceres::LossFunction>
      cached_loss_normal_depth_trackstart_;
  // Auxiliary scale variables.
  std::vector<double> scales_;

  // Scale variables for depth maps (one per image)
  std::map<image_t, double> dmap_scales_;

  // Track number of depth observations per image (for weighting scale prior)
  std::map<image_t, int> dmap_scale_observation_counts_;

  // Set of (image_id, feature_id) pairs that are depth outliers and should
  // skip depth constraints (only use geometry constraints)
  std::set<std::pair<image_t, feature_t>> depth_outliers_;

};

}  // namespace glomap_ra
}  // namespace colmap
