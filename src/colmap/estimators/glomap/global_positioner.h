#pragma once

#include "colmap/estimators/glomap/iteration_callback.h"
#include "colmap/estimators/glomap/global_positioner_options.h"
#include "colmap/glomap/camera.h"
#include "colmap/glomap/image.h"
#include "colmap/glomap/image_pair.h"
#include "colmap/glomap/track.h"
#include "colmap/glomap/view_graph.h"


#include <cmath>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace colmap::glomap {

// Configuration for a loss function
// LossFunctionConfig + GlobalPositionerOptions are defined in global_positioner_options.h (§07).

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
  bool Solve(const ViewGraph& view_graph,
             std::unordered_map<camera_t, Camera>& cameras,
             std::unordered_map<image_t, Image>& images,
             std::unordered_map<track_t, Track>& tracks,
             const std::unordered_set<image_t>& image_ids_to_optimize = {},
             const std::map<image_t, double>& initial_dmap_scales = {},
             const std::vector<image_pair_t>& consecutive_pair_ids = {},
             IterationCallbackFn iteration_callback = nullptr);

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

  // Get constraint statistics
  int GetGeometryOnlyConstraints() const { return geometry_only_constraints_; }
  int GetDepthConstraints() const { return depth_constraints_; }

 protected:
  void SetupProblem(const ViewGraph& view_graph,
                    const std::unordered_map<track_t, Track>& tracks);

  // Initialize all cameras to be random.
  void InitializeRandomPositions(const ViewGraph& view_graph,
                                 std::unordered_map<image_t, Image>& images,
                                 std::unordered_map<track_t, Track>& tracks);

  // Add tracks to the problem
  void AddPointToCameraConstraints(
      const ViewGraph& view_graph,
      std::unordered_map<camera_t, Camera>& cameras,
      std::unordered_map<image_t, Image>& images,
      std::unordered_map<track_t, Track>& tracks);

  // Add a single track to the problem
  void AddTrackToProblem(track_t track_id,
                         const ViewGraph& view_graph,
                         std::unordered_map<camera_t, Camera>& cameras,
                         std::unordered_map<image_t, Image>& images,
                         std::unordered_map<track_t, Track>& tracks);

  // Helper to process a single observation
  void AddObservationToProblem(track_t track_id,
                               image_t image_id,
                               feature_t feature_id,
                               const ViewGraph& view_graph,
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

  // Add relative pose constraints between consecutive pairs
  // Uses MDRP-derived relative translations with covariance weighting
  void AddRelativePoseConstraints(
      const ViewGraph& view_graph,
      std::unordered_map<image_t, Image>& images,
      const std::vector<image_pair_t>& consecutive_pair_ids);

  // Add rotation prior constraints when optimize_rotations=true
  // Pulls rotations toward their initial values captured at start of Solve()
  void AddRotationPriors(std::unordered_map<image_t, Image>& images);

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
  // Loss for rotation prior regularization
  std::shared_ptr<ceres::LossFunction> GetLossRotationPrior() const;

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
  // Loss for relative pose constraints
  mutable std::shared_ptr<ceres::LossFunction> cached_loss_relative_pose_;

  // Auxiliary scale variables.
  std::vector<double> scales_;

  // Scale variables for depth maps (one per image)
  std::map<image_t, double> dmap_scales_;

  // Track number of depth observations per image (for weighting scale prior)
  std::map<image_t, int> dmap_scale_observation_counts_;

  // Set of (image_id, feature_id) pairs that are depth outliers and should
  // skip depth constraints (only use geometry constraints)
  std::set<std::pair<image_t, feature_t>> depth_outliers_;

  // Counters for constraint types
  int geometry_only_constraints_ = 0;
  int depth_constraints_ = 0;

  // Counters for consecutive observations
  int consecutive_observations_ = 0;
  int non_consecutive_observations_ = 0;

  // Counters for residual types
  int normal_residuals_ = 0;
  int lc_residuals_ = 0;
  // Split normal residuals by inlier status (for second GP with split_inliers)
  int normal_inlier_residuals_ = 0;
  int normal_outlier_residuals_ = 0;
  // Counter for MDRP depth outlier residuals
  int mdrp_depth_outlier_residuals_ = 0;
  // Counter for track anchor geometry residuals
  int track_anchor_residuals_ = 0;
  // Counter for track anchor depth residuals
  int track_anchor_depth_residuals_ = 0;
  // Counters for BATA weighting types
  int mahalanobis_bata_residuals_ = 0;
  int diagonal_bata_residuals_ = 0;
  int unweighted_bata_residuals_ = 0;
  // Counter for relative pose constraints
  int relative_pose_constraints_ = 0;
  // Counter for rotation prior constraints
  int rotation_prior_constraints_ = 0;

  // Initial rotations for rotation prior regularization
  // Captured at start of Solve() when optimize_rotations=true
  std::map<image_t, Eigen::Quaterniond> initial_rotations_;

  // Cached loss function for rotation priors
  mutable std::shared_ptr<ceres::LossFunction> cached_loss_rotation_prior_;
};

}  // namespace colmap::glomap
