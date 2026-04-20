#include "colmap/estimators/glomap/global_positioner.h"

#include "colmap/estimators/cost_functions/bata_pairwise_direction_error.h"
#include "colmap/estimators/cost_functions/bata_pairwise_direction_error_with_rotation.h"
#include "colmap/estimators/cost_functions/direct_scale_regularization_error.h"
#include "colmap/estimators/cost_functions/glomap_helpers.h"
#include "colmap/estimators/cost_functions/log_scale_prior_error.h"
#include "colmap/estimators/cost_functions/mahalanobis_bata_directional_error.h"
#include "colmap/estimators/cost_functions/mahalanobis_bata_directional_error_with_rotation.h"
#include "colmap/estimators/cost_functions/metric_depth_error.h"
#include "colmap/estimators/cost_functions/metric_depth_error_with_rotation.h"
#include "colmap/estimators/cost_functions/point_camera_anisotropic_range_error.h"
#include "colmap/estimators/cost_functions/point_camera_direct_scaled_range_error.h"
#include "colmap/estimators/cost_functions/point_camera_range_error.h"
#include "colmap/estimators/cost_functions/point_camera_scaled_metric_range_error.h"
#include "colmap/estimators/cost_functions/point_camera_scaled_range_error.h"
#include "colmap/estimators/cost_functions/relative_translation_error.h"
#include "colmap/estimators/cost_functions/rotation_prior_error.h"
#include "colmap/estimators/cost_functions/scale_prior_error.h"
#include "colmap/estimators/cost_functions/scale_regularization_error.h"
#include "colmap/estimators/cost_functions/weighted_bata_directional_error.h"
#include "colmap/estimators/cost_functions/weighted_bata_directional_error_with_rotation.h"
#include "colmap/estimators/cost_functions/weighted_point_camera_scaled_metric_range_error.h"
#include "colmap/estimators/glomap/iteration_callback.h"

#include <colmap/estimators/manifold.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <typeinfo>

namespace colmap::glomap {
namespace {

Eigen::Vector3d RandVector3d(std::mt19937& random_generator,
                             double low,
                             double high) {
  std::uniform_real_distribution<double> distribution(low, high);
  return Eigen::Vector3d(distribution(random_generator),
                         distribution(random_generator),
                         distribution(random_generator));
}

}  // namespace

std::shared_ptr<ceres::LossFunction>
CreateLossFromConfig(
    const LossFunctionConfig& config) {
  std::shared_ptr<ceres::LossFunction> base_loss;

  // Create the base loss function based on name
  if (config.name == "trivial") {
    base_loss = std::make_shared<ceres::TrivialLoss>();
  } else if (config.name == "huber") {
    base_loss = std::make_shared<ceres::HuberLoss>(config.scale);
  } else if (config.name == "cauchy") {
    base_loss = std::make_shared<ceres::CauchyLoss>(config.scale);
  } else if (config.name == "arctan") {
    base_loss = std::make_shared<ceres::ArctanLoss>(config.scale);
  } else if (config.name == "softlone") {
    base_loss = std::make_shared<ceres::SoftLOneLoss>(config.scale);
  } else {
    LOG(WARNING) << "Unknown loss function name: " << config.name
                 << ", defaulting to Huber";
    base_loss = std::make_shared<ceres::HuberLoss>(config.scale);
  }

  // Apply weight via ScaledLoss if not 1.0
  // Note: We use TAKE_OWNERSHIP to avoid lifetime issues. The shared_ptr
  // will manage the ScaledLoss, which will own the base_loss.
  if (std::abs(config.weight - 1.0) < 1e-9) {
    return base_loss;
  } else {
    // Create a new base loss for ScaledLoss to own
    // We need to release from shared_ptr so ScaledLoss can take ownership
    ceres::LossFunction* raw_base_loss = nullptr;
    if (config.name == "trivial") {
      raw_base_loss = new ceres::TrivialLoss();
    } else if (config.name == "huber") {
      raw_base_loss = new ceres::HuberLoss(config.scale);
    } else if (config.name == "cauchy") {
      raw_base_loss = new ceres::CauchyLoss(config.scale);
    } else if (config.name == "arctan") {
      raw_base_loss = new ceres::ArctanLoss(config.scale);
    } else if (config.name == "softlone") {
      raw_base_loss = new ceres::SoftLOneLoss(config.scale);
    } else {
      raw_base_loss = new ceres::HuberLoss(config.scale);
    }
    // ScaledLoss will take ownership of raw_base_loss
    return std::shared_ptr<ceres::LossFunction>(new ceres::ScaledLoss(
        raw_base_loss, config.weight, ceres::TAKE_OWNERSHIP));
  }
}

GlobalPositioner::GlobalPositioner(const GlobalPositionerOptions& options)
    : options_(options) {
  random_generator_.seed(options_.seed);
}

bool GlobalPositioner::Solve(
    const ViewGraph& view_graph,
    std::unordered_map<camera_t, Camera>& cameras,
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<track_t, Track>& tracks,
    const std::unordered_set<image_t>& image_ids_to_optimize,
    const std::map<image_t, double>& initial_dmap_scales,
    const std::vector<image_pair_t>& consecutive_pair_ids,
    IterationCallbackFn iteration_callback) {
  if (images.empty()) {
    LOG(ERROR) << "Number of images = " << images.size();
    return false;
  }

  // Store the image IDs to optimize
  image_ids_to_optimize_ = image_ids_to_optimize;
  if (!image_ids_to_optimize_.empty()) {
    LOG(INFO) << "Optimizing only " << image_ids_to_optimize_.size()
              << " specified images";
  }

  // Capture initial rotations for regularization (if optimize_rotations=true)
  if (options_.optimize_rotations) {
    LOG(INFO) << "Capturing initial rotations for regularization";
    initial_rotations_.clear();
    for (const auto& [image_id, image] : images) {
      initial_rotations_[image_id] = image.cam_from_world.rotation();
    }
    LOG(INFO) << "Captured " << initial_rotations_.size() << " initial rotations";
  }

  LOG(INFO) << "Setting up the global positioner problem";

  // Setup the problem.
  SetupProblem(view_graph, tracks);

  // Initialize depth map scales from provided values if available
  if (!initial_dmap_scales.empty()) {
    LOG(INFO) << "Initializing " << initial_dmap_scales.size()
              << " depth map scales from provided values";
    for (const auto& [image_id, scale] : initial_dmap_scales) {
      if (scale <= 0.0) {
        LOG(WARNING) << "Invalid initial scale " << scale << " for image "
                     << image_id << ", skipping";
        continue;
      }
      // Convert to log-space if needed, otherwise use linear value
      double initial_value =
          options_.use_log_scale_for_depth_map_scales ? std::log(scale) : scale;
      dmap_scales_[image_id] = initial_value;
      dmap_scale_observation_counts_[image_id] = 0;
    }
  }

  // DEBUG: Print positions BEFORE initialization
  for (const auto& dbg_id : {120166, 120168, 120170, 120173}) {
    auto it = images.find(dbg_id);
    if (it != images.end()) {
      LOG(INFO) << "DEBUG InitPos BEFORE init: image " << dbg_id << " pos = ["
                << it->second.cam_from_world.translation().transpose() << "]";
    }
  }

  // Initialize camera translations to be random.
  // Also, convert the camera pose translation to be the camera center.
  InitializeRandomPositions(view_graph, images, tracks);

  // If use_init=true and no initial_dmap_scales provided, auto-initialize
  // depth map scales from observed 3D points (median of z_est / depth_prior)
  if (options_.use_init && initial_dmap_scales.empty()) {
    LOG(INFO) << "Auto-initializing depth map scales from observed 3D points";
    InitializeDepthMapScalesFromObservations(images, tracks);
  }

  if (options_.debug_only_relative_pose) {
    LOG(INFO) << "DEBUG: Skipping point to camera constraints (only using "
                 "relative pose constraints)";
  } else {
    LOG(INFO) << "Adding point to camera constraints";
    AddPointToCameraConstraints(view_graph, cameras, images, tracks);
  }

  // Add relative pose constraints for consecutive pairs (if enabled)
  AddRelativePoseConstraints(view_graph, images, consecutive_pair_ids);

  // Add rotation prior constraints (if optimize_rotations=true and
  // regularize_rotations=true)
  AddRotationPriors(images);

  AddCamerasAndPointsToParameterGroups(images, tracks);

  // Parameterize the variables, set image poses / tracks / scales to be
  // constant if desired
  ParameterizeVariables(images, tracks);

  LOG(INFO) << "Solving the global positioner problem";

  // Register iteration callback if provided
  std::unique_ptr<SfMIterationCallback> ceres_callback;
  if (iteration_callback) {
    ceres_callback = std::make_unique<SfMIterationCallback>(
        std::move(iteration_callback), images, tracks, view_graph,
        /*gp_mode=*/true);
    options_.solver_base.solver_options.callbacks.push_back(ceres_callback.get());
    options_.solver_base.solver_options.update_state_every_iteration = true;
  }

  ceres::Solver::Summary summary;
  options_.solver_base.solver_options.minimizer_progress_to_stdout =
      true;  // VLOG_IS_ON(2);
  ceres::Solve(options_.solver_base.solver_options, problem_.get(), &summary);

  // Clean up callback from solver options to avoid dangling pointer
  if (ceres_callback) {
    auto& cbs = options_.solver_base.solver_options.callbacks;
    cbs.erase(std::remove(cbs.begin(), cbs.end(), ceres_callback.get()),
              cbs.end());
    options_.solver_base.solver_options.update_state_every_iteration = false;
  }

  if (VLOG_IS_ON(2)) {
    LOG(INFO) << summary.FullReport();
  } else {
    LOG(INFO) << summary.BriefReport();
  }

  LOG(INFO) << "Constraint statistics: " << geometry_only_constraints_
            << " geometry-only constraints, " << depth_constraints_
            << " depth-based constraints";
  LOG(INFO) << "Consecutive observations: " << consecutive_observations_
            << " consecutive, " << non_consecutive_observations_
            << " non-consecutive";
  LOG(INFO) << "Depth map scales count: " << dmap_scales_.size();
  LOG(INFO) << "Regular scales count: " << scales_.size();
  LOG(INFO) << "Total residual blocks: " << problem_->NumResidualBlocks();
  LOG(INFO) << "Total parameter blocks: " << problem_->NumParameterBlocks();

  // Log final scale values for debugging
  if (!dmap_scales_.empty()) {
    LOG(INFO) << "Final depth map scales (first 10):";
    int count = 0;
    for (const auto& [img_id, scale_or_log] : dmap_scales_) {
      if (count++ >= 10) break;
      double scale = options_.use_log_scale_for_depth_map_scales
                         ? std::exp(scale_or_log)
                         : scale_or_log;
      LOG(INFO) << "  Image " << img_id << ": scale = " << scale;
    }
  }

  // Warn if scale optimization is disabled
  if (!options_.optimize_depth_map_scales) {
    LOG(WARNING) << "WARNING: optimize_depth_map_scales is FALSE - "
                 << "depth map scales are NOT being optimized!";
  }

  ConvertResults(images);
  return summary.IsSolutionUsable();
}

void GlobalPositioner::SetupProblem(
    const ViewGraph& view_graph,
    const std::unordered_map<track_t, Track>& tracks) {
  ceres::Problem::Options problem_options;
  problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  problem_ = std::make_unique<ceres::Problem>(problem_options);

  // Loss functions are now created lazily via helper methods
  // This allows options_ to be updated after SetupProblem() and
  // have those changes take effect when residuals are added

  // Log loss functions that will be used
  LOG(INFO) << "Loss functions configured:";
  LOG(INFO) << "  Normal geometry: " << options_.loss_normal_geometry.name
            << " (scale=" << options_.loss_normal_geometry.scale
            << ", weight=" << options_.loss_normal_geometry.weight << ")";
  LOG(INFO) << "  Normal depth: " << options_.loss_normal_depth.name
            << " (scale=" << options_.loss_normal_depth.scale
            << ", weight=" << options_.loss_normal_depth.weight << ")";
  LOG(INFO) << "  LC geometry: " << options_.loss_lc_geometry.name
            << " (scale=" << options_.loss_lc_geometry.scale
            << ", weight=" << options_.loss_lc_geometry.weight << ")";
  LOG(INFO) << "  LC depth: " << options_.loss_lc_depth.name
            << " (scale=" << options_.loss_lc_depth.scale
            << ", weight=" << options_.loss_lc_depth.weight << ")";
  LOG(INFO) << "  Scale prior: " << options_.loss_scale_prior.name
            << " (scale=" << options_.loss_scale_prior.scale
            << ", weight=" << options_.loss_scale_prior.weight << ")";
  LOG(INFO) << "  MDRP depth outlier: "
            << options_.loss_normal_depth_outlier.name
            << " (scale=" << options_.loss_normal_depth_outlier.scale
            << ", weight=" << options_.loss_normal_depth_outlier.weight << ")";

  // Reset counters
  geometry_only_constraints_ = 0;
  depth_constraints_ = 0;
  consecutive_observations_ = 0;
  non_consecutive_observations_ = 0;
  normal_residuals_ = 0;
  lc_residuals_ = 0;
  normal_inlier_residuals_ = 0;
  normal_outlier_residuals_ = 0;
  mdrp_depth_outlier_residuals_ = 0;
  track_anchor_residuals_ = 0;
  track_anchor_depth_residuals_ = 0;
  mahalanobis_bata_residuals_ = 0;
  diagonal_bata_residuals_ = 0;
  unweighted_bata_residuals_ = 0;
  relative_pose_constraints_ = 0;

  // Reserve capacity for scales to prevent reallocation after passing pointers
  // to Ceres Count total observations across all tracks
  size_t estimated_scales = std::accumulate(
      tracks.begin(),
      tracks.end(),
      0,
      [](size_t sum, const std::pair<track_t, Track>& kv) {
        const Track& t = kv.second;
        return sum + t.observations.size() + t.lc_observations.size();
      });
  scales_.clear();
  scales_.reserve(estimated_scales + 100);  // Add buffer for safety

  // Initialize depth map scales (will be initialized later when we have access
  // to images)
  dmap_scales_.clear();
}

void GlobalPositioner::InitializeRandomPositions(
    const ViewGraph& view_graph,
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<track_t, Track>& tracks) {
  std::unordered_set<image_t> constrained_positions;
  constrained_positions.reserve(images.size());
  for (const auto& [pair_id, image_pair] : view_graph.image_pairs) {
    if (image_pair.is_valid == false) continue;

    // If filtering by image IDs, only include pairs where at least one image is
    // in the set This allows us to constrain specified images even if the other
    // image isn't optimized
    if (!image_ids_to_optimize_.empty()) {
      bool id1_in_set = image_ids_to_optimize_.find(image_pair.image_id1) !=
                        image_ids_to_optimize_.end();
      bool id2_in_set = image_ids_to_optimize_.find(image_pair.image_id2) !=
                        image_ids_to_optimize_.end();
      if (!id1_in_set && !id2_in_set) {
        continue;  // Skip pairs where neither image is in the set
      }
    }

    constrained_positions.insert(image_pair.image_id1);
    constrained_positions.insert(image_pair.image_id2);
  }

  for (const auto& [track_id, track] : tracks) {
    if (track.observations.size() < options_.min_num_view_per_track) continue;
    for (const auto& observation : track.observations) {
      // If filtering by image IDs, only include specified images
      if (!image_ids_to_optimize_.empty() &&
          image_ids_to_optimize_.find(observation.first) ==
              image_ids_to_optimize_.end()) {
        continue;
      }
      if (images.find(observation.first) == images.end()) continue;
      Image& image = images[observation.first];
      if (!image.is_registered) continue;
      constrained_positions.insert(observation.first);
    }
  }

  if (!options_.optimize_positions) {
    for (auto& [image_id, image] : images) {
      image.cam_from_world.translation() = image.Center();  // TODO(rigid3d-sweep): manual review needed
    }
    return;
  }

  // Initialize camera positions: use existing values if available, otherwise
  // random or center
  int initialized_count = 0;
  int random_count = 0;
  int center_count = 0;

  for (auto& [image_id, image] : images) {
    // If filtering by image IDs, skip images not in the set
    if (!image_ids_to_optimize_.empty() &&
        image_ids_to_optimize_.find(image_id) == image_ids_to_optimize_.end()) {
      // Don't initialize positions for images we're not optimizing
      continue;
    }

    // Only initialize if this image is constrained
    if (constrained_positions.find(image_id) == constrained_positions.end()) {
      image.cam_from_world.translation() = image.Center();  // TODO(rigid3d-sweep): manual review needed
      center_count++;
      continue;
    }

    // Check if we should use existing initialization
    if (options_.use_init) {
      // Use existing camera position as-is (no checking needed)
      initialized_count++;
      // Position is already set, no need to change it
    } else if (options_.generate_random_positions) {
      // Generate random position
      image.cam_from_world.translation() =  // TODO(rigid3d-sweep): manual review needed
          options_.random_init_scale * RandVector3d(random_generator_, -1, 1);
      random_count++;
    } else {
      // Use camera center as fallback
      image.cam_from_world.translation() = image.Center();  // TODO(rigid3d-sweep): manual review needed
      center_count++;
    }
  }

  LOG(INFO) << "Camera position initialization: " << initialized_count
            << " using existing, " << random_count << " random, "
            << center_count << " set to center";

  // DEBUG: Print some specific positions to verify
  for (const auto& dbg_id : {120166, 120168, 120170, 120173}) {
    auto it = images.find(dbg_id);
    if (it != images.end()) {
      LOG(INFO) << "DEBUG InitPos after init: image " << dbg_id << " pos = ["
                << it->second.cam_from_world.translation().transpose() << "]";
    }
  }

  VLOG(2) << "Constrained positions: " << constrained_positions.size();
}

void GlobalPositioner::AddPointToCameraConstraints(
    const ViewGraph& view_graph,
    std::unordered_map<camera_t, Camera>& cameras,
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<track_t, Track>& tracks) {
  // Find the tracks that are relevant to the current set of cameras
  const size_t num_pt_to_cam = tracks.size();

  LOG(INFO) << "AddPointToCameraConstraints called with " << num_pt_to_cam
            << " tracks";
  LOG(INFO) << "Point constraint type: "
            << static_cast<int>(options_.point_constraint_type);

  // Initialize depth map scales (one per image) if not already done
  // No need to reserve for map, it will grow as needed

  VLOG(2) << num_pt_to_cam
          << " point to camera constriants were added to the position "
             "estimation problem.";

  if (num_pt_to_cam == 0) {
    LOG(INFO) << "No tracks available, returning early from "
                 "AddPointToCameraConstraints";
    return;
  }

  // Depth map scales will be initialized per-image in
  // AddObservationToProblem
  // Preserve initial_dmap_scales that were set in Solve() before clearing
  std::map<image_t, double> preserved_scales = dmap_scales_;
  dmap_scales_.clear();
  dmap_scale_observation_counts_.clear();

  // Restore preserved scales (from initial_dmap_scales)
  for (const auto& [image_id, scale] : preserved_scales) {
    dmap_scales_[image_id] = scale;
    // observation_count stays at 0 for these (they were initialized from
    // initial_dmap_scales)
  }

  // Pre-compute depth outliers if enabled
  depth_outliers_.clear();
  if (options_.filter_depth_outliers) {
    FilterDepthOutliers(images, tracks);
  }

  // Log current loss function configs that will be used
  // (Loss functions are now created dynamically from current options_)
  LOG(INFO) << "Using loss functions (from current options_):";
  LOG(INFO) << "  Normal geometry: " << options_.loss_normal_geometry.name
            << " (scale=" << options_.loss_normal_geometry.scale
            << ", weight=" << options_.loss_normal_geometry.weight << ")";
  LOG(INFO) << "  Normal depth: " << options_.loss_normal_depth.name
            << " (scale=" << options_.loss_normal_depth.scale
            << ", weight=" << options_.loss_normal_depth.weight << ")";
  LOG(INFO) << "  LC geometry: " << options_.loss_lc_geometry.name
            << " (scale=" << options_.loss_lc_geometry.scale
            << ", weight=" << options_.loss_lc_geometry.weight << ")";
  LOG(INFO) << "  LC depth: " << options_.loss_lc_depth.name
            << " (scale=" << options_.loss_lc_depth.scale
            << ", weight=" << options_.loss_lc_depth.weight << ")";
  LOG(INFO) << "  Scale prior: " << options_.loss_scale_prior.name
            << " (scale=" << options_.loss_scale_prior.scale
            << ", weight=" << options_.loss_scale_prior.weight << ")";
  LOG(INFO) << "  Track anchor geometry: "
            << options_.loss_normal_geometry_trackstart.name
            << " (scale=" << options_.loss_normal_geometry_trackstart.scale
            << ", weight=" << options_.loss_normal_geometry_trackstart.weight
            << ")";
  LOG(INFO) << "  Track anchor depth: "
            << options_.loss_normal_depth_trackstart.name
            << " (scale=" << options_.loss_normal_depth_trackstart.scale
            << ", weight=" << options_.loss_normal_depth_trackstart.weight
            << ")";

  // Create loss functions once at the start to keep them alive for all
  // residuals This ensures they don't get destroyed while Ceres is using them
  // We'll use the cached member variables directly in the loop, not call
  // getters
  cached_loss_normal_geometry_ = CreateLossFromConfig(
      options_.loss_normal_geometry);
  cached_loss_normal_depth_ =
      CreateLossFromConfig(options_.loss_normal_depth);
  cached_loss_lc_geometry_ =
      CreateLossFromConfig(options_.loss_lc_geometry);
  cached_loss_normal_geometry_inlier_ =
      CreateLossFromConfig(
          options_.loss_normal_geometry_inlier);
  cached_loss_normal_depth_inlier_ =
      CreateLossFromConfig(
          options_.loss_normal_depth_inlier);
  cached_loss_lc_depth_ =
      CreateLossFromConfig(options_.loss_lc_depth);
  cached_loss_scale_prior_ =
      CreateLossFromConfig(options_.loss_scale_prior);
  cached_loss_normal_geometry_trackstart_ =
      CreateLossFromConfig(
          options_.loss_normal_geometry_trackstart);
  cached_loss_normal_depth_trackstart_ =
      CreateLossFromConfig(
          options_.loss_normal_depth_trackstart);
  cached_loss_normal_depth_outlier_ =
      CreateLossFromConfig(
          options_.loss_normal_depth_outlier);

  int counter = 0;
  int track_initialized_count = 0;
  int track_random_count = 0;
  int track_uninitialized_count = 0;

  for (auto& [track_id, track] : tracks) {
    // If filtering by image IDs, check if track has any observations in the
    // specified images
    if (!image_ids_to_optimize_.empty()) {
      bool has_relevant_observation = false;
      for (const auto& obs : track.observations) {
        if (image_ids_to_optimize_.find(obs.first) !=
            image_ids_to_optimize_.end()) {
          has_relevant_observation = true;
          break;
        }
      }
      // Also check LC observations
      if (!has_relevant_observation) {
        for (const auto& lc_obs : track.lc_observations) {
          if (image_ids_to_optimize_.find(lc_obs.first) !=
              image_ids_to_optimize_.end()) {
            has_relevant_observation = true;
            break;
          }
        }
      }
      if (!has_relevant_observation) {
        continue;  // Skip tracks with no observations in specified images
      }
    }

    if (track.observations.size() < options_.min_num_view_per_track) {
      LOG(INFO) << "Skipping track " << track_id
                << " due to insufficient views: " << track.observations.size()
                << " < " << options_.min_num_view_per_track;
      continue;
    }

    // Initialize track xyz: use existing value if available, otherwise random
    if (options_.optimize_points) {
      if (options_.use_init) {
        // Use existing xyz value as-is (no checking needed)
        track.is_initialized = true;
        track_initialized_count++;
      } else if (options_.generate_random_points) {
        // Generate random position
        track.xyz =
            options_.random_init_scale * RandVector3d(random_generator_, -1, 1);
        track.is_initialized = true;
        track_random_count++;
      } else {
        track_uninitialized_count++;
      }
      // If use_init is false and generate_random_points is false, xyz remains
      // zero
    }

    AddTrackToProblem(track_id, view_graph, cameras, images, tracks);
    counter++;
  }

  // Now add scale prior residuals for all images with depth observations
  // Weight each prior by the number of observations for that image
  // Only add if regularization is enabled
  int scale_prior_count = 0;
  if (options_.regularize_depth_map_scales) {
    for (const auto& [img_id, obs_count] : dmap_scale_observation_counts_) {
      if (obs_count > 0) {
        auto scale_it = dmap_scales_.find(img_id);
        if (scale_it != dmap_scales_.end()) {
          ceres::CostFunction* scale_prior_cost = nullptr;
          if (options_.use_log_scale_for_depth_map_scales) {
            // Use log-space: convert scale_prior_stddev to log-space equivalent
            // For log-space, sigma_log represents relative uncertainty
            // A reasonable conversion: sigma_log ≈ scale_prior_stddev (for
            // small stddev) Or use a fixed conversion factor. Here we use the
            // same value as a starting point.
            double sigma_log = options_.scale_prior_stddev;
            scale_prior_cost = LogScalePriorError::Create(sigma_log);
          } else {
            // Use linear space
            const double scale_prior_val = 1.0;
            scale_prior_cost = ScalePriorError::Create(
                scale_prior_val, options_.scale_prior_stddev);
          }
          if (scale_prior_cost) {
            // Apply the user's weight from config, multiplied by observation
            // count This ensures images with more observations have
            // proportionally stronger priors, while still respecting the user's
            // weight setting We need to create a fresh base loss and apply the
            // total weight, since loss_scale_prior_ may already have the user's
            // weight applied
            double total_weight = options_.loss_scale_prior.weight *
                                  static_cast<double>(obs_count);
            VLOG(1) << "Creating scale prior loss for image " << img_id
                    << ": name=" << options_.loss_scale_prior.name
                    << ", user_weight=" << options_.loss_scale_prior.weight
                    << ", obs_count=" << obs_count
                    << ", total_weight=" << total_weight;

            // Create base loss function from config
            ceres::LossFunction* base_loss = nullptr;
            if (options_.loss_scale_prior.name == "trivial") {
              base_loss = new ceres::TrivialLoss();
            } else if (options_.loss_scale_prior.name == "huber") {
              base_loss = new ceres::HuberLoss(options_.loss_scale_prior.scale);
            } else if (options_.loss_scale_prior.name == "cauchy") {
              base_loss =
                  new ceres::CauchyLoss(options_.loss_scale_prior.scale);
            } else if (options_.loss_scale_prior.name == "arctan") {
              base_loss =
                  new ceres::ArctanLoss(options_.loss_scale_prior.scale);
            } else if (options_.loss_scale_prior.name == "softlone") {
              base_loss =
                  new ceres::SoftLOneLoss(options_.loss_scale_prior.scale);
            } else {
              base_loss = new ceres::HuberLoss(options_.loss_scale_prior.scale);
            }

            // Apply total weight (user weight * obs_count)
            ceres::LossFunction* scaled_prior_loss = nullptr;
            if (std::abs(total_weight - 1.0) < 1e-9) {
              scaled_prior_loss = base_loss;
            } else {
              scaled_prior_loss = new ceres::ScaledLoss(
                  base_loss, total_weight, ceres::TAKE_OWNERSHIP);
            }

            problem_->AddResidualBlock(
                scale_prior_cost, scaled_prior_loss, &scale_it->second);
            scale_prior_count++;

            // Log initial scale value for debugging
            double initial_scale = options_.use_log_scale_for_depth_map_scales
                                       ? std::exp(scale_it->second)
                                       : scale_it->second;
            VLOG(1) << "Added scale prior for image " << img_id
                    << " (obs_count=" << obs_count
                    << ", weight=" << total_weight
                    << ", initial_scale=" << initial_scale << ")";
          } else {
            LOG(ERROR) << "Failed to create scale prior error for image "
                       << img_id;
          }
        }
      }
    }
    LOG(INFO) << "Added " << scale_prior_count << " scale prior residuals";
  } else {
    LOG(INFO) << "Scale prior regularization is disabled";
  }

  LOG(INFO) << "Added " << counter << " point to camera constraints";
  if (options_.optimize_points) {
    LOG(INFO) << "Track initialization: " << track_initialized_count
              << " using existing, " << track_random_count << " random, "
              << track_uninitialized_count << " uninitialized";
  }
  LOG(INFO) << "Residual statistics: " << normal_residuals_
            << " normal residuals, " << lc_residuals_ << " LC residuals";
  if (normal_inlier_residuals_ > 0 || normal_outlier_residuals_ > 0) {
    LOG(INFO) << "  Normal residuals breakdown: " << normal_inlier_residuals_
              << " inlier (trivial loss), " << normal_outlier_residuals_
              << " outlier (robust loss)";
  }
  if (track_anchor_residuals_ > 0) {
    LOG(INFO) << "  Track anchor geometry residuals: "
              << track_anchor_residuals_
              << " (using loss_normal_geometry_trackstart)";
  }
  if (track_anchor_depth_residuals_ > 0) {
    LOG(INFO) << "  Track anchor depth residuals: "
              << track_anchor_depth_residuals_
              << " (using loss_normal_depth_trackstart)";
  }
  LOG(INFO) << "  MDRP depth outlier residuals: "
            << mdrp_depth_outlier_residuals_
            << " (using loss_normal_depth_outlier)";
  LOG(INFO) << "BATA weighting types: " << mahalanobis_bata_residuals_
            << " Mahalanobis (full 2x2 covar), " << diagonal_bata_residuals_
            << " diagonal, " << unweighted_bata_residuals_ << " unweighted";
}

void GlobalPositioner::AddTrackToProblem(
    track_t track_id,
    const ViewGraph& view_graph,
    std::unordered_map<camera_t, Camera>& cameras,
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<track_t, Track>& tracks) {
  // For each view in the track add the point to camera correspondences.
  for (const auto& observation : tracks[track_id].observations) {
    // If filtering by image IDs, only process observations from specified
    // images
    if (!image_ids_to_optimize_.empty() &&
        image_ids_to_optimize_.find(observation.first) ==
            image_ids_to_optimize_.end()) {
      continue;
    }
    AddObservationToProblem(track_id,
                            observation.first,
                            observation.second,
                            view_graph,
                            cameras,
                            images,
                            tracks,
                            false);
  }

  // Add LC observations using the same loss functions for now
  for (const auto& lc_obs : tracks[track_id].lc_observations) {
    // If filtering by image IDs, only process observations from specified
    // images
    if (!image_ids_to_optimize_.empty() &&
        image_ids_to_optimize_.find(lc_obs.first) ==
            image_ids_to_optimize_.end()) {
      continue;
    }
    AddObservationToProblem(track_id,
                            lc_obs.first,
                            lc_obs.second,
                            view_graph,
                            cameras,
                            images,
                            tracks,
                            true);
  }
}

void GlobalPositioner::AddObservationToProblem(
    track_t track_id,
    image_t image_id,
    feature_t feature_id,
    const ViewGraph& view_graph,
    std::unordered_map<camera_t, Camera>& cameras,
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<track_t, Track>& tracks,
    bool is_lc_observation) {
  if (images.find(image_id) == images.end()) return;

  // If filtering by image IDs, skip observations from non-specified images
  if (!image_ids_to_optimize_.empty() &&
      image_ids_to_optimize_.find(image_id) == image_ids_to_optimize_.end()) {
    return;
  }

  Image& image = images[image_id];
  if (!image.is_registered) return;

  // Check hard exclusion flag - skip this observation entirely if marked
  if (feature_id < image.is_excluded.size() && image.is_excluded[feature_id]) {
    return;
  }

  const Eigen::Vector3d& feature_undist = image.features_undist[feature_id];
  if (feature_undist.array().isNaN().any()) {
    LOG(WARNING) << "Ignoring feature because it failed to undistort: track_id="
                 << track_id << ", image_id=" << image_id
                 << ", feature_id=" << feature_id;
    return;
  }

  // v_ik is the world-frame bearing direction (used for non-rotation variants)
  const Eigen::Vector3d v_ik = image.cam_from_world.rotation().inverse() *
                               image.features_undist[feature_id];
  // bearing_cam is the camera-frame bearing (used for WithRotation variants)
  const Eigen::Vector3d& bearing_cam = image.features_undist[feature_id];

  // Check if this observation is a depth outlier
  bool is_outlier = depth_outliers_.find(std::make_pair(
                        image_id, feature_id)) != depth_outliers_.end();

  // Determine if we should use depth constraint
  // - If point_constraint_type is GEOMETRY_ONLY: never use depth
  // - If depth prior is invalid: never use depth
  // - If it's an outlier AND it's an LC observation: skip depth (only geometry)
  // - If it's an outlier AND it's NOT an LC observation: use depth with soft
  // loss
  // - If it's not an outlier: use depth with normal loss
  bool use_depth =
      options_.point_constraint_type !=
          GlobalPositionerOptions::PointConstraintType::GEOMETRY_ONLY &&
      image.depth_prior_validity[feature_id] &&
      !(is_outlier && is_lc_observation);  // Skip depth for LC outliers

  // Determine if we should use soft loss for depth (non-LC outliers only)
  bool use_soft_depth_loss = is_outlier && !is_lc_observation;

  ceres::CostFunction* cost_function = nullptr;
  if (!use_depth && !use_soft_depth_loss) {
    // No depth constraint: only geometry (for LC outliers or invalid depth)
    geometry_only_constraints_++;

    // Use Mahalanobis/Weighted BATA if covariances available, else unweighted
    // Branch based on optimize_rotations: use *WithRotation variants if true
    if (options_.optimize_rotations) {
      // Rotation as optimization parameter - use bearing_cam (camera frame)
      if (feature_id < image.angular_cholesky_xy.size() &&
          feature_id < image.angular_stddevs_z.size()) {
        const Eigen::Vector3d& chol = image.angular_cholesky_xy[feature_id];
        const double sigma_z =
            std::max(1e-9, image.angular_stddevs_z[feature_id]);
        cost_function = MahalanobisBATADirectionalErrorWithRotation::Create(
            bearing_cam, chol[0], chol[1], chol[2], sigma_z);
        mahalanobis_bata_residuals_++;
      } else if (feature_id < image.angular_stddevs.size()) {
        const Eigen::Vector2d& angular_std = image.angular_stddevs[feature_id];
        const double sigma_x = std::max(1e-9, angular_std[0]);
        const double sigma_y = std::max(1e-9, angular_std[1]);
        const double sigma_z = 0.5 * (sigma_x + sigma_y);
        cost_function = WeightedBATADirectionalErrorWithRotation::Create(
            bearing_cam, sigma_x, sigma_y, sigma_z);
        diagonal_bata_residuals_++;
      } else {
        cost_function = BATAPairwiseDirectionErrorWithRotation::Create(bearing_cam);
        unweighted_bata_residuals_++;
      }
    } else {
      // Rotation baked into cost function (default behavior)
      if (feature_id < image.angular_cholesky_xy.size() &&
          feature_id < image.angular_stddevs_z.size()) {
        const Eigen::Vector3d& chol = image.angular_cholesky_xy[feature_id];
        const double sigma_z =
            std::max(1e-9, image.angular_stddevs_z[feature_id]);
        cost_function =
            MahalanobisBATADirectionalError::Create(v_ik,
                                                    image.cam_from_world.rotation(),
                                                    chol[0],
                                                    chol[1],
                                                    chol[2],
                                                    sigma_z);
        mahalanobis_bata_residuals_++;
      } else if (feature_id < image.angular_stddevs.size()) {
        const Eigen::Vector2d& angular_std = image.angular_stddevs[feature_id];
        const double sigma_x = std::max(1e-9, angular_std[0]);
        const double sigma_y = std::max(1e-9, angular_std[1]);
        const double sigma_z = 0.5 * (sigma_x + sigma_y);
        cost_function = WeightedBATADirectionalError::Create(
            v_ik, image.cam_from_world.rotation(), sigma_x, sigma_y, sigma_z);
        diagonal_bata_residuals_++;
      } else {
        cost_function = BATAPairwiseDirectionError::Create(v_ik);
        unweighted_bata_residuals_++;
      }
    }

    if (cost_function == nullptr) {
      LOG(ERROR) << "Failed to create BATA cost function for image " << image_id
                 << " feature " << feature_id;
      return;
    }

    double& d_ik = scales_.emplace_back(1);
    if (!options_.generate_scales && tracks[track_id].is_initialized) {
      const Eigen::Vector3d X_k_minus_c_i =
          tracks[track_id].xyz - image.cam_from_world.translation();
      d_ik =
          std::max(1e-5, v_ik.dot(X_k_minus_c_i) / X_k_minus_c_i.squaredNorm());
    }
    // Select loss function based on observation type and inlier status
    // For non-LC observations, check if it's an inlier (use trivial loss)
    // or track anchor (use trackstart loss)
    bool use_inlier_loss = !is_lc_observation &&
                           feature_id < image.is_inlier.size() &&
                           image.is_inlier[feature_id];
    bool is_track_anchor_obs = !is_lc_observation &&
                               feature_id < image.is_track_anchor.size() &&
                               image.is_track_anchor[feature_id];
    ceres::LossFunction* geometry_loss;
    if (is_lc_observation) {
      geometry_loss = cached_loss_lc_geometry_.get();
    } else if (is_track_anchor_obs) {
      geometry_loss = cached_loss_normal_geometry_trackstart_.get();
    } else if (use_inlier_loss) {
      geometry_loss = cached_loss_normal_geometry_inlier_.get();
    } else {
      geometry_loss = cached_loss_normal_geometry_.get();
    }
    // Add residual block: include rotation as parameter if optimizing rotations
    if (options_.optimize_rotations) {
      problem_->AddResidualBlock(cost_function,
                                 geometry_loss,
                                 image.cam_from_world.rotation().coeffs().data(),
                                 image.cam_from_world.translation().data(),
                                 tracks[track_id].xyz.data(),
                                 &d_ik);
    } else {
      problem_->AddResidualBlock(cost_function,
                                 geometry_loss,
                                 image.cam_from_world.translation().data(),
                                 tracks[track_id].xyz.data(),
                                 &d_ik);
    }
    problem_->SetParameterLowerBound(&d_ik, 0, 1e-5);
    if (is_lc_observation) {
      lc_residuals_++;
    } else {
      normal_residuals_++;
      if (is_track_anchor_obs) {
        track_anchor_residuals_++;
      } else if (use_inlier_loss) {
        normal_inlier_residuals_++;
      } else {
        normal_outlier_residuals_++;
      }
    }
  } else {
    switch (options_.point_constraint_type) {
      case GlobalPositionerOptions::PointConstraintType::SPLIT_METRIC_DEPTH: {
        // Add both BATA (directional, with per-observation d_ik) and
        // MetricDepthError (camera-frame Z vs metric depth prior).
        // Skip if depth is invalid (guarded by outer use_depth).
        depth_constraints_++;

        // 1) Weighted BATA with d_ik:
        //    - First check for Mahalanobis (full 2x2 covariance via Cholesky)
        //    - Then fall back to diagonal weighting (angular_stddevs)
        //    - Finally fall back to unweighted BATA
        //    - Branch based on optimize_rotations for *WithRotation variants
        ceres::CostFunction* cost_dir = nullptr;
        if (options_.optimize_rotations) {
          // Rotation as optimization parameter - use bearing_cam (camera frame)
          if (feature_id < image.angular_cholesky_xy.size() &&
              feature_id < image.angular_stddevs_z.size()) {
            const Eigen::Vector3d& chol = image.angular_cholesky_xy[feature_id];
            const double sigma_z =
                std::max(1e-9, image.angular_stddevs_z[feature_id]);
            cost_dir = MahalanobisBATADirectionalErrorWithRotation::Create(
                bearing_cam, chol[0], chol[1], chol[2], sigma_z);
            mahalanobis_bata_residuals_++;
          } else if (feature_id < image.angular_stddevs.size()) {
            const Eigen::Vector2d& angular_std =
                image.angular_stddevs[feature_id];
            const double sigma_x = std::max(1e-9, angular_std[0]);
            const double sigma_y = std::max(1e-9, angular_std[1]);
            const double sigma_z = 0.5 * (sigma_x + sigma_y);
            cost_dir = WeightedBATADirectionalErrorWithRotation::Create(
                bearing_cam, sigma_x, sigma_y, sigma_z);
            diagonal_bata_residuals_++;
          }
          if (cost_dir == nullptr) {
            cost_dir = BATAPairwiseDirectionErrorWithRotation::Create(bearing_cam);
            unweighted_bata_residuals_++;
          }
        } else {
          // Rotation baked into cost function (default behavior)
          if (feature_id < image.angular_cholesky_xy.size() &&
              feature_id < image.angular_stddevs_z.size()) {
            const Eigen::Vector3d& chol = image.angular_cholesky_xy[feature_id];
            const double sigma_z =
                std::max(1e-9, image.angular_stddevs_z[feature_id]);
            cost_dir = MahalanobisBATADirectionalError::Create(
                v_ik,
                image.cam_from_world.rotation(),
                chol[0],
                chol[1],
                chol[2],
                sigma_z);
            mahalanobis_bata_residuals_++;
          } else if (feature_id < image.angular_stddevs.size()) {
            const Eigen::Vector2d& angular_std =
                image.angular_stddevs[feature_id];
            const double sigma_x = std::max(1e-9, angular_std[0]);
            const double sigma_y = std::max(1e-9, angular_std[1]);
            const double sigma_z = 0.5 * (sigma_x + sigma_y);
            cost_dir = WeightedBATADirectionalError::Create(
                v_ik, image.cam_from_world.rotation(), sigma_x, sigma_y, sigma_z);
            diagonal_bata_residuals_++;
          }
          if (cost_dir == nullptr) {
            cost_dir = BATAPairwiseDirectionError::Create(v_ik);
            unweighted_bata_residuals_++;
          }
        }
        if (cost_dir) {
          double& d_ik = scales_.emplace_back(1.0);
          if (!options_.generate_scales && tracks[track_id].is_initialized) {
            const Eigen::Vector3d X_k_minus_c_i =
                tracks[track_id].xyz - image.cam_from_world.translation();
            d_ik = std::max(
                1e-5, v_ik.dot(X_k_minus_c_i) / X_k_minus_c_i.squaredNorm());
          }
          // Select loss function based on observation type and inlier status
          // or track anchor status
          bool use_inlier_loss_geom = !is_lc_observation &&
                                      feature_id < image.is_inlier.size() &&
                                      image.is_inlier[feature_id];
          bool is_track_anchor_obs_geom =
              !is_lc_observation && feature_id < image.is_track_anchor.size() &&
              image.is_track_anchor[feature_id];
          ceres::LossFunction* geometry_loss;
          if (is_lc_observation) {
            geometry_loss = cached_loss_lc_geometry_.get();
          } else if (is_track_anchor_obs_geom) {
            geometry_loss = cached_loss_normal_geometry_trackstart_.get();
          } else if (use_inlier_loss_geom) {
            geometry_loss = cached_loss_normal_geometry_inlier_.get();
          } else {
            geometry_loss = cached_loss_normal_geometry_.get();
          }
          // Add residual block: include rotation as parameter if optimizing
          if (options_.optimize_rotations) {
            problem_->AddResidualBlock(cost_dir,
                                       geometry_loss,
                                       image.cam_from_world.rotation().coeffs().data(),
                                       image.cam_from_world.translation().data(),
                                       tracks[track_id].xyz.data(),
                                       &d_ik);
          } else {
            problem_->AddResidualBlock(cost_dir,
                                       geometry_loss,
                                       image.cam_from_world.translation().data(),
                                       tracks[track_id].xyz.data(),
                                       &d_ik);
          }
          problem_->SetParameterLowerBound(&d_ik, 0, 1e-5);
          if (is_lc_observation) {
            lc_residuals_++;
          } else {
            normal_residuals_++;
            if (is_track_anchor_obs_geom) {
              track_anchor_residuals_++;
            } else if (use_inlier_loss_geom) {
              normal_inlier_residuals_++;
            } else {
              normal_outlier_residuals_++;
            }
          }
        } else {
          LOG(ERROR) << "Failed to create BATAPairwiseDirectionError";
        }

        // 2) Metric depth residual (priors are Z-depth in camera frame)
        const double depth_prior_local = image.depth_priors[feature_id];
        const double depth_stddev_local =
            feature_id < image.depth_prior_stddevs.size()
                ? std::max(1e-6, image.depth_prior_stddevs[feature_id])
                : 1.0;

        // Get or create the per-image depth map scale parameter
        auto it = dmap_scales_.find(image_id);
        if (it == dmap_scales_.end()) {
          // Initialize: if log-space, initialize to log(1.0) = 0.0, else 1.0
          double initial_value =
              options_.use_log_scale_for_depth_map_scales ? 0.0 : 1.0;
          it = dmap_scales_.emplace(image_id, initial_value).first;
          dmap_scale_observation_counts_[image_id] = 0;
        }

        // Increment observation count for this image
        // The scale prior will be added later with weight proportional to this
        // count
        dmap_scale_observation_counts_[image_id]++;

        double& dmap_scale = it->second;

        // Create depth cost function: branch based on optimize_rotations
        ceres::CostFunction* cost_depth = nullptr;
        if (options_.optimize_rotations) {
          // Rotation as optimization parameter
          cost_depth = MetricDepthErrorWithRotation::Create(
              depth_prior_local,
              depth_stddev_local,
              options_.use_log_scale_for_depth_map_scales,
              options_.use_log_residual_for_depth,
              options_.zero_residual_behind_camera,
              options_.smooth_log_linear_transition,
              options_.log_linear_threshold);
        } else {
          // Rotation baked into cost function (default behavior)
          const Eigen::Quaterniond rotation(image.cam_from_world.rotation());
          cost_depth = MetricDepthError::Create(
              rotation,
              depth_prior_local,
              depth_stddev_local,
              options_.use_log_scale_for_depth_map_scales,
              options_.use_log_residual_for_depth,
              options_.zero_residual_behind_camera,
              options_.smooth_log_linear_transition,
              options_.log_linear_threshold);
        }
        if (cost_depth) {
          // Select loss function based on observation type, outlier status, and
          // inlier status
          ceres::LossFunction* depth_loss = nullptr;
          // Check if this is a track anchor non-LC observation
          bool is_track_anchor_obs_depth =
              !is_lc_observation && !use_soft_depth_loss &&
              feature_id < image.is_track_anchor.size() &&
              image.is_track_anchor[feature_id];
          // Check if this is an inlier non-LC observation (compute before
          // if/else for counter scope)
          bool use_inlier_loss_depth = !is_lc_observation &&
                                       !use_soft_depth_loss &&
                                       feature_id < image.is_inlier.size() &&
                                       image.is_inlier[feature_id];
          // Check if this is an MDRP depth outlier (use robust loss)
          bool use_mdrp_depth_outlier_loss =
              !is_lc_observation && !use_soft_depth_loss &&
              !use_inlier_loss_depth &&
              feature_id < image.is_depth_outlier.size() &&
              image.is_depth_outlier[feature_id];
          if (use_soft_depth_loss) {
            // Non-LC outlier: use soft loss (Huber scale=1, weight=1)
            depth_loss = GetLossOutlierDepth().get();
            VLOG(2)
                << "Adding depth residual with SOFT loss for outlier: image "
                << image_id << " track " << track_id;
          } else {
            if (is_lc_observation) {
              depth_loss = cached_loss_lc_depth_.get();
            } else if (is_track_anchor_obs_depth) {
              // Track anchor: use configurable loss (huber by default)
              depth_loss = cached_loss_normal_depth_trackstart_.get();
              track_anchor_depth_residuals_++;
            } else if (use_inlier_loss_depth) {
              depth_loss = cached_loss_normal_depth_inlier_.get();
            } else if (use_mdrp_depth_outlier_loss) {
              // MDRP depth outlier: use configurable robust loss
              // Use the pre-cached loss function (not the getter which
              // recreates it)
              depth_loss = cached_loss_normal_depth_outlier_.get();
              mdrp_depth_outlier_residuals_++;
            } else {
              depth_loss = cached_loss_normal_depth_.get();
            }
            VLOG(2) << "Adding depth residual for image " << image_id
                    << " track " << track_id << " (is_lc=" << is_lc_observation
                    << ", is_track_anchor=" << is_track_anchor_obs_depth
                    << ", is_inlier=" << use_inlier_loss_depth
                    << ", is_mdrp_depth_outlier=" << use_mdrp_depth_outlier_loss
                    << ")";
          }
          // Add residual block: include rotation as parameter if optimizing
          if (options_.optimize_rotations) {
            problem_->AddResidualBlock(cost_depth,
                                       depth_loss,
                                       image.cam_from_world.rotation().coeffs().data(),
                                       image.cam_from_world.translation().data(),
                                       tracks[track_id].xyz.data(),
                                       &dmap_scale);
          } else {
            problem_->AddResidualBlock(cost_depth,
                                       depth_loss,
                                       image.cam_from_world.translation().data(),
                                       tracks[track_id].xyz.data(),
                                       &dmap_scale);
          }
          // Add a lower bound only for linear space (log-space automatically
          // enforces positivity)
          if (!options_.use_log_scale_for_depth_map_scales) {
            problem_->SetParameterLowerBound(&dmap_scale, 0, 1e-5);
          }
          if (is_lc_observation) {
            lc_residuals_++;
          } else {
            normal_residuals_++;
            if (use_inlier_loss_depth) {
              normal_inlier_residuals_++;
            } else {
              normal_outlier_residuals_++;
            }
          }
        } else {
          LOG(ERROR) << "Failed to create MetricDepthError";
        }

        break;
      }

      default: {
        LOG(ERROR) << "Unknown point constraint type: "
                   << static_cast<int>(options_.point_constraint_type);
        break;
      }
    }
  }
}

void GlobalPositioner::AddCamerasAndPointsToParameterGroups(
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<track_t, Track>& tracks) {
  // Create a custom ordering for Schur-based problems.
  options_.solver_base.solver_options.linear_solver_ordering.reset(
      new ceres::ParameterBlockOrdering);
  ceres::ParameterBlockOrdering* parameter_ordering =
      options_.solver_base.solver_options.linear_solver_ordering.get();

  // Add scale parameters to group 0 (per-observation, independent)
  // Only add scales that are actually in the problem
  for (double& scale : scales_) {
    if (problem_->HasParameterBlock(&scale)) {
      parameter_ordering->AddElementToGroup(&scale, 0);
    }
  }

  // Add point parameters to group 1.
  int group_id = 1;
  if (tracks.size() > 0) {
    for (auto& [track_id, track] : tracks) {
      if (problem_->HasParameterBlock(track.xyz.data()))
        parameter_ordering->AddElementToGroup(track.xyz.data(), group_id);
    }
    group_id++;
  }

  // Add camera parameters to group 2 if there are tracks, otherwise group 1.
  for (auto& [image_id, image] : images) {
    if (problem_->HasParameterBlock(image.cam_from_world.translation().data())) {
      parameter_ordering->AddElementToGroup(
          image.cam_from_world.translation().data(), group_id);
    }
    // Add rotation parameters to same group when optimizing rotations
    if (options_.optimize_rotations) {
      double* rotation_ptr = image.cam_from_world.rotation().coeffs().data();
      if (problem_->HasParameterBlock(rotation_ptr)) {
        parameter_ordering->AddElementToGroup(rotation_ptr, group_id);
      }
    }
  }

  // Add depth map scale parameters to the same group as cameras (group 2)
  // These are per-image scales, shared across multiple observations, so they
  // should be eliminated with cameras, not with independent per-observation
  // scales
  for (auto& [image_id, dmap_scale] : dmap_scales_) {
    if (problem_->HasParameterBlock(&dmap_scale)) {
      parameter_ordering->AddElementToGroup(&dmap_scale, group_id);
    }
  }
}

void GlobalPositioner::ParameterizeVariables(
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<track_t, Track>& tracks) {
  // For the global positioning, do not set any camera to be constant for easier
  // convergence

  // If filtering by image IDs, set cameras from non-specified images to
  // constant
  if (!image_ids_to_optimize_.empty()) {
    for (auto& [image_id, image] : images) {
      if (image_ids_to_optimize_.find(image_id) ==
          image_ids_to_optimize_.end()) {
        if (problem_->HasParameterBlock(
                image.cam_from_world.translation().data())) {
          problem_->SetParameterBlockConstant(
              image.cam_from_world.translation().data());
        }
      }
    }
  }

  // If do not optimize the positions, set the camera positions to be constant
  if (!options_.optimize_positions) {
    for (auto& [image_id, image] : images)
      if (problem_->HasParameterBlock(image.cam_from_world.translation().data()))
        problem_->SetParameterBlockConstant(
            image.cam_from_world.translation().data());
  }

  // Handle rotation optimization: set quaternion manifold and optionally
  // set to constant
  if (options_.optimize_rotations) {
    for (auto& [image_id, image] : images) {
      double* rotation_ptr = image.cam_from_world.rotation().coeffs().data();
      if (problem_->HasParameterBlock(rotation_ptr)) {
        // Set quaternion manifold (colmap provides this utility)
        colmap::SetQuaternionManifold(problem_.get(), rotation_ptr);
      }
    }
    LOG(INFO) << "Set quaternion manifold for " << images.size()
              << " camera rotations";
  }

  // If do not optimize the points, set the track positions to be constant
  if (!options_.optimize_points) {
    for (auto& [track_id, track] : tracks) {
      if (problem_->HasParameterBlock(track.xyz.data())) {
        problem_->SetParameterBlockConstant(track.xyz.data());
      }
    }
  }

  // If do not optimize the scales, set the scales to be constant
  if (!options_.optimize_scales) {
    for (double& scale : scales_) {
      problem_->SetParameterBlockConstant(&scale);
    }
  }

  // If do not optimize the depth map scales, set them to be constant
  if (!options_.optimize_depth_map_scales) {
    LOG(INFO) << "Setting depth map scales to be constant";
    for (auto& [image_id, dmap_scale] : dmap_scales_) {
      problem_->SetParameterBlockConstant(&dmap_scale);
    }
  }
  // else if (!dmap_scales_.empty()) {
  //   // CRITICAL: Fix BOTH the scale AND camera position for one observation
  //   to
  //   // remove gauge freedom This removes both scale ambiguity and translation
  //   // ambiguity
  //   auto first_scale_it = dmap_scales_.begin();
  //   image_t anchor_image_id = first_scale_it->first.first;
  //   feature_t anchor_feature_id = first_scale_it->first.second;
  //   double anchor_scale = first_scale_it->second;

  //   LOG(INFO) << "Fixing depth map scale for observation (" <<
  //   anchor_image_id
  //             << ", " << anchor_feature_id
  //             << ") to remove scale gauge freedom: " << anchor_scale;
  //   problem_->SetParameterBlockConstant(&first_scale_it->second);

  //   // Also fix the camera position for the same image to remove translation
  //   // gauge freedom
  //   if (images.find(anchor_image_id) != images.end()) {
  //     Image& anchor_image = images[anchor_image_id];
  //     if (problem_->HasParameterBlock(
  //             anchor_image.cam_from_world.translation.data())) {
  //       LOG(INFO) << "Fixing camera position for image " << anchor_image_id
  //                 << " to remove translation gauge freedom";
  //       problem_->SetParameterBlockConstant(
  //           anchor_image.cam_from_world.translation.data());
  //     }
  //   }
  // }

  // Set up the options for the solver
  // Do not use iterative solvers, for its suboptimal performance.
  if (tracks.size() > 0) {
    options_.solver_base.solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
    options_.solver_base.solver_options.preconditioner_type = ceres::CLUSTER_TRIDIAGONAL;
  } else {
    options_.solver_base.solver_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options_.solver_base.solver_options.preconditioner_type = ceres::JACOBI;
  }
}

void GlobalPositioner::ConvertResults(
    std::unordered_map<image_t, Image>& images) {
  // translation now stores the camera position, needs to convert back to
  // translation
  for (auto& [image_id, image] : images) {
    image.cam_from_world.translation() =  // TODO(rigid3d-sweep): manual review needed
        -(image.cam_from_world.rotation() * image.cam_from_world.translation());
  }
}

// Helper methods to get loss functions from current options
// These ensure that changes to options_ take effect even after SetupProblem()
std::shared_ptr<ceres::LossFunction> GlobalPositioner::GetLossNormalGeometry()
    const {
  // Always recreate from current options to pick up changes
  // Cache as member variable to keep it alive for Ceres (DO_NOT_TAKE_OWNERSHIP)
  cached_loss_normal_geometry_ = CreateLossFromConfig(
      options_.loss_normal_geometry);
  VLOG(2) << "Updated loss_normal_geometry: "
          << options_.loss_normal_geometry.name
          << " (scale=" << options_.loss_normal_geometry.scale
          << ", weight=" << options_.loss_normal_geometry.weight << ")";
  return cached_loss_normal_geometry_;
}

std::shared_ptr<ceres::LossFunction> GlobalPositioner::GetLossNormalDepth()
    const {
  // Always recreate from current options to pick up changes
  cached_loss_normal_depth_ =
      CreateLossFromConfig(options_.loss_normal_depth);
  VLOG(2) << "Updated loss_normal_depth: " << options_.loss_normal_depth.name
          << " (scale=" << options_.loss_normal_depth.scale
          << ", weight=" << options_.loss_normal_depth.weight << ")";
  return cached_loss_normal_depth_;
}

std::shared_ptr<ceres::LossFunction> GlobalPositioner::GetLossLCGeometry()
    const {
  // Always recreate from current options to pick up changes
  cached_loss_lc_geometry_ =
      CreateLossFromConfig(options_.loss_lc_geometry);
  VLOG(2) << "Updated loss_lc_geometry: " << options_.loss_lc_geometry.name
          << " (scale=" << options_.loss_lc_geometry.scale
          << ", weight=" << options_.loss_lc_geometry.weight << ")";
  return cached_loss_lc_geometry_;
}

std::shared_ptr<ceres::LossFunction> GlobalPositioner::GetLossLCDepth() const {
  // Always recreate from current options to pick up changes
  cached_loss_lc_depth_ =
      CreateLossFromConfig(options_.loss_lc_depth);
  VLOG(2) << "Updated loss_lc_depth: " << options_.loss_lc_depth.name
          << " (scale=" << options_.loss_lc_depth.scale
          << ", weight=" << options_.loss_lc_depth.weight << ")";
  return cached_loss_lc_depth_;
}

std::shared_ptr<ceres::LossFunction> GlobalPositioner::GetLossScalePrior()
    const {
  // Always recreate from current options to pick up changes
  cached_loss_scale_prior_ =
      CreateLossFromConfig(options_.loss_scale_prior);
  VLOG(2) << "Updated loss_scale_prior: " << options_.loss_scale_prior.name
          << " (scale=" << options_.loss_scale_prior.scale
          << ", weight=" << options_.loss_scale_prior.weight << ")";
  return cached_loss_scale_prior_;
}

std::shared_ptr<ceres::LossFunction> GlobalPositioner::GetLossOutlierDepth()
    const {
  // If already created and cached, return it
  if (cached_loss_outlier_depth_) {
    return cached_loss_outlier_depth_;
  }

  // Create soft loss for outliers: default Huber with scale=1, weight=1
  // This is a fixed configuration, not from options_
  // ScaledLoss takes ownership of the HuberLoss (TAKE_OWNERSHIP)
  // The shared_ptr wrapper manages the ScaledLoss (default deleter)
  cached_loss_outlier_depth_ =
      std::shared_ptr<ceres::LossFunction>(new ceres::ScaledLoss(
          new ceres::HuberLoss(1.0), 1.0, ceres::TAKE_OWNERSHIP));
  VLOG(2) << "Created soft loss for outlier depth constraints: Huber "
          << "(scale=1, weight=1)";
  return cached_loss_outlier_depth_;
}

std::shared_ptr<ceres::LossFunction>
GlobalPositioner::GetLossNormalGeometryInlier() const {
  // Always recreate from current options to pick up changes
  cached_loss_normal_geometry_inlier_ =
      CreateLossFromConfig(
          options_.loss_normal_geometry_inlier);
  VLOG(2) << "Updated loss_normal_geometry_inlier: "
          << options_.loss_normal_geometry_inlier.name
          << " (scale=" << options_.loss_normal_geometry_inlier.scale
          << ", weight=" << options_.loss_normal_geometry_inlier.weight << ")";
  return cached_loss_normal_geometry_inlier_;
}

std::shared_ptr<ceres::LossFunction>
GlobalPositioner::GetLossNormalDepthInlier() const {
  // Always recreate from current options to pick up changes
  cached_loss_normal_depth_inlier_ =
      CreateLossFromConfig(
          options_.loss_normal_depth_inlier);
  VLOG(2) << "Updated loss_normal_depth_inlier: "
          << options_.loss_normal_depth_inlier.name
          << " (scale=" << options_.loss_normal_depth_inlier.scale
          << ", weight=" << options_.loss_normal_depth_inlier.weight << ")";
  return cached_loss_normal_depth_inlier_;
}

std::shared_ptr<ceres::LossFunction>
GlobalPositioner::GetLossNormalDepthOutlier() const {
  // Always recreate from current options to pick up changes
  cached_loss_normal_depth_outlier_ =
      CreateLossFromConfig(
          options_.loss_normal_depth_outlier);
  VLOG(2) << "Updated loss_normal_depth_outlier: "
          << options_.loss_normal_depth_outlier.name
          << " (scale=" << options_.loss_normal_depth_outlier.scale
          << ", weight=" << options_.loss_normal_depth_outlier.weight << ")";
  return cached_loss_normal_depth_outlier_;
}

void GlobalPositioner::InitializeDepthMapScalesFromObservations(
    const std::unordered_map<image_t, Image>& images,
    const std::unordered_map<track_t, Track>& tracks) {
  // For each image, collect scale estimates: z_est / depth_prior
  std::map<image_t, std::vector<double>> image_scale_estimates;

  // Iterate through all tracks
  for (const auto& [track_id, track] : tracks) {
    // Check normal observations
    for (const auto& [image_id, feature_id] : track.observations) {
      // Skip if image not found or not registered
      auto img_it = images.find(image_id);
      if (img_it == images.end() || !img_it->second.is_registered) {
        continue;
      }

      const Image& image = img_it->second;

      // Skip if no valid depth prior
      if (feature_id >= image.depth_prior_validity.size() ||
          !image.depth_prior_validity[feature_id]) {
        continue;
      }

      // Skip if no depth prior available
      if (feature_id >= image.depth_priors.size()) {
        continue;
      }

      const double depth_prior = image.depth_priors[feature_id];
      if (depth_prior <= 1e-6) {
        continue;
      }

      // Get 3D point
      const Eigen::Vector3d& point_world = track.xyz;

      // Get camera center and rotation
      const Eigen::Vector3d& cam_center = image.cam_from_world.translation();
      const Eigen::Quaterniond& cam_rotation = image.cam_from_world.rotation();

      // Transform point to camera frame
      const Eigen::Vector3d point_vec_world = point_world - cam_center;
      const Eigen::Vector3d point_cam = cam_rotation * point_vec_world;

      // Get Z-depth in camera frame
      const double z_est = point_cam[2];

      // Skip if point is behind camera or too close
      if (z_est <= 1e-6) {
        continue;
      }

      // Compute scale estimate: z_est / depth_prior
      double scale_estimate = z_est / depth_prior;
      if (scale_estimate > 1e-6 && scale_estimate < 1e6) {  // Reasonable range
        image_scale_estimates[image_id].push_back(scale_estimate);
      }
    }

    // Check LC observations
    for (const auto& [image_id, feature_id] : track.lc_observations) {
      // Skip if image not found or not registered
      auto img_it = images.find(image_id);
      if (img_it == images.end() || !img_it->second.is_registered) {
        continue;
      }

      const Image& image = img_it->second;

      // Skip if no valid depth prior
      if (feature_id >= image.depth_prior_validity.size() ||
          !image.depth_prior_validity[feature_id]) {
        continue;
      }

      // Skip if no depth prior available
      if (feature_id >= image.depth_priors.size()) {
        continue;
      }

      const double depth_prior = image.depth_priors[feature_id];
      if (depth_prior <= 1e-6) {
        continue;
      }

      // Get 3D point
      const Eigen::Vector3d& point_world = track.xyz;

      // Get camera center and rotation
      const Eigen::Vector3d& cam_center = image.cam_from_world.translation();
      const Eigen::Quaterniond& cam_rotation = image.cam_from_world.rotation();

      // Transform point to camera frame
      const Eigen::Vector3d point_vec_world = point_world - cam_center;
      const Eigen::Vector3d point_cam = cam_rotation * point_vec_world;

      // Get Z-depth in camera frame
      const double z_est = point_cam[2];

      // Skip if point is behind camera or too close
      if (z_est <= 1e-6) {
        continue;
      }

      // Compute scale estimate: z_est / depth_prior
      double scale_estimate = z_est / depth_prior;
      if (scale_estimate > 1e-6 && scale_estimate < 1e6) {  // Reasonable range
        image_scale_estimates[image_id].push_back(scale_estimate);
      }
    }
  }

  // Compute median scale for each image and initialize dmap_scales_
  int initialized_count = 0;
  for (const auto& [image_id, scale_estimates] : image_scale_estimates) {
    if (scale_estimates.empty()) {
      continue;
    }

    // Compute median
    std::vector<double> sorted_estimates = scale_estimates;
    std::sort(sorted_estimates.begin(), sorted_estimates.end());
    double median_scale = sorted_estimates[sorted_estimates.size() / 2];

    // Convert to log-space if needed
    double initial_value = options_.use_log_scale_for_depth_map_scales
                               ? std::log(median_scale)
                               : median_scale;

    dmap_scales_[image_id] = initial_value;
    dmap_scale_observation_counts_[image_id] = 0;
    initialized_count++;

    VLOG(1) << "Initialized depth map scale for image " << image_id
            << ": median=" << median_scale << " (from "
            << scale_estimates.size() << " observations)";
  }

  LOG(INFO) << "Auto-initialized " << initialized_count
            << " depth map scales from observed 3D points (median method)";
}

void GlobalPositioner::FilterDepthOutliers(
    const std::unordered_map<image_t, Image>& images,
    const std::unordered_map<track_t, Track>& tracks) {
  int total_checked = 0;
  int outliers_found = 0;

  // Per-image statistics
  std::map<image_t, int> image_checked;
  std::map<image_t, int> image_outliers;
  std::map<image_t, std::vector<std::pair<feature_t, double>>>
      image_outlier_examples;  // feature_id, log_diff

  // Iterate through all tracks
  for (const auto& [track_id, track] : tracks) {
    // Check normal observations
    for (const auto& [image_id, feature_id] : track.observations) {
      // Skip if image not found or not registered
      auto img_it = images.find(image_id);
      if (img_it == images.end() || !img_it->second.is_registered) {
        continue;
      }

      const Image& image = img_it->second;

      // Skip if no valid depth prior
      if (feature_id >= image.depth_prior_validity.size() ||
          !image.depth_prior_validity[feature_id]) {
        continue;
      }

      // Skip if no depth prior or stddev available
      if (feature_id >= image.depth_priors.size() ||
          feature_id >= image.depth_prior_stddevs.size()) {
        continue;
      }

      const double depth_prior_raw = image.depth_priors[feature_id];
      const double stddev_rel = image.depth_prior_stddevs[feature_id];

      // Safety checks
      if (depth_prior_raw <= 1e-6 || stddev_rel <= 1e-9) {
        continue;
      }

      // Get depth map scale for this image (if available from
      // initial_dmap_scales) Scale the depth prior by the depth map scale
      double depth_prior = depth_prior_raw;
      auto scale_it = dmap_scales_.find(image_id);
      if (scale_it != dmap_scales_.end()) {
        // Convert from log-space if needed
        double dmap_scale = options_.use_log_scale_for_depth_map_scales
                                ? std::exp(scale_it->second)
                                : scale_it->second;
        depth_prior = dmap_scale * depth_prior_raw;
      }

      // Get 3D point
      const Eigen::Vector3d& point_world = track.xyz;

      // Get camera center and rotation
      const Eigen::Vector3d& cam_center = image.cam_from_world.translation();
      const Eigen::Quaterniond& cam_rotation = image.cam_from_world.rotation();

      // Transform point to camera frame
      const Eigen::Vector3d point_vec_world = point_world - cam_center;
      const Eigen::Vector3d point_cam = cam_rotation * point_vec_world;

      // Get Z-depth in camera frame
      const double z_est = point_cam[2];

      // Skip if point is behind camera or too close
      if (z_est <= 1e-6) {
        continue;
      }

      total_checked++;

      // Compute metric stddev: relative_stddev * scaled_depth_prior
      const double metric_std = stddev_rel * depth_prior;

      // Log-space check: |log(z_est) - log(scaled_depth_prior)| < 3 * log(1 +
      // metric_std/scaled_depth_prior)
      const double log_z_est = std::log(std::max(z_est, 1e-6));
      const double log_depth_prior = std::log(std::max(depth_prior, 1e-6));
      const double log_diff = std::abs(log_z_est - log_depth_prior);

      const double metric_std_over_depth =
          metric_std / std::max(depth_prior, 1e-6);
      const double threshold =
          3.0 * std::log(1.0 + std::max(metric_std_over_depth, 1e-6));

      // If outside range, mark as outlier
      if (log_diff >= threshold) {
        depth_outliers_.insert(std::make_pair(image_id, feature_id));
        outliers_found++;
        image_outliers[image_id]++;

        // Store example outliers (keep first 5 per image)
        if (image_outlier_examples[image_id].size() < 5) {
          image_outlier_examples[image_id].push_back(
              std::make_pair(feature_id, log_diff));
        }
      }
      image_checked[image_id]++;
    }

    // Check LC observations
    for (const auto& [image_id, feature_id] : track.lc_observations) {
      // Skip if image not found or not registered
      auto img_it = images.find(image_id);
      if (img_it == images.end() || !img_it->second.is_registered) {
        continue;
      }

      const Image& image = img_it->second;

      // Skip if no valid depth prior
      if (feature_id >= image.depth_prior_validity.size() ||
          !image.depth_prior_validity[feature_id]) {
        continue;
      }

      // Skip if no depth prior or stddev available
      if (feature_id >= image.depth_priors.size() ||
          feature_id >= image.depth_prior_stddevs.size()) {
        continue;
      }

      const double depth_prior_raw = image.depth_priors[feature_id];
      const double stddev_rel = image.depth_prior_stddevs[feature_id];

      // Safety checks
      if (depth_prior_raw <= 1e-6 || stddev_rel <= 1e-9) {
        continue;
      }

      // Get depth map scale for this image (if available from
      // initial_dmap_scales) Scale the depth prior by the depth map scale
      double depth_prior = depth_prior_raw;
      auto scale_it = dmap_scales_.find(image_id);
      if (scale_it != dmap_scales_.end()) {
        // Convert from log-space if needed
        double dmap_scale = options_.use_log_scale_for_depth_map_scales
                                ? std::exp(scale_it->second)
                                : scale_it->second;
        depth_prior = dmap_scale * depth_prior_raw;
      }

      // Get 3D point
      const Eigen::Vector3d& point_world = track.xyz;

      // Get camera center and rotation
      const Eigen::Vector3d& cam_center = image.cam_from_world.translation();
      const Eigen::Quaterniond& cam_rotation = image.cam_from_world.rotation();

      // Transform point to camera frame
      const Eigen::Vector3d point_vec_world = point_world - cam_center;
      const Eigen::Vector3d point_cam = cam_rotation * point_vec_world;

      // Get Z-depth in camera frame
      const double z_est = point_cam[2];

      // Skip if point is behind camera or too close
      if (z_est <= 1e-6) {
        continue;
      }

      total_checked++;

      // Compute metric stddev: relative_stddev * scaled_depth_prior
      const double metric_std = stddev_rel * depth_prior;

      // Log-space check: |log(z_est) - log(scaled_depth_prior)| < 3 * log(1 +
      // metric_std/scaled_depth_prior)
      const double log_z_est = std::log(std::max(z_est, 1e-6));
      const double log_depth_prior = std::log(std::max(depth_prior, 1e-6));
      const double log_diff = std::abs(log_z_est - log_depth_prior);

      const double metric_std_over_depth =
          metric_std / std::max(depth_prior, 1e-6);
      const double threshold =
          3.0 * std::log(1.0 + std::max(metric_std_over_depth, 1e-6));

      // If outside range, mark as outlier
      if (log_diff >= threshold) {
        depth_outliers_.insert(std::make_pair(image_id, feature_id));
        outliers_found++;
        image_outliers[image_id]++;

        // Store example outliers (keep first 5 per image)
        if (image_outlier_examples[image_id].size() < 5) {
          image_outlier_examples[image_id].push_back(
              std::make_pair(feature_id, log_diff));
        }
      }
      image_checked[image_id]++;
    }
  }

  // Log summary statistics
  LOG(INFO) << "Depth outlier filtering: checked " << total_checked
            << " observations, found " << outliers_found << " outliers ("
            << (total_checked > 0 ? 100.0 * outliers_found / total_checked
                                  : 0.0)
            << "%)";

  // Log per-image statistics (top 10 images with most outliers)
  if (!image_outliers.empty()) {
    // Sort by outlier count
    std::vector<std::pair<image_t, int>> sorted_images;
    for (const auto& [img_id, count] : image_outliers) {
      sorted_images.push_back(std::make_pair(img_id, count));
    }
    std::sort(sorted_images.begin(),
              sorted_images.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    LOG(INFO) << "Top 10 images by outlier count:";
    int log_count = 0;
    for (const auto& [img_id, outlier_count] : sorted_images) {
      if (log_count++ >= 10) break;
      int checked = image_checked[img_id];
      double pct = checked > 0 ? 100.0 * outlier_count / checked : 0.0;
      LOG(INFO) << "  Image " << img_id << ": " << outlier_count
                << " outliers / " << checked << " checked (" << pct << "%)";

      // Log example outliers for this image
      if (image_outlier_examples.find(img_id) != image_outlier_examples.end() &&
          !image_outlier_examples[img_id].empty()) {
        std::stringstream examples;
        examples << "    Examples (feature_id, log_diff): ";
        for (size_t i = 0; i < image_outlier_examples[img_id].size(); ++i) {
          if (i > 0) examples << ", ";
          examples << "(" << image_outlier_examples[img_id][i].first << ", "
                   << std::fixed << std::setprecision(3)
                   << image_outlier_examples[img_id][i].second << ")";
        }
        LOG(INFO) << examples.str();
      }
    }
  }

  // Log distribution statistics
  if (total_checked > 0) {
    double outlier_rate = 100.0 * outliers_found / total_checked;
    if (outlier_rate > 10.0) {
      LOG(WARNING) << "High outlier rate detected: " << std::fixed
                   << std::setprecision(2) << outlier_rate
                   << "% - consider checking depth map quality";
    } else if (outlier_rate < 0.1) {
      LOG(INFO) << "Low outlier rate: " << std::fixed << std::setprecision(2)
                << outlier_rate << "% - depth priors appear well-aligned";
    }
  }
}

void GlobalPositioner::AddRelativePoseConstraints(
    const ViewGraph& view_graph,
    std::unordered_map<image_t, Image>& images,
    const std::vector<image_pair_t>& consecutive_pair_ids) {
  // Feature disabled or no pairs - exactly current behavior
  if (!options_.use_relative_pose_constraints || consecutive_pair_ids.empty()) {
    return;
  }

  // Create the loss function for relative pose constraints
  cached_loss_relative_pose_ = CreateLossFromConfig(
      options_.loss_relative_pose);

  LOG(INFO) << "Adding relative pose constraints for "
            << consecutive_pair_ids.size() << " consecutive pairs";
  LOG(INFO) << "  Loss function: " << options_.loss_relative_pose.name
            << " (scale=" << options_.loss_relative_pose.scale
            << ", weight=" << options_.loss_relative_pose.weight << ")";

  for (const auto& pair_id : consecutive_pair_ids) {
    auto it = view_graph.image_pairs.find(pair_id);
    if (it == view_graph.image_pairs.end() || !it->second.is_valid) {
      continue;
    }

    const ImagePair& image_pair = it->second;

    // Check that both images are registered
    auto img1_it = images.find(image_pair.image_id1);
    auto img2_it = images.find(image_pair.image_id2);
    if (img1_it == images.end() || img2_it == images.end()) {
      continue;
    }
    if (!img1_it->second.is_registered || !img2_it->second.is_registered) {
      continue;
    }

    Image& image1 = img1_it->second;
    Image& image2 = img2_it->second;

    // The MDRP relative pose (R_21, t_21) defines: p_cam2 = R_21 * p_cam1 +
    // t_21 where t_21 is cam1's center in cam2's local frame.
    //
    // To find the expected displacement c_2 - c_1 in WORLD frame:
    // 1. t_21 is in cam2's MDRP local frame
    // 2. R_21^T * t_21 transforms it to cam1's local frame
    // 3. R_1^T * (R_21^T * t_21) transforms it to world frame
    //
    // The relationship: c_1 - c_2 = R_1^T * R_21^T * t_21
    // So: c_2 - c_1 = -R_1^T * R_21^T * t_21
    //
    // We use the global R_1 (from image1) and the MDRP R_21 (from image_pair)
    // This is correct even if global rotation R_2 != R_21 * R_1

    // Get rotation of cam1 in world frame: R_w1 = R_1^T
    Eigen::Matrix3d R_w1 =
        image1.cam_from_world.rotation().toRotationMatrix().transpose();

    // Get MDRP relative rotation: R_21
    Eigen::Matrix3d R_21 =
        image_pair.cam2_from_cam1.rotation().toRotationMatrix();

    // Get MDRP relative translation: t_21
    Eigen::Vector3d t_21 = image_pair.cam2_from_cam1.translation();

    // Expected displacement in world frame: c_2 - c_1 = -R_w1 * R_21^T * t_21
    // For the cost function, we pass the rotation that transforms the expected
    // displacement to world frame, which is R_w1 * R_21^T
    Eigen::Matrix3d R_transform = R_w1 * R_21.transpose();

    // The expected displacement is: -R_transform * t_21 = R_transform * (-t_21)
    Eigen::Vector3d t_expected = -t_21;

    // DEBUG: Log values for specific pairs
    if (image_pair.image_id1 == 120168 || image_pair.image_id2 == 120168 ||
        image_pair.image_id1 == 120170 || image_pair.image_id2 == 120170) {
      Eigen::Vector3d expected_disp = R_transform * t_expected;
      LOG(INFO) << "DEBUG RelPose pair (" << image_pair.image_id1 << ", "
                << image_pair.image_id2 << "):";
      LOG(INFO) << "  t_21 = [" << t_21.transpose()
                << "], norm=" << t_21.norm();
      LOG(INFO) << "  R_w1 row0 = [" << R_w1.row(0) << "]";
      LOG(INFO) << "  R_21 row0 = [" << R_21.row(0) << "]";
      LOG(INFO) << "  R_transform row0 = [" << R_transform.row(0) << "]";
      LOG(INFO) << "  expected_disp = [" << expected_disp.transpose()
                << "], norm=" << expected_disp.norm();
      LOG(INFO) << "  c1_init = ["
                << image1.cam_from_world.translation().transpose() << "]";
      LOG(INFO) << "  c2_init = ["
                << image2.cam_from_world.translation().transpose() << "]";
    }

    // Get covariance (use default if not set)
    // The covariance is defined in cam2's local frame, so we transform it
    // to world frame using the same R_transform
    Eigen::Matrix3d cov_t = image_pair.cov_t;
    if (cov_t.isZero() || cov_t.norm() < 1e-10) {
      double sigma = options_.relative_pose_default_stddev;
      cov_t = Eigen::Matrix3d::Identity() * (sigma * sigma);
    }

    ceres::CostFunction* cost =
        RelativeTranslationError::Create(R_transform, t_expected, cov_t);
    problem_->AddResidualBlock(cost,
                               cached_loss_relative_pose_.get(),
                               image1.cam_from_world.translation().data(),
                               image2.cam_from_world.translation().data());

    relative_pose_constraints_++;
  }

  LOG(INFO) << "Added " << relative_pose_constraints_
            << " relative pose constraints";
}

void GlobalPositioner::AddRotationPriors(
    std::unordered_map<image_t, Image>& images) {
  // Skip if rotation optimization or regularization is disabled
  if (!options_.optimize_rotations || !options_.regularize_rotations) {
    return;
  }

  // Create the loss function for rotation priors
  cached_loss_rotation_prior_ = CreateLossFromConfig(
      options_.loss_rotation_prior);

  LOG(INFO) << "Adding rotation prior constraints for " << images.size()
            << " images";
  LOG(INFO) << "  Sigma: " << options_.rotation_prior_sigma << " radians";
  LOG(INFO) << "  Loss function: " << options_.loss_rotation_prior.name
            << " (scale=" << options_.loss_rotation_prior.scale
            << ", weight=" << options_.loss_rotation_prior.weight << ")";

  for (auto& [image_id, image] : images) {
    // Skip if we don't have an initial rotation for this image
    auto it = initial_rotations_.find(image_id);
    if (it == initial_rotations_.end()) {
      continue;
    }

    // Skip images that are not being optimized
    if (!image_ids_to_optimize_.empty() &&
        image_ids_to_optimize_.find(image_id) == image_ids_to_optimize_.end()) {
      continue;
    }

    // Check that the rotation is a parameter block
    double* rotation_ptr = image.cam_from_world.rotation().coeffs().data();
    if (!problem_->HasParameterBlock(rotation_ptr)) {
      continue;
    }

    // Create rotation prior cost function
    ceres::CostFunction* cost = RotationPriorError::Create(
        it->second,  // Initial rotation from rotation averaging
        options_.rotation_prior_sigma);

    if (cost == nullptr) {
      LOG(ERROR) << "Failed to create RotationPriorError for image " << image_id;
      continue;
    }

    problem_->AddResidualBlock(cost,
                               cached_loss_rotation_prior_.get(),
                               rotation_ptr);
    rotation_prior_constraints_++;
  }

  LOG(INFO) << "Added " << rotation_prior_constraints_
            << " rotation prior constraints";
}

std::shared_ptr<ceres::LossFunction> GlobalPositioner::GetLossRotationPrior()
    const {
  cached_loss_rotation_prior_ = CreateLossFromConfig(
      options_.loss_rotation_prior);
  VLOG(2) << "Updated loss_rotation_prior: " << options_.loss_rotation_prior.name
          << " (scale=" << options_.loss_rotation_prior.scale
          << ", weight=" << options_.loss_rotation_prior.weight << ")";
  return cached_loss_rotation_prior_;
}

}  // namespace colmap::glomap
