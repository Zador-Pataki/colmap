#include "colmap/sfm/global_positioning_glomap.h"

#include "colmap/sfm/cost_function_glomap.h"

#include "colmap/estimators/cost_functions/manifold.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <typeinfo>
#include <vector>

namespace colmap {
namespace glomap_ra {

namespace {
inline Eigen::Vector3d ImageCenter(const Image& image) {
  return image.cam_from_world.rotation().inverse() *
         -image.cam_from_world.translation();
}
}  // namespace


using ViewGraph = colmap::CorrespondenceGraph;
using ImagePair = colmap::CorrespondenceGraph::ImagePair;

namespace {

Eigen::Vector3d RandVector3d(std::mt19937& random_generator,
                             double low,
                             double high) {
  std::uniform_real_distribution<double> distribution(low, high);
  return Eigen::Vector3d(distribution(random_generator),
                         distribution(random_generator),
                         distribution(random_generator));
}

// =====================================================================
// GP debug instrumentation. Set GP_DEBUG_DUMP=/path/to/file.txt to enable
// per-stage dumps of image translations + track xyz + Ceres convergence
// trace. Zero cost when env var is unset (early-return at top of each
// function). Used by past port debugging — keep around for future Ceres-
// on-glomap regressions. Set GP_SEED=N to override the default seed=1
// for the random init RNG (per Solve invocation).
// =====================================================================
const char* GpDebugDumpPath() {
  static const char* path = std::getenv("GP_DEBUG_DUMP");
  return path;
}

void GpDebugDumpInputs(const char* tag,
                       const std::unordered_map<image_t, Image>& images,
                       const std::unordered_map<track_t, Track>& tracks) {
  const char* path = GpDebugDumpPath();
  if (path == nullptr) return;
  FILE* f = std::fopen(path, "a");
  if (f == nullptr) return;
  std::vector<image_t> img_ids;
  img_ids.reserve(images.size());
  for (const auto& kv : images) img_ids.push_back(kv.first);
  std::sort(img_ids.begin(), img_ids.end());
  for (image_t id : img_ids) {
    const auto& img = images.at(id);
    const auto& t = img.cam_from_world.translation();
    const auto q = img.cam_from_world.rotation().coeffs();
    std::fprintf(f, "%s|image|%llu|t=%.17g,%.17g,%.17g|q=%.17g,%.17g,%.17g,%.17g|nfeat=%llu|ndepth=%llu|reg=%d\n",
                 tag, static_cast<unsigned long long>(id),
                 t(0), t(1), t(2),
                 q(0), q(1), q(2), q(3),
                 static_cast<unsigned long long>(img.features.size()),
                 static_cast<unsigned long long>(img.depth_priors.size()),
                 img.is_registered ? 1 : 0);
  }
  std::vector<track_t> trk_ids;
  trk_ids.reserve(tracks.size());
  for (const auto& kv : tracks) trk_ids.push_back(kv.first);
  std::sort(trk_ids.begin(), trk_ids.end());
  for (track_t id : trk_ids) {
    const auto& trk = tracks.at(id);
    std::fprintf(f, "%s|track|%llu|xyz=%.17g,%.17g,%.17g|init=%d|nobs=%llu|nlc=%llu\n",
                 tag, static_cast<unsigned long long>(id),
                 trk.xyz(0), trk.xyz(1), trk.xyz(2),
                 trk.is_initialized ? 1 : 0,
                 static_cast<unsigned long long>(trk.observations.size()),
                 static_cast<unsigned long long>(trk.lc_observations.size()));
  }
  std::fclose(f);
}

}  // namespace

std::shared_ptr<ceres::LossFunction>
GlobalPositionerOptions::CreateLossFromConfig(
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
  // Default seed = 1; can override via GP_SEED env var for distribution probes.
  unsigned int seed = 1;
  if (const char* s = std::getenv("GP_SEED")) {
    seed = static_cast<unsigned int>(std::atoi(s));
  }
  random_generator_.seed(seed);
}

bool GlobalPositioner::Solve(
    ViewGraph& view_graph,
    std::unordered_map<camera_t, Camera>& cameras,
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<track_t, Track>& tracks,
    const std::unordered_set<image_t>& image_ids_to_optimize,
    const std::map<image_t, double>& initial_dmap_scales,
    const std::vector<image_pair_t>& consecutive_pair_ids) {
  if (images.empty()) {
    LOG(ERROR) << "Number of images = " << images.size();
    return false;
  }

  // Store the image IDs to optimize
  image_ids_to_optimize_ = image_ids_to_optimize;

  GpDebugDumpInputs("solve_enter", images, tracks);

  // Setup the problem.
  SetupProblem(view_graph, tracks);
  GpDebugDumpInputs("post_setup", images, tracks);

  // Initialize depth map scales from provided values if available
  if (!initial_dmap_scales.empty()) {
    for (const auto& [image_id, scale] : initial_dmap_scales) {
      if (scale <= 0.0) {
        continue;
      }
      // Convert to log-space if needed, otherwise use linear value
      double initial_value =
          options_.use_log_scale_for_depth_map_scales ? std::log(scale) : scale;
      dmap_scales_[image_id] = initial_value;
      dmap_scale_observation_counts_[image_id] = 0;
    }
  }

  // Initialize camera translations to be random.
  // Also, convert the camera pose translation to be the camera center.
  InitializeRandomPositions(view_graph, images, tracks);
  GpDebugDumpInputs("post_init_random_positions", images, tracks);

  if (options_.use_init && initial_dmap_scales.empty()) {
    InitializeDepthMapScalesFromObservations(images, tracks);
  }

  if (!options_.debug_only_relative_pose) {
    AddPointToCameraConstraints(view_graph, cameras, images, tracks);
  }
  GpDebugDumpInputs("post_add_point_to_camera", images, tracks);

  AddCamerasAndPointsToParameterGroups(images, tracks);

  ParameterizeVariables(images, tracks);
  GpDebugDumpInputs("pre_ceres_solve", images, tracks);

  ceres::Solver::Summary summary;
  ceres::Solve(options_.solver_options, problem_.get(), &summary);
  if (const char* path = GpDebugDumpPath()) {
    FILE* f = std::fopen(path, "a");
    if (f) {
      std::fprintf(f,
                   "ceres_summary|init=%.17g|final=%.17g|iters=%d|term=%d|"
                   "linear_solver=%d|preconditioner=%d|threading=%d|"
                   "n_resid_blocks=%d|n_param_blocks=%d|n_params=%d\n",
                   summary.initial_cost,
                   summary.final_cost,
                   static_cast<int>(summary.iterations.size()),
                   static_cast<int>(summary.termination_type),
                   static_cast<int>(options_.solver_options.linear_solver_type),
                   static_cast<int>(options_.solver_options.preconditioner_type),
                   options_.solver_options.num_threads,
                   summary.num_residual_blocks,
                   summary.num_parameter_blocks,
                   summary.num_parameters);
      // Per-iteration cost trace
      for (size_t i = 0; i < summary.iterations.size() && i < 50; ++i) {
        std::fprintf(f, "ceres_iter|%zu|cost=%.17g|step_size=%.17g|tr=%.17g\n",
                     i,
                     summary.iterations[i].cost,
                     summary.iterations[i].step_norm,
                     summary.iterations[i].trust_region_radius);
      }
      std::fclose(f);
    }
  }
  GpDebugDumpInputs("post_ceres_solve", images, tracks);

  ConvertResults(images);
  GpDebugDumpInputs("post_convert_results", images, tracks);
  return summary.IsSolutionUsable();
}

void GlobalPositioner::SetupProblem(
    ViewGraph& view_graph,
    const std::unordered_map<track_t, Track>& tracks) {
  ceres::Problem::Options problem_options;
  problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  problem_ = std::make_unique<ceres::Problem>(problem_options);

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
    ViewGraph& view_graph,
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<track_t, Track>& tracks) {
  std::unordered_set<image_t> constrained_positions;
  constrained_positions.reserve(images.size());
  for (const auto& [pair_id, image_pair] : view_graph.MutableImagePairs()) {
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

  for (auto& [image_id, image] : images) {
    if (!image_ids_to_optimize_.empty() &&
        image_ids_to_optimize_.find(image_id) == image_ids_to_optimize_.end()) {
      continue;
    }

    if (constrained_positions.find(image_id) == constrained_positions.end()) {
      image.cam_from_world.translation() = ImageCenter(image);
      continue;
    }

    if (!options_.use_init) {
      image.cam_from_world.translation() =
          options_.random_init_scale * RandVector3d(random_generator_, -1, 1);
    }
  }
}

void GlobalPositioner::AddPointToCameraConstraints(
    ViewGraph& view_graph,
    std::unordered_map<camera_t, Camera>& cameras,
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<track_t, Track>& tracks) {
  // Find the tracks that are relevant to the current set of cameras
  const size_t num_pt_to_cam = tracks.size();

  if (num_pt_to_cam == 0) return;

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

  // Create loss functions once at the start to keep them alive for all
  // residuals This ensures they don't get destroyed while Ceres is using them
  // We'll use the cached member variables directly in the loop, not call
  // getters
  cached_loss_normal_geometry_ = GlobalPositionerOptions::CreateLossFromConfig(
      options_.loss_normal_geometry);
  cached_loss_normal_depth_ =
      GlobalPositionerOptions::CreateLossFromConfig(options_.loss_normal_depth);
  cached_loss_lc_geometry_ =
      GlobalPositionerOptions::CreateLossFromConfig(options_.loss_lc_geometry);
  cached_loss_normal_geometry_inlier_ =
      GlobalPositionerOptions::CreateLossFromConfig(
          options_.loss_normal_geometry_inlier);
  cached_loss_normal_depth_inlier_ =
      GlobalPositionerOptions::CreateLossFromConfig(
          options_.loss_normal_depth_inlier);
  cached_loss_lc_depth_ =
      GlobalPositionerOptions::CreateLossFromConfig(options_.loss_lc_depth);
  cached_loss_scale_prior_ =
      GlobalPositionerOptions::CreateLossFromConfig(options_.loss_scale_prior);
  cached_loss_normal_geometry_trackstart_ =
      GlobalPositionerOptions::CreateLossFromConfig(
          options_.loss_normal_geometry_trackstart);
  cached_loss_normal_depth_trackstart_ =
      GlobalPositionerOptions::CreateLossFromConfig(
          options_.loss_normal_depth_trackstart);
  cached_loss_normal_depth_outlier_ =
      GlobalPositionerOptions::CreateLossFromConfig(
          options_.loss_normal_depth_outlier);

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
      continue;
    }

    // Initialize track xyz: use existing value if available, otherwise random
    if (options_.use_init) {
      track.is_initialized = true;
    } else {
      track.xyz =
          options_.random_init_scale * RandVector3d(random_generator_, -1, 1);
      track.is_initialized = true;
    }

    AddTrackToProblem(track_id, view_graph, cameras, images, tracks);
  }

  // Now add scale prior residuals for all images with depth observations
  // Weight each prior by the number of observations for that image
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
        } else {
          LOG(ERROR) << "Failed to create scale prior error for image "
                     << img_id;
        }
      }
    }
  }
}

void GlobalPositioner::AddTrackToProblem(
    track_t track_id,
    ViewGraph& view_graph,
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
    ViewGraph& view_graph,
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

  const Eigen::Vector3d v_ik = image.cam_from_world.rotation().inverse() *
                               image.features_undist[feature_id];

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
    if (feature_id < image.angular_stddevs.size()) {
      const Eigen::Vector2d& angular_std = image.angular_stddevs[feature_id];
      const double sigma_x = std::max(1e-9, angular_std[0]);
      const double sigma_y = std::max(1e-9, angular_std[1]);
      const double sigma_z = 0.5 * (sigma_x + sigma_y);
      cost_function = WeightedBATADirectionalError::Create(
          v_ik, image.cam_from_world.rotation(), sigma_x, sigma_y, sigma_z);
    } else {
      cost_function = BATAPairwiseDirectionError::Create(v_ik);
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
    // Add residual block (rotation baked into cost function).
    problem_->AddResidualBlock(cost_function,
                               geometry_loss,
                               image.cam_from_world.translation().data(),
                               tracks[track_id].xyz.data(),
                               &d_ik);
    problem_->SetParameterLowerBound(&d_ik, 0, 1e-5);
  } else {
    switch (options_.point_constraint_type) {
      case GlobalPositionerOptions::PointConstraintType::SPLIT_METRIC_DEPTH: {
        ceres::CostFunction* cost_dir = nullptr;
        if (feature_id < image.angular_stddevs.size()) {
          const Eigen::Vector2d& angular_std =
              image.angular_stddevs[feature_id];
          const double sigma_x = std::max(1e-9, angular_std[0]);
          const double sigma_y = std::max(1e-9, angular_std[1]);
          const double sigma_z = 0.5 * (sigma_x + sigma_y);
          cost_dir = WeightedBATADirectionalError::Create(
              v_ik, image.cam_from_world.rotation(), sigma_x, sigma_y, sigma_z);
        }
        if (cost_dir == nullptr) {
          cost_dir = BATAPairwiseDirectionError::Create(v_ik);
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
          // Add residual block (rotation baked into cost function).
          problem_->AddResidualBlock(cost_dir,
                                     geometry_loss,
                                     image.cam_from_world.translation().data(),
                                     tracks[track_id].xyz.data(),
                                     &d_ik);
          problem_->SetParameterLowerBound(&d_ik, 0, 1e-5);
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

        // Create depth cost function (rotation baked into cost function).
        ceres::CostFunction* cost_depth = nullptr;
        {
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
            depth_loss = GetLossOutlierDepth().get();
          } else {
            if (is_lc_observation) {
              depth_loss = cached_loss_lc_depth_.get();
            } else if (is_track_anchor_obs_depth) {
              depth_loss = cached_loss_normal_depth_trackstart_.get();
            } else if (use_inlier_loss_depth) {
              depth_loss = cached_loss_normal_depth_inlier_.get();
            } else if (use_mdrp_depth_outlier_loss) {
              depth_loss = cached_loss_normal_depth_outlier_.get();
            } else {
              depth_loss = cached_loss_normal_depth_.get();
            }
          }
          // Add residual block (rotation baked into cost function).
          problem_->AddResidualBlock(cost_depth,
                                     depth_loss,
                                     image.cam_from_world.translation().data(),
                                     tracks[track_id].xyz.data(),
                                     &dmap_scale);
          // Add a lower bound only for linear space (log-space automatically
          // enforces positivity)
          if (!options_.use_log_scale_for_depth_map_scales) {
            problem_->SetParameterLowerBound(&dmap_scale, 0, 1e-5);
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
  options_.solver_options.linear_solver_ordering.reset(
      new ceres::ParameterBlockOrdering);
  ceres::ParameterBlockOrdering* parameter_ordering =
      options_.solver_options.linear_solver_ordering.get();

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

  // If do not optimize the depth map scales, set them to be constant
  if (!options_.optimize_depth_map_scales) {
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
  //             anchor_image.cam_from_world.translation().data())) {
  //       LOG(INFO) << "Fixing camera position for image " << anchor_image_id
  //                 << " to remove translation gauge freedom";
  //       problem_->SetParameterBlockConstant(
  //           anchor_image.cam_from_world.translation().data());
  //     }
  //   }
  // }

  // Set up the options for the solver
  // Do not use iterative solvers, for its suboptimal performance.
  if (tracks.size() > 0) {
    options_.solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
    options_.solver_options.preconditioner_type = ceres::CLUSTER_TRIDIAGONAL;
  } else {
    options_.solver_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options_.solver_options.preconditioner_type = ceres::JACOBI;
  }
}

void GlobalPositioner::ConvertResults(
    std::unordered_map<image_t, Image>& images) {
  // translation now stores the camera position, needs to convert back to
  // translation
  for (auto& [image_id, image] : images) {
    image.cam_from_world.translation() =
        -(image.cam_from_world.rotation() * image.cam_from_world.translation());
  }
}

// Helper methods to get loss functions from current options
// These ensure that changes to options_ take effect even after SetupProblem()
std::shared_ptr<ceres::LossFunction> GlobalPositioner::GetLossNormalGeometry()
    const {
  // Always recreate from current options to pick up changes
  // Cache as member variable to keep it alive for Ceres (DO_NOT_TAKE_OWNERSHIP)
  cached_loss_normal_geometry_ = GlobalPositionerOptions::CreateLossFromConfig(
      options_.loss_normal_geometry);
  return cached_loss_normal_geometry_;
}

std::shared_ptr<ceres::LossFunction> GlobalPositioner::GetLossNormalDepth()
    const {
  // Always recreate from current options to pick up changes
  cached_loss_normal_depth_ =
      GlobalPositionerOptions::CreateLossFromConfig(options_.loss_normal_depth);
  return cached_loss_normal_depth_;
}

std::shared_ptr<ceres::LossFunction> GlobalPositioner::GetLossLCGeometry()
    const {
  // Always recreate from current options to pick up changes
  cached_loss_lc_geometry_ =
      GlobalPositionerOptions::CreateLossFromConfig(options_.loss_lc_geometry);
  return cached_loss_lc_geometry_;
}

std::shared_ptr<ceres::LossFunction> GlobalPositioner::GetLossLCDepth() const {
  // Always recreate from current options to pick up changes
  cached_loss_lc_depth_ =
      GlobalPositionerOptions::CreateLossFromConfig(options_.loss_lc_depth);
  return cached_loss_lc_depth_;
}

std::shared_ptr<ceres::LossFunction> GlobalPositioner::GetLossScalePrior()
    const {
  // Always recreate from current options to pick up changes
  cached_loss_scale_prior_ =
      GlobalPositionerOptions::CreateLossFromConfig(options_.loss_scale_prior);
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
  return cached_loss_outlier_depth_;
}

std::shared_ptr<ceres::LossFunction>
GlobalPositioner::GetLossNormalGeometryInlier() const {
  // Always recreate from current options to pick up changes
  cached_loss_normal_geometry_inlier_ =
      GlobalPositionerOptions::CreateLossFromConfig(
          options_.loss_normal_geometry_inlier);
  return cached_loss_normal_geometry_inlier_;
}

std::shared_ptr<ceres::LossFunction>
GlobalPositioner::GetLossNormalDepthInlier() const {
  // Always recreate from current options to pick up changes
  cached_loss_normal_depth_inlier_ =
      GlobalPositionerOptions::CreateLossFromConfig(
          options_.loss_normal_depth_inlier);
  return cached_loss_normal_depth_inlier_;
}

std::shared_ptr<ceres::LossFunction>
GlobalPositioner::GetLossNormalDepthOutlier() const {
  // Always recreate from current options to pick up changes
  cached_loss_normal_depth_outlier_ =
      GlobalPositionerOptions::CreateLossFromConfig(
          options_.loss_normal_depth_outlier);
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

  for (const auto& [image_id, scale_estimates] : image_scale_estimates) {
    if (scale_estimates.empty()) continue;

    std::vector<double> sorted_estimates = scale_estimates;
    std::sort(sorted_estimates.begin(), sorted_estimates.end());
    double median_scale = sorted_estimates[sorted_estimates.size() / 2];

    double initial_value = options_.use_log_scale_for_depth_map_scales
                               ? std::log(median_scale)
                               : median_scale;

    dmap_scales_[image_id] = initial_value;
    dmap_scale_observation_counts_[image_id] = 0;
  }
}

void GlobalPositioner::FilterDepthOutliers(
    const std::unordered_map<image_t, Image>& images,
    const std::unordered_map<track_t, Track>& tracks) {
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

      if (z_est <= 1e-6) continue;

      const double metric_std = stddev_rel * depth_prior;
      const double log_z_est = std::log(std::max(z_est, 1e-6));
      const double log_depth_prior = std::log(std::max(depth_prior, 1e-6));
      const double log_diff = std::abs(log_z_est - log_depth_prior);
      const double metric_std_over_depth =
          metric_std / std::max(depth_prior, 1e-6);
      const double threshold =
          3.0 * std::log(1.0 + std::max(metric_std_over_depth, 1e-6));

      if (log_diff >= threshold) {
        depth_outliers_.insert(std::make_pair(image_id, feature_id));
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

      if (z_est <= 1e-6) continue;

      const double metric_std = stddev_rel * depth_prior;
      const double log_z_est = std::log(std::max(z_est, 1e-6));
      const double log_depth_prior = std::log(std::max(depth_prior, 1e-6));
      const double log_diff = std::abs(log_z_est - log_depth_prior);
      const double metric_std_over_depth =
          metric_std / std::max(depth_prior, 1e-6);
      const double threshold =
          3.0 * std::log(1.0 + std::max(metric_std_over_depth, 1e-6));

      if (log_diff >= threshold) {
        depth_outliers_.insert(std::make_pair(image_id, feature_id));
      }
    }
  }
}

}  // namespace glomap_ra
}  // namespace colmap
