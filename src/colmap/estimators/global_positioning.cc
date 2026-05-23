#include "colmap/estimators/global_positioning.h"

#include "colmap/estimators/cost_functions/metric_depth.h"
#include "colmap/estimators/cost_functions/motion_averaging.h"
#include "colmap/estimators/cost_functions/utils.h"
#include "colmap/math/random.h"
#include "colmap/util/cuda.h"
#include "colmap/util/misc.h"
#include "colmap/util/threading.h"

#include <algorithm>
#include <cstdlib>
#include <unordered_map>
#include <vector>

namespace colmap {
namespace {

std::string GpObservationKey(point3D_t point3D_id,
                             image_t image_id,
                             point2D_t point2D_idx,
                             bool is_lc_observation) {
  return std::to_string(point3D_id) + ":" + std::to_string(image_id) + ":" +
         std::to_string(point2D_idx) + ":" + (is_lc_observation ? "1" : "0");
}

Eigen::Vector3d RandVector3d(double low, double high) {
  return Eigen::Vector3d(RandomUniformReal(low, high),
                         RandomUniformReal(low, high),
                         RandomUniformReal(low, high));
}

MetricDepthOptions CreateMetricDepthOptions(
    const GlobalPositionerOptions& options) {
  MetricDepthOptions metric_depth_options;
  metric_depth_options.use_log_scale =
      options.use_log_scale_for_depth_map_scales;
  metric_depth_options.zero_residual_behind = options.zero_residual_behind;
  metric_depth_options.log_linear_threshold = options.log_linear_threshold;

  if (options.smooth_log_linear_transition) {
    metric_depth_options.residual_type = MetricDepthResidualType::kLogLinear;
  } else if (options.use_log_residual_for_depth) {
    metric_depth_options.residual_type = MetricDepthResidualType::kLog;
  } else {
    metric_depth_options.residual_type = MetricDepthResidualType::kLinear;
  }
  return metric_depth_options;
}

// Per-observation depth outlier check. Returns true when the log-space
// residual |log(z_est / scaled_prior)| exceeds sigma * sigma_log.
inline bool DepthOutlierFlag(const Image& image,
                             point2D_t feature_id,
                             const Eigen::Vector3d& point3D_xyz,
                             bool use_log_scale,
                             const std::map<image_t, double>& dmap_scales,
                             image_t image_id,
                             double sigma) {
  if (feature_id >= image.depth_prior_validity.size() ||
      !image.depth_prior_validity[feature_id]) {
    return false;
  }
  if (feature_id >= image.depth_priors.size() ||
      feature_id >= image.depth_prior_stddevs.size()) {
    return false;
  }
  const double depth_prior_raw = image.depth_priors[feature_id];
  const double stddev_rel = image.depth_prior_stddevs[feature_id];
  if (depth_prior_raw <= 1e-6 || stddev_rel <= 1e-9) return false;

  // Apply per-image dmap_scale if available (else use raw prior).
  double depth_prior = depth_prior_raw;
  auto scale_it = dmap_scales.find(image_id);
  if (scale_it != dmap_scales.end()) {
    const double dmap_scale =
        use_log_scale ? std::exp(scale_it->second) : scale_it->second;
    depth_prior = dmap_scale * depth_prior_raw;
  }

  // z_est = (cam_from_world * X_world)[2]
  const Eigen::Vector3d point_cam = image.CamFromWorld() * point3D_xyz;
  const double z_est = point_cam[2];
  if (z_est <= 1e-6) return false;

  const double log_z_est = std::log(std::max(z_est, 1e-6));
  const double log_depth_prior = std::log(std::max(depth_prior, 1e-6));
  const double log_diff = std::abs(log_z_est - log_depth_prior);
  const double threshold = sigma * std::log(1.0 + std::max(stddev_rel, 1e-6));
  return log_diff >= threshold;
}

size_t NumRegularObservationsForMinViewGate(const Track& track) {
  return track.Length();
}

bool IsLossConfigOverride(const LossConfig& loss_config) {
  return loss_config.type != LossFunctionType::TRIVIAL ||
         loss_config.scale != 1.0 || loss_config.weight != 1.0;
}

bool HasDebugInitialization(const GlobalPositionerOptions& options) {
  return !options.debug_initial_frame_centers.empty() ||
         !options.debug_initial_point3D_xyz.empty() ||
         !options.debug_initial_bata_scales.empty();
}

}  // namespace

GlobalPositioner::GlobalPositioner(const GlobalPositionerOptions& options)
    : options_(options) {
  if (options_.random_seed >= 0) {
    SetPRNGSeed(static_cast<unsigned>(options_.random_seed));
  }
}

bool GlobalPositioner::Solve(const PoseGraph& pose_graph,
                             Reconstruction& reconstruction) {
  ValidateDebugInitializationOptions();

  if (reconstruction.NumImages() == 0) {
    LOG(ERROR) << "Number of images = " << reconstruction.NumImages();
    return false;
  }
  if (reconstruction.NumPoints3D() == 0) {
    LOG(ERROR) << "Number of tracks = " << reconstruction.NumPoints3D();
    return false;
  }

  // TODO: extend rig branch in AddObservationToProblem to add MetricDepthError
  // for non-ref images. Until then, fail loud on multi-camera rigs +
  // use_metric_depth_constraint.
  if (options_.use_metric_depth_constraint) {
    for (const auto& [image_id, image] : reconstruction.Images()) {
      THROW_CHECK(image.IsRefInFrame())
          << "use_metric_depth_constraint=true is not yet supported with "
             "multi-camera rigs. Image "
          << image_id
          << " is a non-ref sensor in its frame; its depth "
             "residual would be silently dropped. Either disable "
             "use_metric_depth_constraint or run on single-camera rigs.";
    }
  }

  LOG(INFO) << "Setting up the global positioner problem";

  // Setup the problem.
  SetupProblem(pose_graph, reconstruction);

  // Initialize camera translations to be random.
  // Also, convert the camera pose translation to be the camera center.
  InitializeRandomPositions(pose_graph, reconstruction);

  // No caller-supplied seed for dmap_scales_; derive one from per-image
  // median observed z_est/depth_prior.
  if (options_.use_metric_depth_constraint && options_.use_init &&
      !options_.initial_dmap_scales.has_value()) {
    InitializeDepthMapScalesFromObservations(reconstruction);
  }

  // Add the point to camera constraints to the problem.
  AddPointToCameraConstraints(reconstruction);
  ValidateDebugInitializationConsumed();

  if (options_.use_parameter_block_ordering) {
    AddCamerasAndPointsToParameterGroups(reconstruction);
  }

  // Parameterize the variables, set image poses / tracks / scales to be
  // constant if desired
  ParameterizeVariables(reconstruction);

  LOG(INFO) << "Solving the global positioner problem";

  ceres::Solver::Summary summary;
  options_.solver_options.num_threads =
      GetEffectiveNumThreads(options_.solver_options.num_threads);
  options_.solver_options.minimizer_progress_to_stdout = VLOG_IS_ON(2);
  ceres::Solve(options_.solver_options, problem_.get(), &summary);
  diagnostics_.num_bata_scales = static_cast<int>(scales_.size());
  diagnostics_.num_dmap_scales = static_cast<int>(dmap_scales_.size());
  diagnostics_.num_frame_centers = static_cast<int>(frame_centers_.size());
  diagnostics_.num_point3D_xyz = static_cast<int>(initial_point3D_xyz_.size());
  diagnostics_.num_residual_blocks = summary.num_residual_blocks;
  diagnostics_.num_parameter_blocks = summary.num_parameter_blocks;
  diagnostics_.num_parameters = summary.num_parameters;
  diagnostics_.num_iterations = static_cast<int>(summary.iterations.size());
  diagnostics_.initial_cost = summary.initial_cost;
  diagnostics_.final_cost = summary.final_cost;
  diagnostics_.termination_type = static_cast<int>(summary.termination_type);

  if (VLOG_IS_ON(2)) {
    LOG(INFO) << summary.FullReport();
  } else {
    LOG(INFO) << summary.BriefReport();
  }

  ConvertBackResults(reconstruction);
  return summary.IsSolutionUsable();
}

void GlobalPositioner::SetupProblem(const PoseGraph& pose_graph,
                                    const Reconstruction& reconstruction) {
  ceres::Problem::Options problem_options;
  problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  problem_ = std::make_unique<ceres::Problem>(problem_options);
  loss_function_ = options_.CreateLossFunction();
  diagnostics_ = GlobalPositionerDiagnostics();

  // Clear temporary storage from previous runs.
  frame_centers_.clear();
  initial_frame_centers_.clear();
  initial_point3D_xyz_.clear();
  initial_bata_scales_.clear();
  bata_scale_indices_.clear();
  cams_in_rig_.clear();
  per_image_scale_losses_.clear();

  // Reserve scales_ for both regular observations and lc_elements.
  // Underestimating triggers ``vector::push_back`` reallocation mid-build,
  // which invalidates the ``&scale`` data pointers that earlier residual
  // blocks already stored.
  scales_.clear();
  size_t total_observations = 0;
  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    total_observations += point3D.track.Length();
    total_observations += point3D.track.lc_elements.size();
  }
  scales_.reserve(total_observations);
}

void GlobalPositioner::InitializeRandomPositions(
    const PoseGraph& pose_graph, Reconstruction& reconstruction) {
  std::unordered_set<frame_t> constrained_positions;
  constrained_positions.reserve(reconstruction.NumFrames());
  for (const auto& [pair_id, edge] : pose_graph.ValidEdges()) {
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    constrained_positions.insert(reconstruction.Image(image_id1).FrameId());
    constrained_positions.insert(reconstruction.Image(image_id2).FrameId());
  }

  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    if (NumRegularObservationsForMinViewGate(point3D.track) <
        static_cast<size_t>(options_.min_num_view_per_track)) {
      continue;
    }
    for (const auto& observation : point3D.track.Elements()) {
      THROW_CHECK(reconstruction.ExistsImage(observation.image_id));
      const Image& image = reconstruction.Image(observation.image_id);
      if (!image.HasPose()) continue;
      constrained_positions.insert(image.FrameId());
    }
    if (options_.use_lc_observations) {
      for (const auto& observation : point3D.track.lc_elements) {
        if (!reconstruction.ExistsImage(observation.image_id)) continue;
        const Image& image = reconstruction.Image(observation.image_id);
        if (!image.HasPose()) continue;
        constrained_positions.insert(image.FrameId());
      }
    }
  }

  // Initialize frame centers in temporary storage.
  // The reconstruction poses remain in cam_from_world convention.
  for (const auto& [frame_id, frame] : reconstruction.Frames()) {
    if (constrained_positions.find(frame_id) == constrained_positions.end()) {
      continue;
    }
    if (options_.generate_random_positions && options_.optimize_positions &&
        !options_.use_init) {
      frame_centers_[frame_id] =
          options_.random_init_scale * RandVector3d(-1, 1);
    } else {
      frame_centers_[frame_id] = frame.RigFromWorld().TgtOriginInSrc();
    }
    const auto debug_it = options_.debug_initial_frame_centers.find(frame_id);
    if (debug_it != options_.debug_initial_frame_centers.end()) {
      frame_centers_[frame_id] = debug_it->second;
    }
    initial_frame_centers_[frame_id] = frame_centers_[frame_id];
  }

  VLOG(2) << "Constrained positions: " << constrained_positions.size();
}

void GlobalPositioner::AddPointToCameraConstraints(
    Reconstruction& reconstruction) {
  VLOG(2) << reconstruction.NumPoints3D()
          << " point to camera constraints were added to the position "
             "estimation problem.";

  // Down-weight uncalibrated cameras.
  if (options_.apply_uncalibrated_loss_downweight) {
    loss_function_ptcam_uncalibrated_ = std::make_shared<ceres::ScaledLoss>(
        loss_function_.get(),
        options_.uncalibrated_loss_downweight,
        ceres::DO_NOT_TAKE_OWNERSHIP);
  } else {
    loss_function_ptcam_uncalibrated_ = loss_function_;
  }
  loss_function_ptcam_calibrated_ = loss_function_;

  // Initialize cascade losses.
  cached_loss_normal_geometry_ =
      options_.loss_normal_geometry.CreateLossFunction();
  cached_loss_normal_depth_ = options_.loss_normal_depth.CreateLossFunction();
  cached_loss_lc_geometry_ =
      IsLossConfigOverride(options_.loss_lc_geometry)
          ? options_.loss_lc_geometry.CreateLossFunction()
          : nullptr;
  cached_loss_lc_depth_ = options_.loss_lc_depth.CreateLossFunction();
  cached_loss_normal_geometry_inlier_ =
      options_.loss_normal_geometry_inlier.CreateLossFunction();
  cached_loss_normal_depth_inlier_ =
      options_.loss_normal_depth_inlier.CreateLossFunction();
  cached_loss_normal_depth_outlier_ =
      options_.loss_normal_depth_outlier.CreateLossFunction();
  cached_loss_normal_geometry_trackstart_ =
      options_.loss_normal_geometry_trackstart.CreateLossFunction();
  cached_loss_normal_depth_trackstart_ =
      options_.loss_normal_depth_trackstart.CreateLossFunction();
  cached_loss_scale_prior_ = options_.loss_scale_prior.CreateLossFunction();
  soft_outlier_fallback_loss_.reset();

  dmap_scales_.clear();
  dmap_scale_observation_counts_.clear();

  // Seed dmap_scales_ from initial values before outlier filtering.
  if (options_.use_metric_depth_constraint &&
      options_.initial_dmap_scales.has_value()) {
    for (const auto& [image_id, linear_scale] : *options_.initial_dmap_scales) {
      const double init_value = options_.use_log_scale_for_depth_map_scales
                                    ? std::log(std::max(linear_scale, 1e-9))
                                    : linear_scale;
      dmap_scales_[image_id] = init_value;
      dmap_scale_observation_counts_[image_id] = 0;
    }
  }

  depth_outliers_.clear();
  if (options_.use_metric_depth_constraint && options_.filter_depth_outliers) {
    FilterDepthOutliers(reconstruction);
  }

  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    if (NumRegularObservationsForMinViewGate(point3D.track) <
        static_cast<size_t>(options_.min_num_view_per_track)) {
      continue;
    }

    AddPoint3DToProblem(point3D_id, reconstruction);
  }

  // Emit one scale-prior residual per image with depth observations,
  // weighted by obs_count so dense-depth images get stronger priors.
  if (options_.use_metric_depth_constraint) {
    for (auto& [image_id, scale] : dmap_scales_) {
      auto count_it = dmap_scale_observation_counts_.find(image_id);
      const double obs_count =
          (count_it != dmap_scale_observation_counts_.end())
              ? static_cast<double>(count_it->second)
              : 1.0;

      // Prior: pulls log_s toward 0 (log-space) or s toward 1 (linear).
      const Eigen::Matrix<double, 1, 1> prior_vec(
          options_.use_log_scale_for_depth_map_scales ? 0.0 : 1.0);
      const Eigen::Matrix<double, 1, 1> cov_1x1(options_.scale_prior_stddev *
                                                options_.scale_prior_stddev);
      ceres::CostFunction* scale_prior_cost =
          CovarianceWeightedCostFunctor<NormalPriorCostFunctor<1>>::Create(
              cov_1x1, prior_vec);
      if (scale_prior_cost == nullptr) continue;

      ceres::LossFunction* obs_count_scaled_loss = nullptr;
      if (cached_loss_scale_prior_) {
        per_image_scale_losses_.push_back(
            std::make_unique<ceres::ScaledLoss>(cached_loss_scale_prior_.get(),
                                                obs_count,
                                                ceres::DO_NOT_TAKE_OWNERSHIP));
      } else {
        per_image_scale_losses_.push_back(std::make_unique<ceres::ScaledLoss>(
            new ceres::TrivialLoss(), obs_count, ceres::TAKE_OWNERSHIP));
      }
      obs_count_scaled_loss = per_image_scale_losses_.back().get();

      problem_->AddResidualBlock(
          scale_prior_cost, obs_count_scaled_loss, &scale);
      ++diagnostics_.num_scale_prior_residuals;
    }
  }
  VLOG(2) << "GP: residual blocks=" << problem_->NumResidualBlocks()
          << ", parameter blocks=" << problem_->NumParameterBlocks()
          << ", scales=" << scales_.size()
          << ", frame_centers=" << frame_centers_.size()
          << ", dmap_scales=" << dmap_scales_.size();
}

void GlobalPositioner::AddPoint3DToProblem(point3D_t point3D_id,
                                           Reconstruction& reconstruction) {
  const bool random_initialization = options_.optimize_points &&
                                     options_.generate_random_points &&
                                     !options_.use_init;

  Point3D& point3D = reconstruction.Point3D(point3D_id);

  // Only set the points to be random if they are needed to be optimized
  if (random_initialization) {
    point3D.xyz = options_.random_init_scale * RandVector3d(-1, 1);
  }
  const auto debug_it = options_.debug_initial_point3D_xyz.find(point3D_id);
  if (debug_it != options_.debug_initial_point3D_xyz.end()) {
    point3D.xyz = debug_it->second;
  }
  initial_point3D_xyz_[point3D_id] = point3D.xyz;

  // Walk regular elements then LC elements as separate passes — they
  // share the residual layout but use different loss function groups.
  for (const auto& observation : point3D.track.Elements()) {
    AddObservationToProblem(
        point3D_id, observation, random_initialization, reconstruction);
  }
  if (options_.use_lc_observations) {
    for (const auto& observation : point3D.track.lc_elements) {
      AddObservationToProblem(point3D_id,
                              observation,
                              random_initialization,
                              reconstruction,
                              /*is_lc_observation=*/true);
    }
  }
}

void GlobalPositioner::AddObservationToProblem(point3D_t point3D_id,
                                               const TrackElement& observation,
                                               bool random_initialization,
                                               Reconstruction& reconstruction,
                                               bool is_lc_observation) {
  Point3D& point3D = reconstruction.Point3D(point3D_id);
  if (!reconstruction.ExistsImage(observation.image_id)) return;

  Image& image = reconstruction.Image(observation.image_id);
  if (!image.HasPose()) return;

  const std::optional<Eigen::Vector2d> cam_point =
      image.CameraPtr()->CamFromImg(image.Point2D(observation.point2D_idx).xy);
  if (!cam_point.has_value()) {
    LOG(WARNING) << "Ignoring feature because it failed to project: point3D_id="
                 << point3D_id << ", image_id=" << observation.image_id
                 << ", feature_id=" << observation.point2D_idx;
    return;
  }

  const Eigen::Vector3d cam_from_point3D_dir =
      image.CamFromWorld().rotation().inverse() *
      cam_point->homogeneous().normalized();

  CHECK_GE(scales_.capacity(), scales_.size())
      << "Not enough capacity was reserved for the scales.";
  double& scale = scales_.emplace_back(1);

  if (!options_.generate_scales &&
      (random_initialization || options_.initialize_warm_start_scales)) {
    const Eigen::Vector3d cam_from_point3D_translation =
        point3D.xyz - frame_centers_[image.FrameId()];
    scale = std::max(1e-5,
                     cam_from_point3D_dir.dot(cam_from_point3D_translation) /
                         cam_from_point3D_translation.squaredNorm());
  }
  const std::string scale_key = GpObservationKey(point3D_id,
                                                 observation.image_id,
                                                 observation.point2D_idx,
                                                 is_lc_observation);
  const auto debug_scale_it =
      options_.debug_initial_bata_scales.find(scale_key);
  if (debug_scale_it != options_.debug_initial_bata_scales.end()) {
    scale = debug_scale_it->second;
  }
  initial_bata_scales_[scale_key] = scale;
  bata_scale_indices_[scale_key] = scales_.size() - 1;

  // For calibrated and uncalibrated cameras, use different loss
  // functions
  // Down weight the uncalibrated cameras
  Camera& camera = reconstruction.Camera(image.CameraId());
  ceres::LossFunction* loss_function =
      (camera.has_prior_focal_length) ? loss_function_ptcam_calibrated_.get()
                                      : loss_function_ptcam_uncalibrated_.get();

  // Geometry-loss cascade. Per-observation route:
  //   is_lc                         -> cached_loss_lc_geometry_
  //   TrackElement::is_track_anchor -> cached_loss_normal_geometry_trackstart_
  //   TrackElement::is_inlier       -> cached_loss_normal_geometry_inlier_
  //   else                          -> cached_loss_normal_geometry_
  if (is_lc_observation && cached_loss_lc_geometry_) {
    loss_function = cached_loss_lc_geometry_.get();
  } else if (options_.use_metric_depth_constraint) {
    ceres::LossFunction* cascade = nullptr;
    if (observation.is_track_anchor) {
      cascade = cached_loss_normal_geometry_trackstart_.get();
    } else if (observation.is_inlier) {
      cascade = cached_loss_normal_geometry_inlier_.get();
    } else {
      cascade = cached_loss_normal_geometry_.get();
    }
    if (cascade != nullptr) {
      loss_function = cascade;
    }
  }

  // If the image is not part of a camera rig, use the standard BATA error
  if (image.IsRefInFrame()) {
    // Anisotropic per-keypoint covariance via CovarianceWeightedCostFunctor
    // when angular_stddevs is populated; otherwise bare BATA functor.
    // cov_world = R^T diag(sigma^2) R.
    ceres::CostFunction* cost_function = nullptr;
    if (observation.point2D_idx < image.angular_stddevs.size()) {
      const Eigen::Vector2d& angular_std =
          image.angular_stddevs[observation.point2D_idx];
      const double sigma_x = std::max(1e-9, angular_std[0]);
      const double sigma_y = std::max(1e-9, angular_std[1]);
      const double sigma_z = 0.5 * (sigma_x + sigma_y);
      const Eigen::Matrix3d R =
          image.CamFromWorld().rotation().toRotationMatrix();
      const Eigen::Matrix3d cov_world =
          R.transpose() *
          Eigen::Vector3d(
              sigma_x * sigma_x, sigma_y * sigma_y, sigma_z * sigma_z)
              .asDiagonal() *
          R;
      cost_function = CovarianceWeightedCostFunctor<
          BATAPairwiseDirectionCostFunctor>::Create(cov_world,
                                                    cam_from_point3D_dir);
    }
    if (cost_function == nullptr) {
      cost_function =
          BATAPairwiseDirectionCostFunctor::Create(cam_from_point3D_dir);
    }

    problem_->AddResidualBlock(cost_function,
                               loss_function,
                               frame_centers_[image.FrameId()].data(),
                               point3D.xyz.data(),
                               &scale);
    ++diagnostics_.num_bata_residuals;
    if (is_lc_observation) {
      ++diagnostics_.num_lc_observations_used;
    } else {
      ++diagnostics_.num_regular_observations_used;
    }

    // 1-D MetricDepthError: anchors absolute scale via depth prior.
    if (options_.use_metric_depth_constraint) {
      AddMetricDepthResidual(
          point3D_id, observation, is_lc_observation, reconstruction);
    }
  } else {
    // If the image is part of a camera rig, use the RigBATA error.

    const rig_t rig_id = image.FramePtr()->RigId();
    Rig& rig = reconstruction.Rig(rig_id);
    Rigid3d& cam_from_rig = rig.SensorFromRig(image.CameraPtr()->SensorId());

    if (!cam_from_rig.translation().hasNaN()) {
      const Eigen::Vector3d cam_from_rig_dir =
          image.CamFromWorld().rotation().inverse() *
          cam_from_rig.translation();

      ceres::CostFunction* cost_function =
          RigBATAPairwiseDirectionConstantRigCostFunctor::Create(
              cam_from_point3D_dir, cam_from_rig_dir);

      problem_->AddResidualBlock(cost_function,
                                 loss_function,
                                 point3D.xyz.data(),
                                 frame_centers_[image.FrameId()].data(),
                                 &scale);
      ++diagnostics_.num_bata_residuals;
      if (is_lc_observation) {
        ++diagnostics_.num_lc_observations_used;
      } else {
        ++diagnostics_.num_regular_observations_used;
      }
    } else {
      // If the cam_from_rig contains nan values, it needs to be re-estimated.
      // Initialize cams_in_rig_ if not already done.
      const sensor_t sensor_id = image.CameraPtr()->SensorId();
      if (cams_in_rig_.find(sensor_id) == cams_in_rig_.end()) {
        // Will be initialized to random values in ParameterizeVariables().
        cams_in_rig_[sensor_id] = Eigen::Vector3d::Zero();
      }

      ceres::CostFunction* cost_function =
          RigBATAPairwiseDirectionCostFunctor::Create(
              cam_from_point3D_dir,
              image.FramePtr()->RigFromWorld().rotation());

      problem_->AddResidualBlock(cost_function,
                                 loss_function,
                                 point3D.xyz.data(),
                                 frame_centers_[image.FrameId()].data(),
                                 cams_in_rig_[sensor_id].data(),
                                 &scale);
      ++diagnostics_.num_bata_residuals;
      if (is_lc_observation) {
        ++diagnostics_.num_lc_observations_used;
      } else {
        ++diagnostics_.num_regular_observations_used;
      }
    }
  }

  problem_->SetParameterLowerBound(&scale, 0, 1e-5);
}

void GlobalPositioner::AddMetricDepthResidual(point3D_t point3D_id,
                                              const TrackElement& observation,
                                              bool is_lc_observation,
                                              Reconstruction& reconstruction) {
  if (!reconstruction.ExistsImage(observation.image_id)) return;
  const Image& image = reconstruction.Image(observation.image_id);

  if (observation.point2D_idx >= image.depth_prior_validity.size() ||
      !image.depth_prior_validity[observation.point2D_idx]) {
    return;
  }
  THROW_CHECK_LT(observation.point2D_idx, image.depth_priors.size());
  THROW_CHECK_LT(observation.point2D_idx, image.depth_prior_stddevs.size());

  const double depth_prior = image.depth_priors[observation.point2D_idx];
  const double depth_sigma = image.depth_prior_stddevs[observation.point2D_idx];

  if (depth_prior <= 0.0 || depth_sigma <= 1e-9) return;

  // Lazy-insert dmap_scales_ on first valid observation per image.
  if (dmap_scales_.find(observation.image_id) == dmap_scales_.end()) {
    double init_value = options_.use_log_scale_for_depth_map_scales ? 0.0 : 1.0;
    if (options_.initial_dmap_scales.has_value()) {
      const auto& init_map = *options_.initial_dmap_scales;
      auto it = init_map.find(observation.image_id);
      if (it != init_map.end()) {
        // Convert caller-supplied linear value to log if needed.
        init_value = options_.use_log_scale_for_depth_map_scales
                         ? std::log(std::max(it->second, 1e-9))
                         : it->second;
      }
    }
    dmap_scales_[observation.image_id] = init_value;
    dmap_scale_observation_counts_[observation.image_id] = 0;
  }
  dmap_scale_observation_counts_[observation.image_id]++;

  ceres::CostFunction* metric_depth_cost =
      MetricDepthError::Create(image.CamFromWorld().rotation(),
                               depth_prior,
                               depth_sigma,
                               CreateMetricDepthOptions(options_));

  if (metric_depth_cost == nullptr) return;

  // Outlier dual routing:
  //   depth_outliers_ = runtime geometry-driven filter
  //     (Nσ log-residual, N = filter_depth_outlier_sigma),
  //     populated by FilterDepthOutliers between BA iterations.
  //   TrackElement::is_depth_outlier = external annotation
  //     (MDRP/boundary heuristic), populated by Python pipeline
  //     pre-solve.
  //   depth_outliers_ is checked first and is strictly more
  //   aggressive: LC observations skip the depth residual
  //   entirely, non-LC get a soft fallback loss.
  //
  // 5-way depth-loss cascade:
  //   pre-pass outlier  -> soft fallback (skip on LC)
  //   is_lc             -> cached_loss_lc_depth_
  //   TrackElement::is_track_anchor  -> cached_loss_normal_depth_trackstart_
  //   TrackElement::is_inlier        -> cached_loss_normal_depth_inlier_
  //   TrackElement::is_depth_outlier -> cached_loss_normal_depth_outlier_
  //   else              -> cached_loss_normal_depth_
  ceres::LossFunction* depth_loss = nullptr;
  const std::pair<image_t, point2D_t> obs_key{observation.image_id,
                                              observation.point2D_idx};
  if (depth_outliers_.count(obs_key) > 0) {
    if (is_lc_observation) {
      // LC outlier: skip depth residual entirely.
      delete metric_depth_cost;
      return;
    }
    // Non-LC outlier: soft fallback (HuberLoss(1)).
    if (!soft_outlier_fallback_loss_) {
      soft_outlier_fallback_loss_ =
          options_.loss_soft_outlier_fallback.CreateLossFunction();
    }
    depth_loss = soft_outlier_fallback_loss_.get();
  } else if (is_lc_observation) {
    depth_loss = cached_loss_lc_depth_.get();
  } else if (observation.is_track_anchor) {
    depth_loss = cached_loss_normal_depth_trackstart_.get();
  } else if (observation.is_inlier) {
    depth_loss = cached_loss_normal_depth_inlier_.get();
  } else if (observation.is_depth_outlier) {
    depth_loss = cached_loss_normal_depth_outlier_.get();
  } else {
    depth_loss = cached_loss_normal_depth_.get();
  }

  Point3D& point3D = reconstruction.Point3D(point3D_id);
  problem_->AddResidualBlock(metric_depth_cost,
                             depth_loss,
                             frame_centers_[image.FrameId()].data(),
                             point3D.xyz.data(),
                             &dmap_scales_[observation.image_id]);
  ++diagnostics_.num_metric_depth_residuals;
}

void GlobalPositioner::AddCamerasAndPointsToParameterGroups(
    Reconstruction& reconstruction) {
  // Create a custom ordering for Schur-based problems.
  options_.solver_options.linear_solver_ordering.reset(
      new ceres::ParameterBlockOrdering);
  ceres::ParameterBlockOrdering* parameter_ordering =
      options_.solver_options.linear_solver_ordering.get();

  // Add scale parameters to group 0 (large and independent)
  for (double& scale : scales_) {
    parameter_ordering->AddElementToGroup(&scale, 0);
  }

  // Add point parameters to group 1.
  int group_id = 1;
  if (reconstruction.NumPoints3D() > 0) {
    for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
      if (problem_->HasParameterBlock(point3D.xyz.data()))
        parameter_ordering->AddElementToGroup(
            reconstruction.Point3D(point3D_id).xyz.data(), group_id);
    }
    group_id++;
  }

  for (auto& [frame_id, center] : frame_centers_) {
    if (problem_->HasParameterBlock(center.data())) {
      parameter_ordering->AddElementToGroup(center.data(), group_id);
    }
  }

  // Add the cam_in_rig to be estimated into the parameter group
  for (auto& [sensor_id, center] : cams_in_rig_) {
    if (problem_->HasParameterBlock(center.data())) {
      parameter_ordering->AddElementToGroup(center.data(), group_id);
    }
  }

  // dmap_scales_ in own group (1-D vs 3-D frame_centers/cams_in_rig).
  ++group_id;
  for (auto& [image_id, scale] : dmap_scales_) {
    if (problem_->HasParameterBlock(&scale)) {
      parameter_ordering->AddElementToGroup(&scale, group_id);
    }
  }
}

void GlobalPositioner::ParameterizeVariables(Reconstruction& reconstruction) {
  // For the global positioning, do not set any camera to be constant for easier
  // convergence

  // Initialize cams_in_rig_ with random values if optimizing positions.
  if (options_.optimize_positions) {
    for (auto& [sensor_id, center] : cams_in_rig_) {
      if (problem_->HasParameterBlock(center.data())) {
        center = RandVector3d(-1, 1);
      }
    }
  }

  // If not optimizing positions, set frame centers to be constant.
  if (!options_.optimize_positions) {
    for (auto& [frame_id, center] : frame_centers_) {
      if (problem_->HasParameterBlock(center.data())) {
        problem_->SetParameterBlockConstant(center.data());
      }
    }
  }

  // If do not optimize the rotations, set the camera rotations to be constant
  if (!options_.optimize_points) {
    for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
      if (problem_->HasParameterBlock(point3D.xyz.data())) {
        problem_->SetParameterBlockConstant(
            reconstruction.Point3D(point3D_id).xyz.data());
      }
    }
  }

  // If do not optimize the scales, set the scales to be constant
  if (!options_.optimize_scales) {
    for (double& scale : scales_) {
      if (problem_->HasParameterBlock(&scale)) {
        problem_->SetParameterBlockConstant(&scale);
      }
    }
  }

  // Lower-bound dmap_scales_ in linear space (no bound needed in log space).
  if (!options_.use_log_scale_for_depth_map_scales) {
    for (auto& [image_id, scale] : dmap_scales_) {
      if (problem_->HasParameterBlock(&scale)) {
        problem_->SetParameterLowerBound(&scale, 0, 1e-5);
      }
    }
  }
  // Pin first scale to remove gauge ambiguity. Skip when metric-depth is
  // active (depth priors + scale priors already anchor the gauge).
  if (!options_.use_metric_depth_constraint) {
    for (double& scale : scales_) {
      if (problem_->HasParameterBlock(&scale)) {
        problem_->SetParameterBlockConstant(&scale);
        break;
      }
    }
  }

#ifdef COLMAP_CUDA_ENABLED
  bool cuda_solver_enabled = false;

#if (CERES_VERSION_MAJOR >= 3 ||                                \
     (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 2)) && \
    !defined(CERES_NO_CUDA)
  if (options_.use_gpu &&
      reconstruction.NumImages() >=
          static_cast<size_t>(options_.min_num_images_gpu_solver)) {
    cuda_solver_enabled = true;
    options_.solver_options.dense_linear_algebra_library_type = ceres::CUDA;
  }
#else
  if (options_.use_gpu) {
    LOG_FIRST_N(WARNING, 1)
        << "Requested to use GPU for bundle adjustment, but Ceres was "
           "compiled without CUDA support. Falling back to CPU-based dense "
           "solvers.";
  }
#endif

#if (CERES_VERSION_MAJOR >= 3 ||                                \
     (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 3)) && \
    !defined(CERES_NO_CUDSS)
  if (options_.use_gpu &&
      reconstruction.NumImages() >=
          static_cast<size_t>(options_.min_num_images_gpu_solver)) {
    cuda_solver_enabled = true;
    options_.solver_options.sparse_linear_algebra_library_type =
        ceres::CUDA_SPARSE;
  }
#else
  if (options_.use_gpu) {
    LOG_FIRST_N(WARNING, 1)
        << "Requested to use GPU for bundle adjustment, but Ceres was "
           "compiled without cuDSS support. Falling back to CPU-based sparse "
           "solvers.";
  }
#endif

  if (cuda_solver_enabled) {
    const std::vector<int> gpu_indices = CSVToVector<int>(options_.gpu_index);
    THROW_CHECK_GT(gpu_indices.size(), 0);
    SetBestCudaDevice(gpu_indices[0]);
  }
#else
  if (options_.use_gpu) {
    LOG_FIRST_N(WARNING, 1)
        << "Requested to use GPU for bundle adjustment, but COLMAP was "
           "compiled without CUDA support. Falling back to CPU-based "
           "solvers.";
  }
#endif  // COLMAP_CUDA_ENABLED

  // Set up the options for the solver
  // Do not use iterative solvers, for its suboptimal performance.
  if (reconstruction.NumPoints3D() > 0) {
    options_.solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
    options_.solver_options.preconditioner_type = ceres::CLUSTER_TRIDIAGONAL;
  } else {
    options_.solver_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options_.solver_options.preconditioner_type = ceres::JACOBI;
  }
}

void GlobalPositioner::ConvertBackResults(Reconstruction& reconstruction) {
  // Convert optimized frame centers back to rig_from_world translations.
  for (const auto& [frame_id, center] : frame_centers_) {
    Rigid3d& rig_from_world = reconstruction.Frame(frame_id).RigFromWorld();
    rig_from_world.translation() = rig_from_world.rotation() * -center;
  }

  // Convert optimized cam_in_rig back to sensor_from_rig translations.
  for (const auto& [sensor_id, center] : cams_in_rig_) {
    // Find the rig containing this sensor.
    for (const auto& [rig_id, rig] : reconstruction.Rigs()) {
      if (!rig.HasSensor(sensor_id)) {
        continue;
      }
      Rigid3d& sensor_from_rig =
          reconstruction.Rig(rig_id).SensorFromRig(sensor_id);
      sensor_from_rig.translation() = sensor_from_rig.rotation() * -center;
      break;
    }
  }
}

void GlobalPositioner::FilterDepthOutliers(
    const Reconstruction& reconstruction) {
  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    // Regular observations
    for (const auto& observation : point3D.track.Elements()) {
      if (!reconstruction.ExistsImage(observation.image_id)) continue;
      const Image& image = reconstruction.Image(observation.image_id);
      if (!image.HasPose()) continue;
      if (DepthOutlierFlag(image,
                           observation.point2D_idx,
                           point3D.xyz,
                           options_.use_log_scale_for_depth_map_scales,
                           dmap_scales_,
                           observation.image_id,
                           options_.filter_depth_outlier_sigma)) {
        depth_outliers_.insert({observation.image_id, observation.point2D_idx});
      }
    }
    if (options_.use_lc_observations) {
      for (const auto& observation : point3D.track.lc_elements) {
        if (!reconstruction.ExistsImage(observation.image_id)) continue;
        const Image& image = reconstruction.Image(observation.image_id);
        if (!image.HasPose()) continue;
        if (DepthOutlierFlag(image,
                             observation.point2D_idx,
                             point3D.xyz,
                             options_.use_log_scale_for_depth_map_scales,
                             dmap_scales_,
                             observation.image_id,
                             options_.filter_depth_outlier_sigma)) {
          depth_outliers_.insert(
              {observation.image_id, observation.point2D_idx});
        }
      }
    }
  }
  VLOG(2) << "FilterDepthOutliers: flagged " << depth_outliers_.size()
          << " observations as depth outliers.";
}

void GlobalPositioner::InitializeDepthMapScalesFromObservations(
    const Reconstruction& reconstruction) {
  // Per-image scale estimates: scale = z_est / depth_prior.
  std::map<image_t, std::vector<double>> image_scale_estimates;

  auto consume_observation = [&](image_t image_id,
                                 point2D_t feature_id,
                                 const Eigen::Vector3d& point_world) {
    if (!reconstruction.ExistsImage(image_id)) return;
    const Image& image = reconstruction.Image(image_id);
    if (!image.HasPose()) return;

    if (feature_id >= image.depth_prior_validity.size() ||
        !image.depth_prior_validity[feature_id]) {
      return;
    }
    if (feature_id >= image.depth_priors.size()) return;

    const double depth_prior = image.depth_priors[feature_id];
    if (depth_prior <= 1e-6) return;

    // z_est = (cam_from_world * X_world)[2]
    const Eigen::Vector3d point_cam = image.CamFromWorld() * point_world;
    const double z_est = point_cam[2];
    if (z_est <= 1e-6) return;

    const double scale_estimate = z_est / depth_prior;
    if (scale_estimate > 1e-6 && scale_estimate < 1e6) {
      image_scale_estimates[image_id].push_back(scale_estimate);
    }
  };

  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    for (const auto& observation : point3D.track.Elements()) {
      consume_observation(
          observation.image_id, observation.point2D_idx, point3D.xyz);
    }
    if (options_.use_lc_observations) {
      for (const auto& observation : point3D.track.lc_elements) {
        consume_observation(
            observation.image_id, observation.point2D_idx, point3D.xyz);
      }
    }
  }

  for (auto& [image_id, scale_estimates] : image_scale_estimates) {
    if (scale_estimates.empty()) continue;

    std::sort(scale_estimates.begin(), scale_estimates.end());
    const double median_scale = scale_estimates[scale_estimates.size() / 2];

    const double initial_value = options_.use_log_scale_for_depth_map_scales
                                     ? std::log(median_scale)
                                     : median_scale;

    dmap_scales_[image_id] = initial_value;
    dmap_scale_observation_counts_[image_id] = 0;
  }

  VLOG(2) << "InitializeDepthMapScalesFromObservations: seeded "
          << dmap_scales_.size() << " image scales from observations.";
}

bool RunGlobalPositioning(const GlobalPositionerOptions& options,
                          const PoseGraph& pose_graph,
                          Reconstruction& reconstruction) {
  GlobalPositioner positioner(options);
  return positioner.Solve(pose_graph, reconstruction);
}

void GlobalPositioner::ValidateDebugInitializationOptions() const {
  if (options_.debug_initialization_stage == "gp2") {
    THROW_CHECK(false)
        << "GP2 replay must not load a separate random initialization. "
           "Run GP2 from GP1 output plus deterministic GP2 inputs.";
  }
  if (!options_.debug_initialization_stage.empty() &&
      options_.debug_initialization_stage != "gp1") {
    THROW_CHECK(false)
        << "Debug GP initialization stage must be empty, 'gp1', or rejected "
           "'gp2'; got '"
        << options_.debug_initialization_stage << "'.";
  }
  if (HasDebugInitialization(options_) && options_.use_init) {
    THROW_CHECK(false)
        << "Debug GP initialization maps are only for GP1 random-init replay. "
           "Refusing to combine them with use_init=true, which is the normal "
           "GP2 warm-start path.";
  }
}

void GlobalPositioner::ValidateDebugInitializationConsumed() const {
  for (const auto& [frame_id, _] : options_.debug_initial_frame_centers) {
    if (initial_frame_centers_.find(frame_id) == initial_frame_centers_.end()) {
      THROW_CHECK(false)
          << "Debug GP frame-center init contains unused frame_id=" << frame_id
          << ". Input graph/reconstruction does not match the "
          << "recorded GP1 initialization state.";
    }
  }
  for (const auto& [point3D_id, _] : options_.debug_initial_point3D_xyz) {
    if (initial_point3D_xyz_.find(point3D_id) == initial_point3D_xyz_.end()) {
      THROW_CHECK(false) << "Debug GP point3D init contains unused point3D_id="
                         << point3D_id
                         << ". Input graph/reconstruction does not match the "
                         << "recorded GP1 initialization state.";
    }
  }
  for (const auto& [scale_key, _] : options_.debug_initial_bata_scales) {
    if (initial_bata_scales_.find(scale_key) == initial_bata_scales_.end()) {
      THROW_CHECK(false)
          << "Debug GP BATA-scale init contains unused observation key="
          << scale_key << ". Input graph/reconstruction does not match the "
          << "recorded GP1 initialization state.";
    }
  }
}

std::unordered_map<point3D_t, Eigen::Vector3d>
GlobalPositioner::GetFinalPoint3DXYZ(
    const Reconstruction& reconstruction) const {
  std::unordered_map<point3D_t, Eigen::Vector3d> out;
  out.reserve(initial_point3D_xyz_.size());
  for (const auto& [point3D_id, _] : initial_point3D_xyz_) {
    if (reconstruction.ExistsPoint3D(point3D_id)) {
      out.emplace(point3D_id, reconstruction.Point3D(point3D_id).xyz);
    }
  }
  return out;
}

std::unordered_map<std::string, double> GlobalPositioner::GetFinalBataScales()
    const {
  std::unordered_map<std::string, double> out;
  out.reserve(bata_scale_indices_.size());
  for (const auto& [key, index] : bata_scale_indices_) {
    if (index < scales_.size()) {
      out.emplace(key, scales_[index]);
    }
  }
  return out;
}

}  // namespace colmap
