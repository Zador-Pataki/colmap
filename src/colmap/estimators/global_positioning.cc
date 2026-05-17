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
#include <optional>
#include <utility>

namespace colmap {
namespace {

Eigen::Vector3d RandVector3d(double low, double high) {
  return Eigen::Vector3d(RandomUniformReal(low, high),
                         RandomUniformReal(low, high),
                         RandomUniformReal(low, high));
}

GlobalPositioningTraceParameterBlockDescriptor TraceParameterBlock(
    std::string role, std::string kind, const uint64_t id) {
  return {std::move(role), std::move(kind), id};
}

std::vector<double> TraceVector3d(const Eigen::Vector3d& value) {
  return {value.x(), value.y(), value.z()};
}

std::vector<double> TraceMatrix3dRowMajor(const Eigen::Matrix3d& value) {
  std::vector<double> entries;
  entries.reserve(9);
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      entries.push_back(value(row, col));
    }
  }
  return entries;
}

std::vector<double> TraceQuaternionWxyz(const Eigen::Quaterniond& value) {
  return {value.w(), value.x(), value.y(), value.z()};
}

std::string TraceLossFunctionType(const LossFunctionType type) {
  switch (type) {
    case LossFunctionType::TRIVIAL:
      return "trivial";
    case LossFunctionType::SOFT_L1:
      return "soft_l1";
    case LossFunctionType::CAUCHY:
      return "cauchy";
    case LossFunctionType::HUBER:
      return "huber";
  }
  LOG(FATAL) << "Unhandled LossFunctionType: " << static_cast<int>(type);
}

GlobalPositioningTraceLossConfig TraceLossConfig(
    const std::string& bucket,
    const LossConfig& loss,
    std::string source,
    const double extra_weight = 1.0,
    const std::optional<double> observation_count_weight = std::nullopt) {
  GlobalPositioningTraceLossConfig trace_loss;
  trace_loss.bucket = bucket;
  trace_loss.type = TraceLossFunctionType(loss.type);
  trace_loss.scale = loss.scale;
  trace_loss.weight = loss.weight * extra_weight;
  trace_loss.source = std::move(source);
  trace_loss.observation_count_weight = observation_count_weight;
  return trace_loss;
}

std::string TraceMetricDepthResidualType(
    const MetricDepthResidualType residual_type) {
  switch (residual_type) {
    case MetricDepthResidualType::kLinear:
      return "linear";
    case MetricDepthResidualType::kLog:
      return "log";
    case MetricDepthResidualType::kLogLinear:
      return "log_linear";
  }
  LOG(FATAL) << "Unhandled MetricDepthResidualType: "
             << static_cast<int>(residual_type);
}

GlobalPositioningTraceFixedParameters TraceBataRefFrameFixedParameters(
    const Eigen::Vector3d& cam_from_point3D_dir,
    const std::optional<Eigen::Matrix3d>& keypoint_covariance_world) {
  GlobalPositioningTraceFixedParameters fixed_parameters;
  fixed_parameters.cam_from_point3D_dir = TraceVector3d(cam_from_point3D_dir);
  if (keypoint_covariance_world.has_value()) {
    fixed_parameters.keypoint_covariance_world_row_major =
        TraceMatrix3dRowMajor(*keypoint_covariance_world);
  }
  return fixed_parameters;
}

GlobalPositioningTraceFixedParameters TraceBataConstantRigFixedParameters(
    const Eigen::Vector3d& cam_from_point3D_dir,
    const Eigen::Vector3d& cam_from_rig_dir) {
  GlobalPositioningTraceFixedParameters fixed_parameters;
  fixed_parameters.cam_from_point3D_dir = TraceVector3d(cam_from_point3D_dir);
  fixed_parameters.cam_from_rig_dir = TraceVector3d(cam_from_rig_dir);
  return fixed_parameters;
}

GlobalPositioningTraceFixedParameters TraceBataVariableRigFixedParameters(
    const Eigen::Vector3d& cam_from_point3D_dir,
    const Eigen::Quaterniond& rig_from_world_rotation) {
  GlobalPositioningTraceFixedParameters fixed_parameters;
  fixed_parameters.cam_from_point3D_dir = TraceVector3d(cam_from_point3D_dir);
  fixed_parameters.rig_from_world_rotation_wxyz =
      TraceQuaternionWxyz(rig_from_world_rotation);
  fixed_parameters.world_from_rig_rotation_wxyz =
      TraceQuaternionWxyz(rig_from_world_rotation.inverse());
  return fixed_parameters;
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

GlobalPositioningTraceFixedParameters TraceMetricDepthFixedParameters(
    const Eigen::Quaterniond& camera_rotation,
    const MetricDepthOptions& metric_depth_options) {
  GlobalPositioningTraceFixedParameters fixed_parameters;
  fixed_parameters.camera_rotation_wxyz = TraceQuaternionWxyz(camera_rotation);
  fixed_parameters.metric_depth_use_log_scale =
      metric_depth_options.use_log_scale;
  fixed_parameters.metric_depth_residual_type =
      TraceMetricDepthResidualType(metric_depth_options.residual_type);
  fixed_parameters.metric_depth_zero_residual_behind =
      metric_depth_options.zero_residual_behind;
  fixed_parameters.metric_depth_log_linear_threshold =
      metric_depth_options.log_linear_threshold;
  return fixed_parameters;
}

GlobalPositioningTraceFixedParameters TraceScalePriorFixedParameters(
    const bool use_log_scale_for_depth_map_scales,
    const double scale_prior_stddev) {
  GlobalPositioningTraceFixedParameters fixed_parameters;
  fixed_parameters.scale_prior_target =
      use_log_scale_for_depth_map_scales ? 0.0 : 1.0;
  fixed_parameters.scale_prior_stddev = scale_prior_stddev;
  return fixed_parameters;
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

using TraceValue = GlobalPositioningTraceValue;

std::string DepthOutlierSource(
    const std::set<std::pair<image_t, point2D_t>>& runtime_depth_outliers,
    const Image& image,
    const TrackElement& observation) {
  if (runtime_depth_outliers.count(
          {observation.image_id, observation.point2D_idx}) > 0) {
    return "runtime_filter";
  }
  if (observation.point2D_idx < image.is_depth_outlier.size() &&
      image.is_depth_outlier[observation.point2D_idx]) {
    return "external_annotation";
  }
  return "none";
}

bool HasValidDepthPriorMetadata(const Image& image,
                                const TrackElement& observation) {
  return observation.point2D_idx < image.depth_prior_validity.size() &&
         image.depth_prior_validity[observation.point2D_idx] &&
         observation.point2D_idx < image.depth_priors.size() &&
         observation.point2D_idx < image.depth_prior_stddevs.size();
}

GlobalPositioningTraceLossConfig GeometryTraceLossConfig(
    const GlobalPositionerOptions& options, const std::string& bucket) {
  if (bucket == "geometry_uncalibrated_downweighted") {
    return TraceLossConfig(bucket,
                           options.loss,
                           "GlobalPositionerOptions.loss+"
                           "uncalibrated_loss_downweight",
                           options.uncalibrated_loss_downweight);
  }
  if (bucket == "geometry_normal_trackstart") {
    return TraceLossConfig(bucket,
                           options.loss_normal_geometry_trackstart,
                           "GlobalPositionerOptions."
                           "loss_normal_geometry_trackstart");
  }
  if (bucket == "geometry_normal_inlier") {
    return TraceLossConfig(bucket,
                           options.loss_normal_geometry_inlier,
                           "GlobalPositionerOptions."
                           "loss_normal_geometry_inlier");
  }
  if (bucket == "geometry_normal_default" &&
      options.use_metric_depth_constraint) {
    return TraceLossConfig(bucket,
                           options.loss_normal_geometry,
                           "GlobalPositionerOptions.loss_normal_geometry");
  }
  if (bucket == "geometry_lc") {
    return TraceLossConfig(bucket,
                           options.loss_lc_geometry,
                           "GlobalPositionerOptions.loss_lc_geometry");
  }
  return TraceLossConfig(bucket, options.loss, "GlobalPositionerOptions.loss");
}

GlobalPositioningTraceLossConfig DepthTraceLossConfig(
    const GlobalPositionerOptions& options, const std::string& bucket) {
  if (bucket == "depth_runtime_outlier_soft_fallback") {
    return TraceLossConfig(bucket,
                           options.loss_soft_outlier_fallback,
                           "GlobalPositionerOptions."
                           "loss_soft_outlier_fallback");
  }
  if (bucket == "depth_lc") {
    return TraceLossConfig(
        bucket, options.loss_lc_depth, "GlobalPositionerOptions.loss_lc_depth");
  }
  if (bucket == "depth_normal_trackstart") {
    return TraceLossConfig(bucket,
                           options.loss_normal_depth_trackstart,
                           "GlobalPositionerOptions."
                           "loss_normal_depth_trackstart");
  }
  if (bucket == "depth_normal_inlier") {
    return TraceLossConfig(bucket,
                           options.loss_normal_depth_inlier,
                           "GlobalPositionerOptions.loss_normal_depth_inlier");
  }
  if (bucket == "depth_normal_external_outlier") {
    return TraceLossConfig(bucket,
                           options.loss_normal_depth_outlier,
                           "GlobalPositionerOptions."
                           "loss_normal_depth_outlier");
  }
  return TraceLossConfig(bucket,
                         options.loss_normal_depth,
                         "GlobalPositionerOptions.loss_normal_depth");
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
  if (reconstruction.NumImages() == 0) {
    LOG(ERROR) << "Number of images = " << reconstruction.NumImages();
    return false;
  }
  if (reconstruction.NumPoints3D() == 0) {
    LOG(ERROR) << "Number of tracks = " << reconstruction.NumPoints3D();
    return false;
  }

  tracer_ = std::make_unique<GlobalPositioningTracer>(options_.trace);

  {
    tracer_->WriteEvent(
        "run_started",
        "solve",
        {{"num_images", TraceValue::UInt(reconstruction.NumImages())},
         {"num_points3D", TraceValue::UInt(reconstruction.NumPoints3D())},
         {"num_frames", TraceValue::UInt(reconstruction.NumFrames())},
         {"num_cameras", TraceValue::UInt(reconstruction.NumCameras())},
         {"trace_level",
          TraceValue::String(
              GlobalPositioningTraceLevelToString(options_.trace.level))}});

    // TODO: extend rig branch in AddObservationToProblem to add
    // MetricDepthError for non-ref images. Until then, fail loud on
    // multi-camera rigs + use_metric_depth_constraint.
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

    tracer_->WriteEvent("problem_setup_started", "setup_problem");

    // Setup the problem.
    SetupProblem(pose_graph, reconstruction);

    tracer_->WriteEvent(
        "problem_setup_finished",
        "setup_problem",
        {{"reserved_scale_capacity", TraceValue::UInt(scales_.capacity())}});

    // Initialize camera translations to be random.
    // Also, convert the camera pose translation to be the camera center.
    InitializeRandomPositions(pose_graph, reconstruction);

    tracer_->WriteEvent(
        "initialization_finished",
        "initialization",
        {{"num_frame_centers", TraceValue::UInt(frame_centers_.size())},
         {"use_init", TraceValue::Bool(options_.use_init)},
         {"generate_random_positions",
          TraceValue::Bool(options_.generate_random_positions)},
         {"generate_random_points",
          TraceValue::Bool(options_.generate_random_points)}});

    // No caller-supplied seed for dmap_scales_; derive one from per-image
    // median observed z_est/depth_prior.
    const bool initialize_dmap_scales =
        options_.use_metric_depth_constraint && options_.use_init &&
        !options_.initial_dmap_scales.has_value();
    std::string dmap_scale_skip_reason;
    if (initialize_dmap_scales) {
      InitializeDepthMapScalesFromObservations(reconstruction);
    } else if (!options_.use_metric_depth_constraint) {
      dmap_scale_skip_reason = "metric_depth_disabled";
    } else if (!options_.use_init) {
      dmap_scale_skip_reason = "use_init_disabled";
    } else {
      dmap_scale_skip_reason = "initial_dmap_scales_provided";
    }
    tracer_->WriteEvent(
        "dmap_scale_initialization_finished",
        "initialization",
        {{"skipped", TraceValue::Bool(!initialize_dmap_scales)},
         {"reason", TraceValue::String(dmap_scale_skip_reason)},
         {"num_dmap_scales", TraceValue::UInt(dmap_scales_.size())}});

    tracer_->WriteEvent("residual_build_started", "problem_build");

    // Add the point to camera constraints to the problem.
    AddPointToCameraConstraints(reconstruction);

    tracer_->WriteEvent(
        "problem_built",
        "problem_build",
        {{"num_residual_blocks",
          TraceValue::Int(problem_->NumResidualBlocks())},
         {"num_parameter_blocks",
          TraceValue::Int(problem_->NumParameterBlocks())},
         {"num_scales", TraceValue::UInt(scales_.size())},
         {"num_frame_centers", TraceValue::UInt(frame_centers_.size())},
         {"num_dmap_scales", TraceValue::UInt(dmap_scales_.size())}});

    if (options_.use_parameter_block_ordering) {
      AddCamerasAndPointsToParameterGroups(reconstruction);
    }

    // Parameterize the variables, set image poses / tracks / scales to be
    // constant if desired.
    ParameterizeVariables(reconstruction);

    tracer_->WriteEvent(
        "parameterization_finished",
        "parameterization",
        {{"optimize_positions", TraceValue::Bool(options_.optimize_positions)},
         {"optimize_points", TraceValue::Bool(options_.optimize_points)},
         {"optimize_scales", TraceValue::Bool(options_.optimize_scales)},
         {"use_gpu_effective", TraceValue::Bool(use_gpu_effective_)}});

    LOG(INFO) << "Solving the global positioner problem";

    ceres::Solver::Summary summary;
    options_.solver_options.num_threads =
        GetEffectiveNumThreads(options_.solver_options.num_threads);
    options_.solver_options.minimizer_progress_to_stdout = VLOG_IS_ON(2);

    ceres::Solver::Options solver_options = options_.solver_options;
    const bool trace_parameter_snapshots = tracer_->ParameterSnapshotsEnabled();
    const bool trace_residual_values = tracer_->ResidualValuesEnabled();
    if (trace_parameter_snapshots || trace_residual_values) {
      THROW_CHECK_GT(options_.trace.snapshot_every_n_iterations, 0)
          << "Global positioning trace snapshot_every_n_iterations must be "
             "positive when sampled trace artifacts are enabled.";
      solver_options.update_state_every_iteration = true;
    }
    std::unique_ptr<ceres::IterationCallback> trace_callback;
    if (tracer_->Enabled()) {
      trace_callback = tracer_->CreateIterationCallback({
          *problem_,
          reconstruction,
          frame_centers_,
          scales_,
          dmap_scales_,
          cams_in_rig_,
      });
      solver_options.callbacks.push_back(trace_callback.get());
    }

    tracer_->WriteEvent(
        "solve_started",
        "ceres_solve",
        {{"max_num_iterations",
          TraceValue::Int(solver_options.max_num_iterations)},
         {"num_threads", TraceValue::Int(solver_options.num_threads)},
         {"linear_solver_type",
          TraceValue::String(ceres::LinearSolverTypeToString(
              solver_options.linear_solver_type))},
         {"preconditioner_type",
          TraceValue::String(ceres::PreconditionerTypeToString(
              solver_options.preconditioner_type))}});

    ceres::Solve(solver_options, problem_.get(), &summary);

    const bool is_solution_usable = summary.IsSolutionUsable();
    tracer_->WriteEvent(
        "solve_finished",
        "ceres_solve",
        {{"is_solution_usable", TraceValue::Bool(is_solution_usable)},
         {"termination_type",
          TraceValue::String(
              ceres::TerminationTypeToString(summary.termination_type))},
         {"message", TraceValue::String(summary.message)},
         {"initial_cost", TraceValue::Double(summary.initial_cost)},
         {"final_cost", TraceValue::Double(summary.final_cost)},
         {"num_successful_steps",
          TraceValue::Int(summary.num_successful_steps)},
         {"num_unsuccessful_steps",
          TraceValue::Int(summary.num_unsuccessful_steps)},
         {"total_time_sec",
          TraceValue::Double(summary.total_time_in_seconds)}});

    if (VLOG_IS_ON(2)) {
      LOG(INFO) << summary.FullReport();
    } else {
      LOG(INFO) << summary.BriefReport();
    }

    ConvertBackResults(reconstruction);
    tracer_->WriteEvent(
        "results_converted",
        "convert_results",
        {{"num_frame_centers", TraceValue::UInt(frame_centers_.size())},
         {"num_cams_in_rig", TraceValue::UInt(cams_in_rig_.size())}});
    tracer_->MarkFinished();
    return is_solution_usable;
  }
}

void GlobalPositioner::SetupProblem(const PoseGraph& pose_graph,
                                    const Reconstruction& reconstruction) {
  ceres::Problem::Options problem_options;
  problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  problem_ = std::make_unique<ceres::Problem>(problem_options);
  loss_function_ = options_.CreateLossFunction();

  // Clear temporary storage from previous runs.
  frame_centers_.clear();
  cams_in_rig_.clear();
  per_image_scale_losses_.clear();
  if (tracer_ == nullptr) {
    tracer_ = std::make_unique<GlobalPositioningTracer>(options_.trace);
  }
  tracer_->ResetProblemState();

  // Reserve scales_ for both regular observations and lc_elements.
  // Underestimating triggers ``vector::push_back`` reallocation mid-build,
  // which invalidates the ``&scale`` data pointers that earlier residual
  // blocks already stored.
  scales_.clear();
  cams_in_rig_.reserve(reconstruction.NumCameras());
  size_t total_observations = 0;
  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    total_observations += point3D.track.Length();
    total_observations += point3D.track.lc_elements.size();
  }
  scales_.reserve(total_observations);
}

const std::vector<GlobalPositioningResidualReplayEntry>&
GlobalPositioner::ResidualReplayEntriesForTest() const {
  static const std::vector<GlobalPositioningResidualReplayEntry>
      kEmptyReplayEntries;
  return tracer_ == nullptr ? kEmptyReplayEntries : tracer_->ReplayEntries();
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
      frame_centers_[frame_id] = options_.random_init_scale * RandVector3d(-1, 1);
    } else {
      frame_centers_[frame_id] = frame.RigFromWorld().TgtOriginInSrc();
    }
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
      if (tracer_->ResidualLedgerEnabled()) {
        for (const auto& observation : point3D.track.Elements()) {
          GlobalPositioningResidualDescriptor skip;
          skip.residual_type = "bata_ref_frame";
          skip.point3D_id = point3D_id;
          skip.image_id = observation.image_id;
          skip.point2D_idx = observation.point2D_idx;
          tracer_->RecordSkip(skip, "track_min_view_gate");
          if (options_.use_metric_depth_constraint) {
            skip.residual_type = "metric_depth";
            tracer_->RecordSkip(skip, "track_min_view_gate");
          }
        }
        for (const auto& observation : point3D.track.lc_elements) {
          GlobalPositioningResidualDescriptor skip;
          skip.residual_type = "bata_ref_frame";
          skip.point3D_id = point3D_id;
          skip.image_id = observation.image_id;
          skip.point2D_idx = observation.point2D_idx;
          skip.is_lc_observation = true;
          const std::string skip_reason = options_.use_lc_observations
                                              ? "track_min_view_gate"
                                              : "lc_observation_disabled";
          tracer_->RecordSkip(skip, skip_reason);
          if (options_.use_metric_depth_constraint) {
            skip.residual_type = "metric_depth";
            tracer_->RecordSkip(skip, skip_reason);
          }
        }
      }
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
      if (scale_prior_cost == nullptr) {
        GlobalPositioningResidualDescriptor skip;
        skip.residual_type = "scale_prior";
        skip.image_id = image_id;
        skip.dmap_scale_image_id = image_id;
        skip.loss_bucket = "scale_prior";
        tracer_->RecordSkip(skip, "scale_prior_cost_create_failed");
        continue;
      }

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

      GlobalPositioningResidualDescriptor residual;
      residual.residual_type = "scale_prior";
      residual.image_id = image_id;
      residual.dmap_scale_image_id = image_id;
      residual.loss_bucket = "scale_prior";
      residual.loss = TraceLossConfig("scale_prior",
                                      options_.loss_scale_prior,
                                      "GlobalPositionerOptions."
                                      "loss_scale_prior+observation_count",
                                      obs_count,
                                      obs_count);
      residual.fixed_parameters = TraceScalePriorFixedParameters(
          options_.use_log_scale_for_depth_map_scales,
          options_.scale_prior_stddev);
      tracer_->RecordResidual(
          residual,
          scale_prior_cost,
          obs_count_scaled_loss,
          {&scale},
          {TraceParameterBlock("dmap_scale", "dmap_scale", image_id)});
    }
  }
  tracer_->RecordBucketSummaries();
  VLOG(2) << "GP: residual blocks=" << problem_->NumResidualBlocks()
          << ", parameter blocks=" << problem_->NumParameterBlocks()
          << ", scales=" << scales_.size()
          << ", frame_centers=" << frame_centers_.size()
          << ", dmap_scales=" << dmap_scales_.size();
}

void GlobalPositioner::AddPoint3DToProblem(point3D_t point3D_id,
                                           Reconstruction& reconstruction) {
  const bool random_initialization =
      options_.optimize_points && options_.generate_random_points &&
      !options_.use_init;

  Point3D& point3D = reconstruction.Point3D(point3D_id);

  // Only set the points to be random if they are needed to be optimized
  if (random_initialization) {
    point3D.xyz = options_.random_init_scale * RandVector3d(-1, 1);
  }

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
  } else if (tracer_->ResidualLedgerEnabled()) {
    for (const auto& observation : point3D.track.lc_elements) {
      GlobalPositioningResidualDescriptor skip;
      skip.residual_type = "bata_ref_frame";
      skip.point3D_id = point3D_id;
      skip.image_id = observation.image_id;
      skip.point2D_idx = observation.point2D_idx;
      skip.is_lc_observation = true;
      tracer_->RecordSkip(skip, "lc_observation_disabled");
      if (options_.use_metric_depth_constraint) {
        skip.residual_type = "metric_depth";
        tracer_->RecordSkip(skip, "lc_observation_disabled");
      }
    }
  }
}

void GlobalPositioner::AddObservationToProblem(point3D_t point3D_id,
                                               const TrackElement& observation,
                                               bool random_initialization,
                                               Reconstruction& reconstruction,
                                               bool is_lc_observation) {
  Point3D& point3D = reconstruction.Point3D(point3D_id);
  if (!reconstruction.ExistsImage(observation.image_id)) {
    GlobalPositioningResidualDescriptor skip;
    skip.residual_type = "bata_ref_frame";
    skip.point3D_id = point3D_id;
    skip.image_id = observation.image_id;
    skip.point2D_idx = observation.point2D_idx;
    skip.is_lc_observation = is_lc_observation;
    tracer_->RecordSkip(skip, "missing_image");
    if (options_.use_metric_depth_constraint) {
      skip.residual_type = "metric_depth";
      tracer_->RecordSkip(skip, "missing_image");
    }
    return;
  }

  Image& image = reconstruction.Image(observation.image_id);
  Camera& camera = reconstruction.Camera(image.CameraId());
  GlobalPositioningResidualDescriptor observation_record;
  observation_record.point3D_id = point3D_id;
  observation_record.image_id = observation.image_id;
  observation_record.point2D_idx = observation.point2D_idx;
  observation_record.frame_id = image.FrameId();
  observation_record.camera_id = image.CameraId();
  observation_record.sensor_id = camera.SensorId();
  observation_record.is_lc_observation = is_lc_observation;
  observation_record.is_ref_in_frame = image.IsRefInFrame();
  observation_record.camera_has_prior_focal_length =
      camera.has_prior_focal_length;
  observation_record.has_depth_prior =
      HasValidDepthPriorMetadata(image, observation);
  if (observation_record.has_depth_prior) {
    observation_record.depth_prior =
        image.depth_priors[observation.point2D_idx];
    observation_record.depth_sigma =
        image.depth_prior_stddevs[observation.point2D_idx];
  }
  observation_record.depth_outlier_source =
      DepthOutlierSource(depth_outliers_, image, observation);
  if (options_.use_metric_depth_constraint) {
    observation_record.dmap_scale_image_id = observation.image_id;
  }

  if (!image.HasPose()) {
    observation_record.residual_type =
        image.IsRefInFrame() ? "bata_ref_frame" : "bata_variable_rig";
    tracer_->RecordSkip(observation_record, "image_without_pose");
    if (options_.use_metric_depth_constraint) {
      observation_record.residual_type = "metric_depth";
      tracer_->RecordSkip(observation_record, "image_without_pose");
    }
    return;
  }

  const std::optional<Eigen::Vector2d> cam_point =
      image.CameraPtr()->CamFromImg(image.Point2D(observation.point2D_idx).xy);
  if (!cam_point.has_value()) {
    observation_record.residual_type =
        image.IsRefInFrame() ? "bata_ref_frame" : "bata_variable_rig";
    tracer_->RecordSkip(observation_record, "invalid_keypoint_projection");
    if (options_.use_metric_depth_constraint) {
      observation_record.residual_type = "metric_depth";
      tracer_->RecordSkip(observation_record, "invalid_keypoint_projection");
    }
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
  const size_t scale_index = scales_.size() - 1;
  tracer_->RecordScaleObservation(
      scale_index, point3D_id, observation, is_lc_observation);

  if (!options_.generate_scales && random_initialization) {
    const Eigen::Vector3d cam_from_point3D_translation =
        point3D.xyz - frame_centers_[image.FrameId()];
    scale = std::max(1e-5,
                     cam_from_point3D_dir.dot(cam_from_point3D_translation) /
                         cam_from_point3D_translation.squaredNorm());
  }

  // For calibrated and uncalibrated cameras, use different loss
  // functions
  // Down weight the uncalibrated cameras
  ceres::LossFunction* loss_function =
      (camera.has_prior_focal_length) ? loss_function_ptcam_calibrated_.get()
                                      : loss_function_ptcam_uncalibrated_.get();
  std::string geometry_loss_bucket =
      camera.has_prior_focal_length
          ? "geometry_calibrated"
          : (options_.apply_uncalibrated_loss_downweight
                 ? "geometry_uncalibrated_downweighted"
                 : "geometry_normal_default");

  // Geometry-loss cascade. Per-observation route:
  //   is_lc                         -> cached_loss_lc_geometry_
  //   TrackElement::is_track_anchor -> cached_loss_normal_geometry_trackstart_
  //   TrackElement::is_inlier       -> cached_loss_normal_geometry_inlier_
  //   else                          -> cached_loss_normal_geometry_
  if (is_lc_observation && cached_loss_lc_geometry_) {
    loss_function = cached_loss_lc_geometry_.get();
    geometry_loss_bucket = "geometry_lc";
  } else if (options_.use_metric_depth_constraint) {
    ceres::LossFunction* cascade = nullptr;
    if (observation.is_track_anchor) {
      cascade = cached_loss_normal_geometry_trackstart_.get();
      geometry_loss_bucket = "geometry_normal_trackstart";
    } else if (observation.is_inlier) {
      cascade = cached_loss_normal_geometry_inlier_.get();
      geometry_loss_bucket = "geometry_normal_inlier";
    } else {
      cascade = cached_loss_normal_geometry_.get();
      geometry_loss_bucket = "geometry_normal_default";
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
    bool uses_keypoint_covariance = false;
    std::optional<Eigen::Matrix3d> keypoint_covariance_world;
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
      uses_keypoint_covariance = cost_function != nullptr;
      if (uses_keypoint_covariance) {
        keypoint_covariance_world = cov_world;
      }
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

    GlobalPositioningResidualDescriptor residual = observation_record;
    residual.residual_type = "bata_ref_frame";
    residual.loss_bucket = geometry_loss_bucket;
    residual.loss = GeometryTraceLossConfig(options_, geometry_loss_bucket);
    residual.uses_keypoint_covariance = uses_keypoint_covariance;
    residual.fixed_parameters = TraceBataRefFrameFixedParameters(
        cam_from_point3D_dir, keypoint_covariance_world);
    tracer_->RecordResidual(
        residual,
        cost_function,
        loss_function,
        {frame_centers_[image.FrameId()].data(), point3D.xyz.data(), &scale},
        {TraceParameterBlock("frame_center", "frame_center", image.FrameId()),
         TraceParameterBlock("point3D", "point3D", point3D_id),
         TraceParameterBlock("bata_scale", "bata_scale", scale_index)});

    // 1-D MetricDepthError: anchors absolute scale via depth prior.
    if (options_.use_metric_depth_constraint) {
      AddMetricDepthResidual(
          point3D_id, observation, is_lc_observation, reconstruction);
    } else {
      GlobalPositioningResidualDescriptor skip = observation_record;
      skip.residual_type = "metric_depth";
      tracer_->RecordSkip(skip, "metric_depth_disabled");
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

      GlobalPositioningResidualDescriptor residual = observation_record;
      residual.residual_type = "bata_constant_rig";
      residual.loss_bucket = geometry_loss_bucket;
      residual.loss = GeometryTraceLossConfig(options_, geometry_loss_bucket);
      residual.fixed_parameters = TraceBataConstantRigFixedParameters(
          cam_from_point3D_dir, cam_from_rig_dir);
      tracer_->RecordResidual(
          residual,
          cost_function,
          loss_function,
          {point3D.xyz.data(), frame_centers_[image.FrameId()].data(), &scale},
          {TraceParameterBlock("point3D", "point3D", point3D_id),
           TraceParameterBlock("frame_center", "frame_center", image.FrameId()),
           TraceParameterBlock("bata_scale", "bata_scale", scale_index)});
    } else {
      // If the cam_from_rig contains nan values, it needs to be re-estimated.
      // Initialize cams_in_rig_ if not already done.
      const sensor_t sensor_id = image.CameraPtr()->SensorId();
      if (cams_in_rig_.find(sensor_id) == cams_in_rig_.end()) {
        // Will be initialized to random values in ParameterizeVariables().
        cams_in_rig_[sensor_id] = Eigen::Vector3d::Zero();
      }

      const Eigen::Quaterniond rig_from_world_rotation =
          image.FramePtr()->RigFromWorld().rotation();
      ceres::CostFunction* cost_function =
          RigBATAPairwiseDirectionCostFunctor::Create(cam_from_point3D_dir,
                                                      rig_from_world_rotation);

      problem_->AddResidualBlock(cost_function,
                                 loss_function,
                                 point3D.xyz.data(),
                                 frame_centers_[image.FrameId()].data(),
                                 cams_in_rig_[sensor_id].data(),
                                 &scale);

      GlobalPositioningResidualDescriptor residual = observation_record;
      residual.residual_type = "bata_variable_rig";
      residual.loss_bucket = geometry_loss_bucket;
      residual.loss = GeometryTraceLossConfig(options_, geometry_loss_bucket);
      residual.fixed_parameters = TraceBataVariableRigFixedParameters(
          cam_from_point3D_dir, rig_from_world_rotation);
      tracer_->RecordResidual(
          residual,
          cost_function,
          loss_function,
          {point3D.xyz.data(),
           frame_centers_[image.FrameId()].data(),
           cams_in_rig_[sensor_id].data(),
           &scale},
          {TraceParameterBlock("point3D", "point3D", point3D_id),
           TraceParameterBlock("frame_center", "frame_center", image.FrameId()),
           TraceParameterBlock("cam_in_rig", "cam_in_rig", sensor_id.id),
           TraceParameterBlock("bata_scale", "bata_scale", scale_index)});
    }
    if (!options_.use_metric_depth_constraint) {
      GlobalPositioningResidualDescriptor skip = observation_record;
      skip.residual_type = "metric_depth";
      tracer_->RecordSkip(skip, "metric_depth_disabled");
    }
  }

  problem_->SetParameterLowerBound(&scale, 0, 1e-5);
}

void GlobalPositioner::AddMetricDepthResidual(point3D_t point3D_id,
                                              const TrackElement& observation,
                                              bool is_lc_observation,
                                              Reconstruction& reconstruction) {
  if (!reconstruction.ExistsImage(observation.image_id)) {
    GlobalPositioningResidualDescriptor skip;
    skip.residual_type = "metric_depth";
    skip.point3D_id = point3D_id;
    skip.image_id = observation.image_id;
    skip.point2D_idx = observation.point2D_idx;
    skip.is_lc_observation = is_lc_observation;
    tracer_->RecordSkip(skip, "missing_image");
    return;
  }
  const Image& image = reconstruction.Image(observation.image_id);
  const Camera& camera = reconstruction.Camera(image.CameraId());

  GlobalPositioningResidualDescriptor residual;
  residual.residual_type = "metric_depth";
  residual.point3D_id = point3D_id;
  residual.image_id = observation.image_id;
  residual.point2D_idx = observation.point2D_idx;
  residual.frame_id = image.FrameId();
  residual.camera_id = image.CameraId();
  residual.sensor_id = camera.SensorId();
  residual.is_lc_observation = is_lc_observation;
  residual.is_ref_in_frame = image.IsRefInFrame();
  residual.camera_has_prior_focal_length = camera.has_prior_focal_length;
  residual.dmap_scale_image_id = observation.image_id;
  residual.depth_outlier_source =
      DepthOutlierSource(depth_outliers_, image, observation);

  if (observation.point2D_idx >= image.depth_prior_validity.size() ||
      !image.depth_prior_validity[observation.point2D_idx]) {
    tracer_->RecordSkip(residual, "missing_depth_validity");
    return;
  }
  THROW_CHECK_LT(observation.point2D_idx, image.depth_priors.size());
  THROW_CHECK_LT(observation.point2D_idx, image.depth_prior_stddevs.size());

  const double depth_prior = image.depth_priors[observation.point2D_idx];
  const double depth_sigma = image.depth_prior_stddevs[observation.point2D_idx];
  residual.has_depth_prior = true;
  residual.depth_prior = depth_prior;
  residual.depth_sigma = depth_sigma;

  if (depth_prior <= 0.0) {
    tracer_->RecordSkip(residual, "invalid_depth_prior");
    return;
  }
  if (depth_sigma <= 1e-9) {
    tracer_->RecordSkip(residual, "invalid_depth_sigma");
    return;
  }

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

  const MetricDepthOptions metric_depth_options =
      CreateMetricDepthOptions(options_);
  ceres::CostFunction* metric_depth_cost =
      MetricDepthError::Create(image.CamFromWorld().rotation(),
                               depth_prior,
                               depth_sigma,
                               metric_depth_options);

  if (metric_depth_cost == nullptr) {
    tracer_->RecordSkip(residual, "metric_depth_cost_create_failed");
    return;
  }

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
  std::string depth_loss_bucket = "none";
  const std::pair<image_t, point2D_t> obs_key{observation.image_id,
                                              observation.point2D_idx};
  if (depth_outliers_.count(obs_key) > 0) {
    if (is_lc_observation) {
      // LC outlier: skip depth residual entirely.
      delete metric_depth_cost;
      residual.loss_bucket = "depth_lc";
      tracer_->RecordSkip(residual, "runtime_lc_depth_outlier_skipped");
      return;
    }
    // Non-LC outlier: soft fallback (HuberLoss(1)).
    if (!soft_outlier_fallback_loss_) {
      soft_outlier_fallback_loss_ =
          options_.loss_soft_outlier_fallback.CreateLossFunction();
    }
    depth_loss = soft_outlier_fallback_loss_.get();
    depth_loss_bucket = "depth_runtime_outlier_soft_fallback";
  } else if (is_lc_observation) {
    depth_loss = cached_loss_lc_depth_.get();
    depth_loss_bucket = "depth_lc";
  } else if (observation.is_track_anchor) {
    depth_loss = cached_loss_normal_depth_trackstart_.get();
    depth_loss_bucket = "depth_normal_trackstart";
  } else if (observation.is_inlier) {
    depth_loss = cached_loss_normal_depth_inlier_.get();
    depth_loss_bucket = "depth_normal_inlier";
  } else if (observation.is_depth_outlier) {
    depth_loss = cached_loss_normal_depth_outlier_.get();
    depth_loss_bucket = "depth_normal_external_outlier";
  } else {
    depth_loss = cached_loss_normal_depth_.get();
    depth_loss_bucket = "depth_normal_default";
  }

  Point3D& point3D = reconstruction.Point3D(point3D_id);
  problem_->AddResidualBlock(metric_depth_cost,
                             depth_loss,
                             frame_centers_[image.FrameId()].data(),
                             point3D.xyz.data(),
                             &dmap_scales_[observation.image_id]);
  residual.loss_bucket = depth_loss_bucket;
  residual.loss = DepthTraceLossConfig(options_, depth_loss_bucket);
  residual.fixed_parameters = TraceMetricDepthFixedParameters(
      image.CamFromWorld().rotation(), metric_depth_options);
  tracer_->RecordResidual(
      residual,
      metric_depth_cost,
      depth_loss,
      {frame_centers_[image.FrameId()].data(),
       point3D.xyz.data(),
       &dmap_scales_[observation.image_id]},
      {TraceParameterBlock("frame_center", "frame_center", image.FrameId()),
       TraceParameterBlock("point3D", "point3D", point3D_id),
       TraceParameterBlock("dmap_scale", "dmap_scale", observation.image_id)});
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
  use_gpu_effective_ = false;

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
    use_gpu_effective_ = true;
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

}  // namespace colmap
