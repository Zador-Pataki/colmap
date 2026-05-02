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
#include <limits>
#include <utility>

namespace colmap {
namespace {

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

bool IsTraceEnabled(const GlobalPositioningTraceOptions& options) {
  return options.level != GlobalPositioningTraceLevel::kOff;
}

bool IsResidualLedgerTraceEnabled(
    const GlobalPositioningTraceOptions& options) {
  return static_cast<int>(options.level) >=
         static_cast<int>(GlobalPositioningTraceLevel::kResidualLedger);
}

bool IsParameterSnapshotTraceEnabled(
    const GlobalPositioningTraceOptions& options) {
  return static_cast<int>(options.level) >=
         static_cast<int>(GlobalPositioningTraceLevel::kParameterSnapshots);
}

bool IsResidualValuesTraceEnabled(
    const GlobalPositioningTraceOptions& options) {
  return static_cast<int>(options.level) >=
         static_cast<int>(GlobalPositioningTraceLevel::kResidualValues);
}

using TraceAttrs = std::map<std::string, GlobalPositioningTraceValue>;
using TraceValue = GlobalPositioningTraceValue;

template <typename T>
TraceValue TraceOptionalId(const std::optional<T>& value) {
  if (!value.has_value()) {
    return TraceValue::Null();
  }
  return TraceValue::UInt(static_cast<uint64_t>(*value));
}

TraceValue TraceOptionalDouble(const std::optional<double>& value) {
  if (!value.has_value()) {
    return TraceValue::Null();
  }
  return TraceValue::Double(*value);
}

std::optional<uint64_t> TraceSensorId(
    const std::optional<sensor_t>& sensor_id) {
  if (!sensor_id.has_value() || *sensor_id == kInvalidSensorId) {
    return std::nullopt;
  }
  return static_cast<uint64_t>(sensor_id->id);
}

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

void WriteTraceEvent(GlobalPositioningTraceRecorder* recorder,
                     std::string event_type,
                     std::string stage,
                     TraceAttrs attrs = {}) {
  if (recorder == nullptr) {
    return;
  }

  GlobalPositioningTraceRecord record;
  record.event_type = std::move(event_type);
  record.stage = std::move(stage);
  record.attrs = std::move(attrs);
  recorder->WriteEvent(std::move(record));
}

void WriteParameterSnapshot(
    GlobalPositioningTraceRecorder* recorder,
    const int iteration,
    const Reconstruction& reconstruction,
    const std::unordered_map<frame_t, Eigen::Vector3d>& frame_centers,
    const std::vector<double>& scales,
    const std::map<image_t, double>& dmap_scales,
    const std::unordered_map<sensor_t, Eigen::Vector3d>& cams_in_rig,
    const int max_snapshotted_points) {
  if (recorder == nullptr) {
    return;
  }

  GlobalPositioningTraceParameterSnapshot snapshot;
  snapshot.iteration = iteration;

  std::vector<frame_t> frame_ids;
  frame_ids.reserve(frame_centers.size());
  for (const auto& [frame_id, center] : frame_centers) {
    frame_ids.push_back(frame_id);
  }
  std::sort(frame_ids.begin(), frame_ids.end());
  snapshot.frame_centers.shape = {frame_ids.size(), 3};
  snapshot.frame_centers.ids.reserve(frame_ids.size());
  snapshot.frame_centers.values.reserve(3 * frame_ids.size());
  for (const frame_t frame_id : frame_ids) {
    const Eigen::Vector3d& center = frame_centers.at(frame_id);
    snapshot.frame_centers.ids.push_back(static_cast<uint64_t>(frame_id));
    snapshot.frame_centers.values.push_back(center.x());
    snapshot.frame_centers.values.push_back(center.y());
    snapshot.frame_centers.values.push_back(center.z());
  }

  std::vector<point3D_t> point3D_ids;
  point3D_ids.reserve(reconstruction.NumPoints3D());
  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    point3D_ids.push_back(point3D_id);
  }
  std::sort(point3D_ids.begin(), point3D_ids.end());
  if (max_snapshotted_points >= 0 &&
      point3D_ids.size() > static_cast<size_t>(max_snapshotted_points)) {
    point3D_ids.resize(static_cast<size_t>(max_snapshotted_points));
  }
  snapshot.points3D.shape = {point3D_ids.size(), 3};
  snapshot.points3D.ids.reserve(point3D_ids.size());
  snapshot.points3D.values.reserve(3 * point3D_ids.size());
  for (const point3D_t point3D_id : point3D_ids) {
    const Eigen::Vector3d& xyz = reconstruction.Point3D(point3D_id).xyz;
    snapshot.points3D.ids.push_back(static_cast<uint64_t>(point3D_id));
    snapshot.points3D.values.push_back(xyz.x());
    snapshot.points3D.values.push_back(xyz.y());
    snapshot.points3D.values.push_back(xyz.z());
  }

  snapshot.scales.shape = {scales.size()};
  snapshot.scales.ids.reserve(scales.size());
  snapshot.scales.values.reserve(scales.size());
  for (size_t scale_idx = 0; scale_idx < scales.size(); ++scale_idx) {
    snapshot.scales.ids.push_back(static_cast<uint64_t>(scale_idx));
    snapshot.scales.values.push_back(scales[scale_idx]);
  }

  if (!dmap_scales.empty()) {
    snapshot.dmap_scales.emplace();
    snapshot.dmap_scales->shape = {dmap_scales.size()};
    snapshot.dmap_scales->ids.reserve(dmap_scales.size());
    snapshot.dmap_scales->values.reserve(dmap_scales.size());
    for (const auto& [image_id, scale] : dmap_scales) {
      snapshot.dmap_scales->ids.push_back(static_cast<uint64_t>(image_id));
      snapshot.dmap_scales->values.push_back(scale);
    }
  }

  std::vector<sensor_t> sensor_ids;
  sensor_ids.reserve(cams_in_rig.size());
  for (const auto& [sensor_id, center] : cams_in_rig) {
    sensor_ids.push_back(sensor_id);
  }
  std::sort(sensor_ids.begin(), sensor_ids.end());
  if (!sensor_ids.empty()) {
    snapshot.cams_in_rig.emplace();
    snapshot.cams_in_rig->shape = {sensor_ids.size(), 3};
    snapshot.cams_in_rig->ids.reserve(sensor_ids.size());
    snapshot.cams_in_rig->values.reserve(3 * sensor_ids.size());
    for (const sensor_t sensor_id : sensor_ids) {
      const Eigen::Vector3d& cam_in_rig = cams_in_rig.at(sensor_id);
      snapshot.cams_in_rig->ids.push_back(static_cast<uint64_t>(sensor_id.id));
      snapshot.cams_in_rig->values.push_back(cam_in_rig.x());
      snapshot.cams_in_rig->values.push_back(cam_in_rig.y());
      snapshot.cams_in_rig->values.push_back(cam_in_rig.z());
    }
  }

  recorder->WriteParameterSnapshot(snapshot);
}

void WriteResidualValues(
    GlobalPositioningTraceRecorder* recorder,
    const int iteration,
    const std::vector<GlobalPositioningResidualReplayEntry>& replay_entries) {
  if (recorder == nullptr) {
    return;
  }

  GlobalPositioningTraceResidualValues residual_values;
  residual_values.iteration = iteration;
  residual_values.residual_ids.reserve(replay_entries.size());
  residual_values.residual_dims.reserve(replay_entries.size());
  residual_values.residual_offsets.reserve(replay_entries.size());
  residual_values.evaluation_success.resize(replay_entries.size(), false);
  residual_values.raw_costs.resize(replay_entries.size(),
                                   std::numeric_limits<double>::quiet_NaN());
  residual_values.robust_costs.resize(replay_entries.size(),
                                      std::numeric_limits<double>::quiet_NaN());

  size_t total_scalar_residuals = 0;
  for (const GlobalPositioningResidualReplayEntry& entry : replay_entries) {
    residual_values.residual_ids.push_back(entry.residual_id);
    residual_values.residual_dims.push_back(entry.residual_dimension);
    residual_values.residual_offsets.push_back(total_scalar_residuals);
    total_scalar_residuals += entry.residual_dimension;
  }
  residual_values.raw_residuals.resize(
      total_scalar_residuals, std::numeric_limits<double>::quiet_NaN());

  for (size_t entry_idx = 0; entry_idx < replay_entries.size(); ++entry_idx) {
    const GlobalPositioningResidualReplayEntry& entry =
        replay_entries[entry_idx];
    if (entry.cost_function == nullptr) {
      continue;
    }

    double* raw_residuals = residual_values.raw_residuals.data() +
                            residual_values.residual_offsets[entry_idx];
    const bool evaluation_success = entry.cost_function->Evaluate(
        entry.parameter_blocks.data(), raw_residuals, nullptr);
    residual_values.evaluation_success[entry_idx] = evaluation_success;
    if (!evaluation_success) {
      continue;
    }

    double squared_norm = 0.0;
    for (size_t residual_idx = 0; residual_idx < entry.residual_dimension;
         ++residual_idx) {
      squared_norm += raw_residuals[residual_idx] * raw_residuals[residual_idx];
    }

    const double raw_cost = 0.5 * squared_norm;
    residual_values.raw_costs[entry_idx] = raw_cost;
    if (entry.loss_function != nullptr) {
      double rho[3];
      entry.loss_function->Evaluate(squared_norm, rho);
      residual_values.robust_costs[entry_idx] = 0.5 * rho[0];
    } else {
      residual_values.robust_costs[entry_idx] = raw_cost;
    }
  }

  recorder->WriteResidualValues(residual_values);
}

class GlobalPositioningTraceIterationCallback
    : public ceres::IterationCallback {
 public:
  GlobalPositioningTraceIterationCallback(
      GlobalPositioningTraceRecorder* recorder,
      const Reconstruction* reconstruction,
      const std::unordered_map<frame_t, Eigen::Vector3d>* frame_centers,
      const std::vector<double>* scales,
      const std::map<image_t, double>* dmap_scales,
      const std::unordered_map<sensor_t, Eigen::Vector3d>* cams_in_rig,
      const std::vector<GlobalPositioningResidualReplayEntry>*
          residual_replay_entries,
      const int snapshot_every_n_iterations,
      const int max_snapshotted_points,
      const bool write_parameter_snapshots,
      const bool write_residual_values)
      : recorder_(recorder),
        reconstruction_(reconstruction),
        frame_centers_(frame_centers),
        scales_(scales),
        dmap_scales_(dmap_scales),
        cams_in_rig_(cams_in_rig),
        residual_replay_entries_(residual_replay_entries),
        snapshot_every_n_iterations_(snapshot_every_n_iterations),
        max_snapshotted_points_(max_snapshotted_points),
        write_parameter_snapshots_(write_parameter_snapshots),
        write_residual_values_(write_residual_values) {}

  ceres::CallbackReturnType operator()(
      const ceres::IterationSummary& summary) override {
    if (recorder_ != nullptr) {
      recorder_->WriteIteration(summary);
    }
    if ((write_parameter_snapshots_ || write_residual_values_) &&
        summary.iteration % snapshot_every_n_iterations_ == 0) {
      if (write_parameter_snapshots_) {
        WriteParameterSnapshot(recorder_,
                               summary.iteration,
                               *reconstruction_,
                               *frame_centers_,
                               *scales_,
                               *dmap_scales_,
                               *cams_in_rig_,
                               max_snapshotted_points_);
      }
      if (write_residual_values_) {
        WriteResidualValues(
            recorder_, summary.iteration, *residual_replay_entries_);
      }
    }
    return ceres::SOLVER_CONTINUE;
  }

 private:
  GlobalPositioningTraceRecorder* recorder_ = nullptr;
  const Reconstruction* reconstruction_ = nullptr;
  const std::unordered_map<frame_t, Eigen::Vector3d>* frame_centers_ = nullptr;
  const std::vector<double>* scales_ = nullptr;
  const std::map<image_t, double>* dmap_scales_ = nullptr;
  const std::unordered_map<sensor_t, Eigen::Vector3d>* cams_in_rig_ = nullptr;
  const std::vector<GlobalPositioningResidualReplayEntry>*
      residual_replay_entries_ = nullptr;
  int snapshot_every_n_iterations_ = 1;
  int max_snapshotted_points_ = -1;
  bool write_parameter_snapshots_ = false;
  bool write_residual_values_ = false;
};

class GlobalPositioningTraceStatusGuard {
 public:
  explicit GlobalPositioningTraceStatusGuard(
      GlobalPositioningTraceRecorder* recorder,
      GlobalPositioningTraceRecorder** active_recorder)
      : recorder_(recorder), active_recorder_(active_recorder) {}

  ~GlobalPositioningTraceStatusGuard() {
    if (active_recorder_ != nullptr) {
      *active_recorder_ = nullptr;
    }
  }

  void MarkFinished() {
    if (recorder_ != nullptr) {
      recorder_->MarkFinished("finished");
      recorder_ = nullptr;
    }
    if (active_recorder_ != nullptr) {
      *active_recorder_ = nullptr;
    }
  }

 private:
  GlobalPositioningTraceRecorder* recorder_ = nullptr;
  GlobalPositioningTraceRecorder** active_recorder_ = nullptr;
};

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

  std::unique_ptr<GlobalPositioningTraceRecorder> trace_recorder;
  if (IsTraceEnabled(options_.trace)) {
    trace_recorder =
        std::make_unique<GlobalPositioningTraceRecorder>(options_.trace);
  }
  trace_recorder_ = trace_recorder.get();

  {
    GlobalPositioningTraceStatusGuard trace_status(trace_recorder.get(),
                                                   &trace_recorder_);
    WriteTraceEvent(
        trace_recorder.get(),
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

    WriteTraceEvent(
        trace_recorder.get(), "problem_setup_started", "setup_problem");

    // Setup the problem.
    SetupProblem(pose_graph, reconstruction);

    WriteTraceEvent(
        trace_recorder.get(),
        "problem_setup_finished",
        "setup_problem",
        {{"reserved_scale_capacity", TraceValue::UInt(scales_.capacity())}});

    // Initialize camera translations to be random.
    // Also, convert the camera pose translation to be the camera center.
    InitializeRandomPositions(pose_graph, reconstruction);

    WriteTraceEvent(
        trace_recorder.get(),
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
    WriteTraceEvent(
        trace_recorder.get(),
        "dmap_scale_initialization_finished",
        "initialization",
        {{"skipped", TraceValue::Bool(!initialize_dmap_scales)},
         {"reason", TraceValue::String(dmap_scale_skip_reason)},
         {"num_dmap_scales", TraceValue::UInt(dmap_scales_.size())}});

    WriteTraceEvent(
        trace_recorder.get(), "residual_build_started", "problem_build");

    // Add the point to camera constraints to the problem.
    AddPointToCameraConstraints(reconstruction);

    WriteTraceEvent(
        trace_recorder.get(),
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

    WriteTraceEvent(
        trace_recorder.get(),
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
    const bool trace_parameter_snapshots = ShouldTraceParameterSnapshots();
    const bool trace_residual_values = ShouldTraceResidualValues();
    if (trace_parameter_snapshots || trace_residual_values) {
      THROW_CHECK_GT(options_.trace.snapshot_every_n_iterations, 0)
          << "Global positioning trace snapshot_every_n_iterations must be "
             "positive when sampled trace artifacts are enabled.";
      solver_options.update_state_every_iteration = true;
    }
    std::unique_ptr<GlobalPositioningTraceIterationCallback> trace_callback;
    if (trace_recorder != nullptr) {
      trace_callback =
          std::make_unique<GlobalPositioningTraceIterationCallback>(
              trace_recorder.get(),
              &reconstruction,
              &frame_centers_,
              &scales_,
              &dmap_scales_,
              &cams_in_rig_,
              &residual_replay_entries_,
              options_.trace.snapshot_every_n_iterations,
              options_.trace.max_snapshotted_points,
              trace_parameter_snapshots,
              trace_residual_values);
      solver_options.callbacks.push_back(trace_callback.get());
    }

    WriteTraceEvent(
        trace_recorder.get(),
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
    WriteTraceEvent(
        trace_recorder.get(),
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
    WriteTraceEvent(
        trace_recorder.get(),
        "results_converted",
        "convert_results",
        {{"num_frame_centers", TraceValue::UInt(frame_centers_.size())},
         {"num_cams_in_rig", TraceValue::UInt(cams_in_rig_.size())}});
    trace_status.MarkFinished();
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
  ResetResidualLedger();

  // Reserve to avoid pointer-invalidating reallocs.
  scales_.clear();
  size_t total_observations = 0;
  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    total_observations += point3D.track.Length();
    total_observations += point3D.track.lc_elements.size();
  }
  scales_.reserve(total_observations);
  scale_observations_.reserve(total_observations);
}

bool GlobalPositioner::ShouldTraceResidualLedger() const {
  return trace_recorder_ != nullptr &&
         IsResidualLedgerTraceEnabled(options_.trace);
}

bool GlobalPositioner::ShouldTraceParameterSnapshots() const {
  return trace_recorder_ != nullptr &&
         IsParameterSnapshotTraceEnabled(options_.trace);
}

bool GlobalPositioner::ShouldTraceResidualValues() const {
  return trace_recorder_ != nullptr &&
         IsResidualValuesTraceEnabled(options_.trace);
}

void GlobalPositioner::ResetResidualLedger() {
  residual_bucket_counts_.clear();
  scale_observations_.clear();
  residual_replay_entries_.clear();
}

void GlobalPositioner::RecordScaleObservation(const size_t scale_index,
                                              const point3D_t point3D_id,
                                              const TrackElement& observation,
                                              const bool is_lc_observation) {
  if (!ShouldTraceResidualLedger()) {
    return;
  }
  if (scale_observations_.size() <= scale_index) {
    scale_observations_.resize(scale_index + 1);
  }
  scale_observations_[scale_index] = {
      point3D_id,
      observation.image_id,
      observation.point2D_idx,
      is_lc_observation,
  };
}

std::string GlobalPositioner::RecordResidualBlock(
    const ResidualLedgerEntry& entry) {
  if (!ShouldTraceResidualLedger()) {
    return "";
  }

  std::string residual_id = trace_recorder_->AllocateResidualId();
  GlobalPositioningTraceRecord record;
  record.event_type = "residual_added";
  record.stage = "problem_build";
  record.attrs = {
      {"residual_id", TraceValue::String(residual_id)},
      {"residual_type", TraceValue::String(entry.residual_type)},
      {"point3D_id", TraceOptionalId(entry.point3D_id)},
      {"image_id", TraceOptionalId(entry.image_id)},
      {"point2D_idx", TraceOptionalId(entry.point2D_idx)},
      {"frame_id", TraceOptionalId(entry.frame_id)},
      {"camera_id", TraceOptionalId(entry.camera_id)},
      {"sensor_id", TraceOptionalId(TraceSensorId(entry.sensor_id))},
      {"is_lc_observation", TraceValue::Bool(entry.is_lc_observation)},
      {"is_ref_in_frame", TraceValue::Bool(entry.is_ref_in_frame)},
      {"camera_has_prior_focal_length",
       TraceValue::Bool(entry.camera_has_prior_focal_length)},
      {"loss_bucket", TraceValue::String(entry.loss_bucket)},
      {"uses_keypoint_covariance",
       TraceValue::Bool(entry.uses_keypoint_covariance)},
      {"has_depth_prior", TraceValue::Bool(entry.has_depth_prior)},
      {"depth_prior", TraceOptionalDouble(entry.depth_prior)},
      {"depth_sigma", TraceOptionalDouble(entry.depth_sigma)},
      {"dmap_scale_image_id", TraceOptionalId(entry.dmap_scale_image_id)},
      {"depth_outlier_source", TraceValue::String(entry.depth_outlier_source)},
  };
  trace_recorder_->WriteResidualBlock(std::move(record));
  ++residual_bucket_counts_[entry.residual_type + "|" + entry.loss_bucket];
  return residual_id;
}

void GlobalPositioner::RecordReplayResidual(
    const std::string& residual_id,
    const ceres::CostFunction* cost_function,
    const ceres::LossFunction* loss_function,
    std::vector<const double*> parameter_blocks) {
  if (!ShouldTraceResidualValues() || residual_id.empty() ||
      cost_function == nullptr) {
    return;
  }

  residual_replay_entries_.push_back(
      {residual_id,
       cost_function,
       loss_function,
       static_cast<size_t>(cost_function->num_residuals()),
       std::move(parameter_blocks)});
}

void GlobalPositioner::RecordResidualSkip(const ResidualLedgerEntry& entry,
                                          const std::string& skip_reason) {
  if (!ShouldTraceResidualLedger()) {
    return;
  }

  GlobalPositioningTraceRecord record;
  record.event_type = "residual_skipped";
  record.stage = "problem_build";
  record.attrs = {
      {"residual_type", TraceValue::String(entry.residual_type)},
      {"skip_reason", TraceValue::String(skip_reason)},
      {"point3D_id", TraceOptionalId(entry.point3D_id)},
      {"image_id", TraceOptionalId(entry.image_id)},
      {"point2D_idx", TraceOptionalId(entry.point2D_idx)},
      {"frame_id", TraceOptionalId(entry.frame_id)},
      {"camera_id", TraceOptionalId(entry.camera_id)},
      {"sensor_id", TraceOptionalId(TraceSensorId(entry.sensor_id))},
      {"is_lc_observation", TraceValue::Bool(entry.is_lc_observation)},
      {"is_ref_in_frame", TraceValue::Bool(entry.is_ref_in_frame)},
      {"camera_has_prior_focal_length",
       TraceValue::Bool(entry.camera_has_prior_focal_length)},
      {"loss_bucket", TraceValue::String(entry.loss_bucket)},
      {"uses_keypoint_covariance",
       TraceValue::Bool(entry.uses_keypoint_covariance)},
      {"has_depth_prior", TraceValue::Bool(entry.has_depth_prior)},
      {"depth_prior", TraceOptionalDouble(entry.depth_prior)},
      {"depth_sigma", TraceOptionalDouble(entry.depth_sigma)},
      {"dmap_scale_image_id", TraceOptionalId(entry.dmap_scale_image_id)},
      {"depth_outlier_source", TraceValue::String(entry.depth_outlier_source)},
  };
  trace_recorder_->WriteResidualSkip(std::move(record));
}

void GlobalPositioner::RecordResidualBucketSummaries() {
  if (!ShouldTraceResidualLedger()) {
    return;
  }

  for (const auto& [key, count] : residual_bucket_counts_) {
    const size_t separator = key.find('|');
    GlobalPositioningTraceRecord record;
    record.event_type = "residual_bucket_summary";
    record.stage = "problem_build";
    record.attrs = {
        {"residual_type", TraceValue::String(key.substr(0, separator))},
        {"loss_bucket",
         TraceValue::String(separator == std::string::npos
                                ? "none"
                                : key.substr(separator + 1))},
        {"count", TraceValue::UInt(count)},
    };
    trace_recorder_->WriteResidualBucketSummary(std::move(record));
  }
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
      if (ShouldTraceResidualLedger()) {
        for (const auto& observation : point3D.track.Elements()) {
          ResidualLedgerEntry skip;
          skip.residual_type = "bata_ref_frame";
          skip.point3D_id = point3D_id;
          skip.image_id = observation.image_id;
          skip.point2D_idx = observation.point2D_idx;
          RecordResidualSkip(skip, "track_min_view_gate");
          if (options_.use_metric_depth_constraint) {
            skip.residual_type = "metric_depth";
            RecordResidualSkip(skip, "track_min_view_gate");
          }
        }
        for (const auto& observation : point3D.track.lc_elements) {
          ResidualLedgerEntry skip;
          skip.residual_type = "bata_ref_frame";
          skip.point3D_id = point3D_id;
          skip.image_id = observation.image_id;
          skip.point2D_idx = observation.point2D_idx;
          skip.is_lc_observation = true;
          const std::string skip_reason = options_.use_lc_observations
                                              ? "track_min_view_gate"
                                              : "lc_observation_disabled";
          RecordResidualSkip(skip, skip_reason);
          if (options_.use_metric_depth_constraint) {
            skip.residual_type = "metric_depth";
            RecordResidualSkip(skip, skip_reason);
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
        ResidualLedgerEntry skip;
        skip.residual_type = "scale_prior";
        skip.image_id = image_id;
        skip.dmap_scale_image_id = image_id;
        skip.loss_bucket = "scale_prior";
        RecordResidualSkip(skip, "scale_prior_cost_create_failed");
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

      ResidualLedgerEntry residual;
      residual.residual_type = "scale_prior";
      residual.image_id = image_id;
      residual.dmap_scale_image_id = image_id;
      residual.loss_bucket = "scale_prior";
      const std::string residual_id = RecordResidualBlock(residual);
      RecordReplayResidual(
          residual_id, scale_prior_cost, obs_count_scaled_loss, {&scale});
    }
  }
  RecordResidualBucketSummaries();
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
  } else if (ShouldTraceResidualLedger()) {
    for (const auto& observation : point3D.track.lc_elements) {
      ResidualLedgerEntry skip;
      skip.residual_type = "bata_ref_frame";
      skip.point3D_id = point3D_id;
      skip.image_id = observation.image_id;
      skip.point2D_idx = observation.point2D_idx;
      skip.is_lc_observation = true;
      RecordResidualSkip(skip, "lc_observation_disabled");
      if (options_.use_metric_depth_constraint) {
        skip.residual_type = "metric_depth";
        RecordResidualSkip(skip, "lc_observation_disabled");
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
    ResidualLedgerEntry skip;
    skip.residual_type = "bata_ref_frame";
    skip.point3D_id = point3D_id;
    skip.image_id = observation.image_id;
    skip.point2D_idx = observation.point2D_idx;
    skip.is_lc_observation = is_lc_observation;
    RecordResidualSkip(skip, "missing_image");
    if (options_.use_metric_depth_constraint) {
      skip.residual_type = "metric_depth";
      RecordResidualSkip(skip, "missing_image");
    }
    return;
  }

  Image& image = reconstruction.Image(observation.image_id);
  Camera& camera = reconstruction.Camera(image.CameraId());
  ResidualLedgerEntry observation_record;
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
    RecordResidualSkip(observation_record, "image_without_pose");
    if (options_.use_metric_depth_constraint) {
      observation_record.residual_type = "metric_depth";
      RecordResidualSkip(observation_record, "image_without_pose");
    }
    return;
  }

  const std::optional<Eigen::Vector2d> cam_point =
      image.CameraPtr()->CamFromImg(image.Point2D(observation.point2D_idx).xy);
  if (!cam_point.has_value()) {
    observation_record.residual_type =
        image.IsRefInFrame() ? "bata_ref_frame" : "bata_variable_rig";
    RecordResidualSkip(observation_record, "invalid_keypoint_projection");
    if (options_.use_metric_depth_constraint) {
      observation_record.residual_type = "metric_depth";
      RecordResidualSkip(observation_record, "invalid_keypoint_projection");
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
  RecordScaleObservation(
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

    ResidualLedgerEntry residual = observation_record;
    residual.residual_type = "bata_ref_frame";
    residual.loss_bucket = geometry_loss_bucket;
    residual.uses_keypoint_covariance = uses_keypoint_covariance;
    const std::string residual_id = RecordResidualBlock(residual);
    RecordReplayResidual(
        residual_id,
        cost_function,
        loss_function,
        {frame_centers_[image.FrameId()].data(), point3D.xyz.data(), &scale});

    // 1-D MetricDepthError: anchors absolute scale via depth prior.
    if (options_.use_metric_depth_constraint) {
      AddMetricDepthResidual(
          point3D_id, observation, is_lc_observation, reconstruction);
    } else {
      ResidualLedgerEntry skip = observation_record;
      skip.residual_type = "metric_depth";
      RecordResidualSkip(skip, "metric_depth_disabled");
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

      ResidualLedgerEntry residual = observation_record;
      residual.residual_type = "bata_constant_rig";
      residual.loss_bucket = geometry_loss_bucket;
      const std::string residual_id = RecordResidualBlock(residual);
      RecordReplayResidual(
          residual_id,
          cost_function,
          loss_function,
          {point3D.xyz.data(), frame_centers_[image.FrameId()].data(), &scale});
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

      ResidualLedgerEntry residual = observation_record;
      residual.residual_type = "bata_variable_rig";
      residual.loss_bucket = geometry_loss_bucket;
      const std::string residual_id = RecordResidualBlock(residual);
      RecordReplayResidual(residual_id,
                           cost_function,
                           loss_function,
                           {point3D.xyz.data(),
                            frame_centers_[image.FrameId()].data(),
                            cams_in_rig_[sensor_id].data(),
                            &scale});
    }
    if (!options_.use_metric_depth_constraint) {
      ResidualLedgerEntry skip = observation_record;
      skip.residual_type = "metric_depth";
      RecordResidualSkip(skip, "metric_depth_disabled");
    }
  }

  problem_->SetParameterLowerBound(&scale, 0, 1e-5);
}

void GlobalPositioner::AddMetricDepthResidual(point3D_t point3D_id,
                                              const TrackElement& observation,
                                              bool is_lc_observation,
                                              Reconstruction& reconstruction) {
  if (!reconstruction.ExistsImage(observation.image_id)) {
    ResidualLedgerEntry skip;
    skip.residual_type = "metric_depth";
    skip.point3D_id = point3D_id;
    skip.image_id = observation.image_id;
    skip.point2D_idx = observation.point2D_idx;
    skip.is_lc_observation = is_lc_observation;
    RecordResidualSkip(skip, "missing_image");
    return;
  }
  const Image& image = reconstruction.Image(observation.image_id);
  const Camera& camera = reconstruction.Camera(image.CameraId());

  ResidualLedgerEntry residual;
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
    RecordResidualSkip(residual, "missing_depth_validity");
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
    RecordResidualSkip(residual, "invalid_depth_prior");
    return;
  }
  if (depth_sigma <= 1e-9) {
    RecordResidualSkip(residual, "invalid_depth_sigma");
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

  ceres::CostFunction* metric_depth_cost =
      MetricDepthError::Create(image.CamFromWorld().rotation(),
                               depth_prior,
                               depth_sigma,
                               CreateMetricDepthOptions(options_));

  if (metric_depth_cost == nullptr) {
    RecordResidualSkip(residual, "metric_depth_cost_create_failed");
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
      RecordResidualSkip(residual, "runtime_lc_depth_outlier_skipped");
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
  const std::string residual_id = RecordResidualBlock(residual);
  RecordReplayResidual(residual_id,
                       metric_depth_cost,
                       depth_loss,
                       {frame_centers_[image.FrameId()].data(),
                        point3D.xyz.data(),
                        &dmap_scales_[observation.image_id]});
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

  // If do not optimize the points, set the points to be constant
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
