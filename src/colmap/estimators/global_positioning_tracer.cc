#include "colmap/estimators/global_positioning_tracer.h"

#include "colmap/estimators/global_positioning_residual_evaluation.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/misc.h"

#include <algorithm>
#include <utility>

namespace colmap {
namespace {

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

std::vector<std::string> DeferredFixedParametersForReplay(
    const GlobalPositioningResidualDescriptor& residual) {
  if (residual.residual_type == "bata_ref_frame") {
    std::vector<std::string> missing = {"cam_from_point3D_dir"};
    if (residual.uses_keypoint_covariance) {
      missing.push_back("keypoint_covariance_left_sqrt_info");
    }
    return missing;
  }
  if (residual.residual_type == "bata_constant_rig") {
    return {"cam_from_point3D_dir", "cam_from_rig_dir"};
  }
  if (residual.residual_type == "bata_variable_rig") {
    return {"cam_from_point3D_dir", "world_from_rig_rot"};
  }
  if (residual.residual_type == "metric_depth") {
    return {"camera_rotation", "metric_depth_options"};
  }
  if (residual.residual_type == "scale_prior") {
    return {"scale_prior_target", "scale_prior_stddev"};
  }
  return {"unclassified_fixed_parameters"};
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

class GlobalPositioningTraceIterationCallback
    : public ceres::IterationCallback {
 public:
  GlobalPositioningTraceIterationCallback(
      GlobalPositioningTraceRecorder* recorder,
      const ceres::Problem* problem,
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
      const bool write_residual_values,
      const bool write_raw_jacobians)
      : recorder_(recorder),
        problem_(problem),
        reconstruction_(reconstruction),
        frame_centers_(frame_centers),
        scales_(scales),
        dmap_scales_(dmap_scales),
        cams_in_rig_(cams_in_rig),
        residual_replay_entries_(residual_replay_entries),
        snapshot_every_n_iterations_(snapshot_every_n_iterations),
        max_snapshotted_points_(max_snapshotted_points),
        write_parameter_snapshots_(write_parameter_snapshots),
        write_residual_values_(write_residual_values),
        write_raw_jacobians_(write_raw_jacobians) {}

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
        recorder_->WriteResidualValues(
            EvaluateGlobalPositioningResiduals({*problem_,
                                                summary.iteration,
                                                *residual_replay_entries_,
                                                write_raw_jacobians_}));
      }
    }
    return ceres::SOLVER_CONTINUE;
  }

 private:
  GlobalPositioningTraceRecorder* recorder_ = nullptr;
  const ceres::Problem* problem_ = nullptr;
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
  bool write_raw_jacobians_ = false;
};

}  // namespace

GlobalPositioningTracer::GlobalPositioningTracer(
    const GlobalPositioningTraceOptions& options)
    : options_(options) {
  if (options_.level != GlobalPositioningTraceLevel::kOff) {
    recorder_ = std::make_unique<GlobalPositioningTraceRecorder>(options_);
  }
}

bool GlobalPositioningTracer::Enabled() const { return recorder_ != nullptr; }

GlobalPositioningTracer::~GlobalPositioningTracer() {
  if (recorder_ != nullptr && !finished_) {
    recorder_->MarkFinished("aborted");
  }
}

bool GlobalPositioningTracer::ResidualLedgerEnabled() const {
  return recorder_ != nullptr && recorder_->IsResidualLedgerEnabled();
}

bool GlobalPositioningTracer::ParameterSnapshotsEnabled() const {
  return recorder_ != nullptr && recorder_->IsParameterSnapshotsEnabled();
}

bool GlobalPositioningTracer::ResidualValuesEnabled() const {
  return recorder_ != nullptr && recorder_->IsResidualValuesEnabled();
}

bool GlobalPositioningTracer::ResidualJacobiansEnabled() const {
  return recorder_ != nullptr && recorder_->IsResidualJacobiansEnabled();
}

void GlobalPositioningTracer::WriteEvent(std::string event_type,
                                         std::string stage,
                                         GlobalPositioningTraceAttrs attrs) {
  if (recorder_ == nullptr) {
    return;
  }

  GlobalPositioningTraceRecord record;
  record.event_type = std::move(event_type);
  record.stage = std::move(stage);
  record.attrs = std::move(attrs);
  recorder_->WriteEvent(std::move(record));
}

void GlobalPositioningTracer::MarkFinished() {
  if (recorder_ == nullptr || finished_) {
    return;
  }
  recorder_->MarkFinished("finished");
  finished_ = true;
}

void GlobalPositioningTracer::ResetProblemState() {
  residual_bucket_counts_.clear();
  scale_observations_.clear();
  residual_replay_entries_.clear();
}

void GlobalPositioningTracer::RecordScaleObservation(
    const size_t scale_index,
    const point3D_t point3D_id,
    const TrackElement& observation,
    const bool is_lc_observation) {
  if (!ResidualLedgerEnabled()) {
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

std::string GlobalPositioningTracer::RecordResidual(
    const GlobalPositioningResidualDescriptor& residual,
    const ceres::CostFunction* cost_function,
    const ceres::LossFunction* loss_function,
    std::vector<const double*> parameter_blocks,
    std::vector<GlobalPositioningTraceParameterBlockDescriptor>
        parameter_block_descriptors) {
  if (!ResidualLedgerEnabled()) {
    return "";
  }

  std::string residual_id = recorder_->AllocateResidualId();
  GlobalPositioningTraceRecord record =
      MakeResidualRecord(residual, "residual_added", "problem_build");
  record.attrs["residual_id"] = TraceValue::String(residual_id);
  record.attrs["replay_schema_version"] = TraceValue::Int(1);
  THROW_CHECK(cost_function != nullptr)
      << "Residual ledger tracing requires every recorded residual to have a "
         "cost function so parameter block descriptors can be serialized.";
  const std::vector<int>& parameter_block_sizes =
      cost_function->parameter_block_sizes();
  THROW_CHECK_EQ(parameter_blocks.size(), parameter_block_sizes.size())
      << "Residual ledger parameter-block count does not match the Ceres cost "
         "function.";
  THROW_CHECK_EQ(parameter_block_descriptors.size(),
                 parameter_block_sizes.size())
      << "Residual ledger parameter-block descriptor count does not match the "
         "Ceres cost function.";
  std::vector<GlobalPositioningTraceParameterBlockDescriptor>
      ledger_parameter_block_descriptors = parameter_block_descriptors;
  for (size_t block_idx = 0; block_idx < parameter_block_sizes.size();
       ++block_idx) {
    THROW_CHECK(parameter_blocks[block_idx] != nullptr)
        << "Residual ledger parameter block pointer is null.";
    THROW_CHECK(!ledger_parameter_block_descriptors[block_idx].role.empty())
        << "Residual ledger parameter block role must be non-empty.";
    THROW_CHECK(!ledger_parameter_block_descriptors[block_idx].kind.empty())
        << "Residual ledger parameter block kind must be non-empty.";
    THROW_CHECK_GT(parameter_block_sizes[block_idx], 0)
        << "Residual ledger parameter block size must be positive.";
    ledger_parameter_block_descriptors[block_idx].size =
        static_cast<size_t>(parameter_block_sizes[block_idx]);
  }
  record.attrs["parameter_blocks"] = TraceValue::ParameterBlockArray(
      std::move(ledger_parameter_block_descriptors));
  THROW_CHECK(residual.loss.has_value())
      << "Residual ledger tracing requires an explicit loss config for "
         "residual type "
      << residual.residual_type << " and loss bucket " << residual.loss_bucket
      << ".";
  record.attrs["loss"] = TraceValue::LossConfig(*residual.loss);
  if (residual.fixed_parameters.has_value()) {
    record.attrs["fixed_parameters_status"] = TraceValue::String("serialized");
    record.attrs["fixed_parameters"] =
        TraceValue::FixedParameters(*residual.fixed_parameters);
  } else {
    record.attrs["fixed_parameters_status"] =
        TraceValue::String("deferred_not_serialized");
    record.attrs["fixed_parameters_todo"] = TraceValue::String(
        "GP_REPLAY_FIXED_PARAMETERS_" + residual.residual_type);
    record.attrs["fixed_parameters_missing"] =
        TraceValue::StringArray(DeferredFixedParametersForReplay(residual));
  }
  recorder_->WriteResidualBlock(std::move(record));
  ++residual_bucket_counts_[residual.residual_type + "|" +
                            residual.loss_bucket];

  if (ResidualValuesEnabled()) {
    residual_replay_entries_.push_back(
        {residual_id,
         cost_function,
         loss_function,
         static_cast<size_t>(cost_function->num_residuals()),
         parameter_block_sizes,
         std::move(parameter_blocks),
         std::move(parameter_block_descriptors)});
  }
  return residual_id;
}

void GlobalPositioningTracer::RecordSkip(
    const GlobalPositioningResidualDescriptor& residual,
    std::string skip_reason) {
  if (!ResidualLedgerEnabled()) {
    return;
  }

  GlobalPositioningTraceRecord record =
      MakeResidualRecord(residual, "residual_skipped", "problem_build");
  record.attrs["skip_reason"] = TraceValue::String(std::move(skip_reason));
  recorder_->WriteResidualSkip(std::move(record));
}

void GlobalPositioningTracer::RecordBucketSummaries() {
  if (!ResidualLedgerEnabled()) {
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
    recorder_->WriteResidualBucketSummary(std::move(record));
  }
}

std::unique_ptr<ceres::IterationCallback>
GlobalPositioningTracer::CreateIterationCallback(
    GlobalPositioningTraceLiveState live_state) {
  if (!Enabled()) {
    return nullptr;
  }

  return std::make_unique<GlobalPositioningTraceIterationCallback>(
      recorder_.get(),
      &live_state.problem,
      &live_state.reconstruction,
      &live_state.frame_centers,
      &live_state.scales,
      &live_state.dmap_scales,
      &live_state.cams_in_rig,
      &residual_replay_entries_,
      options_.snapshot_every_n_iterations,
      options_.max_snapshotted_points,
      ParameterSnapshotsEnabled(),
      ResidualValuesEnabled(),
      ResidualJacobiansEnabled());
}

GlobalPositioningTraceRecord GlobalPositioningTracer::MakeResidualRecord(
    const GlobalPositioningResidualDescriptor& residual,
    std::string event_type,
    std::string stage) const {
  GlobalPositioningTraceRecord record;
  record.event_type = std::move(event_type);
  record.stage = std::move(stage);
  record.attrs = {
      {"residual_type", TraceValue::String(residual.residual_type)},
      {"point3D_id", TraceOptionalId(residual.point3D_id)},
      {"image_id", TraceOptionalId(residual.image_id)},
      {"point2D_idx", TraceOptionalId(residual.point2D_idx)},
      {"frame_id", TraceOptionalId(residual.frame_id)},
      {"camera_id", TraceOptionalId(residual.camera_id)},
      {"sensor_id", TraceOptionalId(TraceSensorId(residual.sensor_id))},
      {"is_lc_observation", TraceValue::Bool(residual.is_lc_observation)},
      {"is_ref_in_frame", TraceValue::Bool(residual.is_ref_in_frame)},
      {"camera_has_prior_focal_length",
       TraceValue::Bool(residual.camera_has_prior_focal_length)},
      {"loss_bucket", TraceValue::String(residual.loss_bucket)},
      {"uses_keypoint_covariance",
       TraceValue::Bool(residual.uses_keypoint_covariance)},
      {"has_depth_prior", TraceValue::Bool(residual.has_depth_prior)},
      {"depth_prior", TraceOptionalDouble(residual.depth_prior)},
      {"depth_sigma", TraceOptionalDouble(residual.depth_sigma)},
      {"dmap_scale_image_id", TraceOptionalId(residual.dmap_scale_image_id)},
      {"depth_outlier_source",
       TraceValue::String(residual.depth_outlier_source)},
  };
  return record;
}

}  // namespace colmap
