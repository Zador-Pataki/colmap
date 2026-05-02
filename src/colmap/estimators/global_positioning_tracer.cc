#include "colmap/estimators/global_positioning_tracer.h"

#include "colmap/scene/reconstruction.h"
#include "colmap/util/misc.h"

#include <algorithm>
#include <limits>
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
    const ceres::Problem& problem,
    const int iteration,
    const std::vector<GlobalPositioningResidualReplayEntry>& replay_entries,
    const bool write_raw_jacobians) {
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
  residual_values.loss_rho_values.resize(
      replay_entries.size() * 3, std::numeric_limits<double>::quiet_NaN());
  residual_values.has_raw_jacobians = write_raw_jacobians;
  if (write_raw_jacobians) {
    residual_values.parameter_block_sizes.reserve(replay_entries.size());
    residual_values.raw_jacobian_offsets.reserve(replay_entries.size());
    residual_values.parameter_blocks.reserve(replay_entries.size());
    residual_values.parameter_block_is_constant.reserve(replay_entries.size());
    residual_values.parameter_block_lower_bounds.reserve(replay_entries.size());
  }

  size_t total_scalar_residuals = 0;
  size_t total_jacobian_scalars = 0;
  for (const GlobalPositioningResidualReplayEntry& entry : replay_entries) {
    THROW_CHECK(!entry.residual_id.empty())
        << "Residual replay entry has an empty residual id.";
    THROW_CHECK(entry.cost_function != nullptr)
        << "Residual replay entry " << entry.residual_id
        << " has a null cost function.";
    THROW_CHECK_EQ(entry.residual_dimension,
                   static_cast<size_t>(entry.cost_function->num_residuals()))
        << "Residual replay entry " << entry.residual_id
        << " residual dimension does not match the Ceres cost function.";
    const std::vector<int>& cost_function_parameter_block_sizes =
        entry.cost_function->parameter_block_sizes();
    THROW_CHECK_EQ(entry.parameter_block_sizes.size(),
                   cost_function_parameter_block_sizes.size())
        << "Residual replay entry " << entry.residual_id
        << " parameter-block size count does not match the Ceres cost "
           "function.";
    THROW_CHECK_EQ(entry.parameter_blocks.size(),
                   entry.parameter_block_sizes.size())
        << "Residual replay entry " << entry.residual_id
        << " parameter-block pointer count does not match the stored block "
           "sizes.";
    THROW_CHECK_EQ(entry.parameter_block_descriptors.size(),
                   entry.parameter_block_sizes.size())
        << "Residual replay entry " << entry.residual_id
        << " parameter-block descriptor count does not match the stored block "
           "sizes.";
    for (size_t block_idx = 0; block_idx < entry.parameter_blocks.size();
         ++block_idx) {
      THROW_CHECK(entry.parameter_blocks[block_idx] != nullptr)
          << "Residual replay entry " << entry.residual_id
          << " has a null parameter block pointer at index " << block_idx
          << ".";
      THROW_CHECK_EQ(entry.parameter_block_sizes[block_idx],
                     cost_function_parameter_block_sizes[block_idx])
          << "Residual replay entry " << entry.residual_id
          << " parameter block size at index " << block_idx
          << " does not match the Ceres cost function.";
      THROW_CHECK_GT(entry.parameter_block_sizes[block_idx], 0)
          << "Residual replay entry " << entry.residual_id
          << " parameter block size at index " << block_idx
          << " must be positive.";
      THROW_CHECK(!entry.parameter_block_descriptors[block_idx].role.empty())
          << "Residual replay entry " << entry.residual_id
          << " parameter block descriptor at index " << block_idx
          << " has an empty role.";
      THROW_CHECK(!entry.parameter_block_descriptors[block_idx].kind.empty())
          << "Residual replay entry " << entry.residual_id
          << " parameter block descriptor at index " << block_idx
          << " has an empty kind.";
    }

    residual_values.residual_ids.push_back(entry.residual_id);
    residual_values.residual_dims.push_back(entry.residual_dimension);
    residual_values.residual_offsets.push_back(total_scalar_residuals);
    total_scalar_residuals += entry.residual_dimension;
    if (write_raw_jacobians) {
      std::vector<size_t> parameter_block_sizes;
      std::vector<size_t> raw_jacobian_offsets;
      std::vector<bool> parameter_block_is_constant;
      std::vector<std::vector<double>> parameter_block_lower_bounds;
      parameter_block_sizes.reserve(entry.parameter_block_sizes.size());
      raw_jacobian_offsets.reserve(entry.parameter_block_sizes.size());
      parameter_block_is_constant.reserve(entry.parameter_block_sizes.size());
      parameter_block_lower_bounds.reserve(entry.parameter_block_sizes.size());
      for (size_t block_idx = 0; block_idx < entry.parameter_block_sizes.size();
           ++block_idx) {
        const int parameter_block_size = entry.parameter_block_sizes[block_idx];
        parameter_block_sizes.push_back(
            static_cast<size_t>(parameter_block_size));
        raw_jacobian_offsets.push_back(total_jacobian_scalars);
        parameter_block_is_constant.push_back(problem.IsParameterBlockConstant(
            entry.parameter_blocks[block_idx]));
        std::vector<double> lower_bounds;
        lower_bounds.reserve(static_cast<size_t>(parameter_block_size));
        for (int parameter_idx = 0; parameter_idx < parameter_block_size;
             ++parameter_idx) {
          lower_bounds.push_back(problem.GetParameterLowerBound(
              entry.parameter_blocks[block_idx], parameter_idx));
        }
        parameter_block_lower_bounds.push_back(std::move(lower_bounds));
        total_jacobian_scalars += entry.residual_dimension *
                                  static_cast<size_t>(parameter_block_size);
      }
      residual_values.parameter_block_sizes.push_back(
          std::move(parameter_block_sizes));
      residual_values.raw_jacobian_offsets.push_back(
          std::move(raw_jacobian_offsets));
      residual_values.parameter_blocks.push_back(
          entry.parameter_block_descriptors);
      residual_values.parameter_block_is_constant.push_back(
          std::move(parameter_block_is_constant));
      residual_values.parameter_block_lower_bounds.push_back(
          std::move(parameter_block_lower_bounds));
    }
  }
  residual_values.raw_residuals.resize(
      total_scalar_residuals, std::numeric_limits<double>::quiet_NaN());
  if (write_raw_jacobians) {
    residual_values.raw_jacobians.resize(
        total_jacobian_scalars, std::numeric_limits<double>::quiet_NaN());
  }

  for (size_t entry_idx = 0; entry_idx < replay_entries.size(); ++entry_idx) {
    const GlobalPositioningResidualReplayEntry& entry =
        replay_entries[entry_idx];
    std::vector<double> raw_jacobian_workspace;
    std::vector<double*> raw_jacobian_blocks;
    if (write_raw_jacobians) {
      size_t workspace_offset = 0;
      for (const int parameter_block_size : entry.parameter_block_sizes) {
        workspace_offset += entry.residual_dimension *
                            static_cast<size_t>(parameter_block_size);
      }
      raw_jacobian_workspace.assign(workspace_offset,
                                    std::numeric_limits<double>::quiet_NaN());
      raw_jacobian_blocks.reserve(entry.parameter_block_sizes.size());
      workspace_offset = 0;
      for (const int parameter_block_size : entry.parameter_block_sizes) {
        raw_jacobian_blocks.push_back(raw_jacobian_workspace.data() +
                                      workspace_offset);
        workspace_offset += entry.residual_dimension *
                            static_cast<size_t>(parameter_block_size);
      }
    }

    double* raw_residuals = residual_values.raw_residuals.data() +
                            residual_values.residual_offsets[entry_idx];
    const bool evaluation_success = entry.cost_function->Evaluate(
        entry.parameter_blocks.data(),
        raw_residuals,
        write_raw_jacobians ? raw_jacobian_blocks.data() : nullptr);
    residual_values.evaluation_success[entry_idx] = evaluation_success;
    if (!evaluation_success) {
      continue;
    }
    if (write_raw_jacobians) {
      for (size_t block_idx = 0; block_idx < entry.parameter_block_sizes.size();
           ++block_idx) {
        const size_t jacobian_size =
            entry.residual_dimension *
            static_cast<size_t>(entry.parameter_block_sizes[block_idx]);
        std::copy_n(
            raw_jacobian_blocks[block_idx],
            jacobian_size,
            residual_values.raw_jacobians.data() +
                residual_values.raw_jacobian_offsets[entry_idx][block_idx]);
      }
    }

    double squared_norm = 0.0;
    for (size_t residual_idx = 0; residual_idx < entry.residual_dimension;
         ++residual_idx) {
      squared_norm += raw_residuals[residual_idx] * raw_residuals[residual_idx];
    }

    const double raw_cost = 0.5 * squared_norm;
    residual_values.raw_costs[entry_idx] = raw_cost;
    double rho[3] = {squared_norm, 1.0, 0.0};
    if (entry.loss_function != nullptr) {
      entry.loss_function->Evaluate(squared_norm, rho);
    }
    residual_values.loss_rho_values[3 * entry_idx] = rho[0];
    residual_values.loss_rho_values[3 * entry_idx + 1] = rho[1];
    residual_values.loss_rho_values[3 * entry_idx + 2] = rho[2];
    residual_values.robust_costs[entry_idx] = 0.5 * rho[0];
  }

  recorder->WriteResidualValues(residual_values);
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
        WriteResidualValues(recorder_,
                            *problem_,
                            summary.iteration,
                            *residual_replay_entries_,
                            write_raw_jacobians_);
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
  recorder_->WriteResidualBlock(std::move(record));
  ++residual_bucket_counts_[residual.residual_type + "|" +
                            residual.loss_bucket];

  if (ResidualValuesEnabled()) {
    THROW_CHECK(cost_function != nullptr)
        << "Residual-values tracing requires every recorded residual to have a "
           "cost function.";
    const std::vector<int>& parameter_block_sizes =
        cost_function->parameter_block_sizes();
    THROW_CHECK_EQ(parameter_blocks.size(), parameter_block_sizes.size())
        << "Residual replay parameter-block count does not match the Ceres "
           "cost function.";
    THROW_CHECK_EQ(parameter_block_descriptors.size(),
                   parameter_block_sizes.size())
        << "Residual replay parameter-block descriptor count does not match "
           "the Ceres cost function.";
    for (const double* parameter_block : parameter_blocks) {
      THROW_CHECK(parameter_block != nullptr)
          << "Residual replay parameter block pointer is null.";
    }
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
