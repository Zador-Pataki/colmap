#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <ceres/ceres.h>

namespace colmap {

enum class GlobalPositioningTraceLevel {
  kOff = 0,
  kSummary = 1,
  kResidualLedger = 2,
  kParameterSnapshots = 3,
  kResidualValues = 4,
  kResidualJacobians = 5,
};

std::string GlobalPositioningTraceLevelToString(
    GlobalPositioningTraceLevel level);

struct GlobalPositioningTraceOptions {
  GlobalPositioningTraceLevel level = GlobalPositioningTraceLevel::kOff;
  std::filesystem::path output_path;
  std::string run_label;
  int snapshot_every_n_iterations = 1;
  bool write_legacy_jsonl = true;
};

struct GlobalPositioningTraceParameterBlockDescriptor {
  std::string role;
  std::string kind;
  uint64_t id = 0;
  std::optional<size_t> size;
};

struct GlobalPositioningTraceLossConfig {
  std::string bucket;
  std::string type;
  std::optional<double> scale;
  std::optional<double> weight;
  std::string source;
  std::optional<double> observation_count_weight;
};

struct GlobalPositioningTraceFixedParameters {
  std::optional<std::vector<double>> cam_from_point3D_dir;
  std::optional<std::vector<double>> keypoint_covariance_world_row_major;
  std::optional<std::vector<double>> cam_from_rig_dir;
  std::optional<std::vector<double>> rig_from_world_rotation_wxyz;
  std::optional<std::vector<double>> world_from_rig_rotation_wxyz;
  std::optional<std::vector<double>> camera_rotation_wxyz;
  std::optional<bool> metric_depth_use_log_scale;
  std::optional<std::string> metric_depth_residual_type;
  std::optional<bool> metric_depth_zero_residual_behind;
  std::optional<double> metric_depth_log_linear_threshold;
  std::optional<double> scale_prior_target;
  std::optional<double> scale_prior_stddev;
};

struct GlobalPositioningTraceValue {
  enum class Type {
    kNull,
    kBool,
    kInt,
    kUInt,
    kDouble,
    kString,
    kStringArray,
    kParameterBlockArray,
    kLossConfig,
    kFixedParameters,
  };

  static GlobalPositioningTraceValue Null();
  static GlobalPositioningTraceValue Bool(bool value);
  static GlobalPositioningTraceValue Int(int64_t value);
  static GlobalPositioningTraceValue UInt(uint64_t value);
  static GlobalPositioningTraceValue Double(double value);
  static GlobalPositioningTraceValue String(std::string value);
  static GlobalPositioningTraceValue StringArray(
      std::vector<std::string> value);
  static GlobalPositioningTraceValue ParameterBlockArray(
      std::vector<GlobalPositioningTraceParameterBlockDescriptor> value);
  static GlobalPositioningTraceValue LossConfig(
      GlobalPositioningTraceLossConfig value);
  static GlobalPositioningTraceValue FixedParameters(
      GlobalPositioningTraceFixedParameters value);

  Type type = Type::kNull;
  bool bool_value = false;
  int64_t int_value = 0;
  uint64_t uint_value = 0;
  double double_value = 0.0;
  std::string string_value;
  std::vector<std::string> string_array_value;
  std::vector<GlobalPositioningTraceParameterBlockDescriptor>
      parameter_block_array_value;
  GlobalPositioningTraceLossConfig loss_config_value;
  GlobalPositioningTraceFixedParameters fixed_parameters_value;
};

struct GlobalPositioningTraceRecord {
  std::string event_type;
  std::string stage;
  std::optional<int> iteration;
  std::map<std::string, GlobalPositioningTraceValue> attrs;
};

struct GlobalPositioningTraceSnapshotArray {
  std::vector<uint64_t> ids;
  std::vector<size_t> shape;
  std::vector<double> values;
};

struct GlobalPositioningTraceParameterSnapshot {
  int iteration = 0;
  GlobalPositioningTraceSnapshotArray frame_centers;
  GlobalPositioningTraceSnapshotArray points3D;
  GlobalPositioningTraceSnapshotArray scales;
  std::optional<GlobalPositioningTraceSnapshotArray> dmap_scales;
  std::optional<GlobalPositioningTraceSnapshotArray> cams_in_rig;
};

struct GlobalPositioningTraceResidualValues {
  int iteration = 0;
  std::vector<std::string> residual_ids;
  std::vector<size_t> residual_dims;
  std::vector<size_t> residual_offsets;
  std::vector<bool> evaluation_success;
  std::vector<double> raw_residuals;
  std::vector<double> raw_costs;
  std::vector<double> robust_costs;
  std::vector<double> loss_rho_values;
  bool has_raw_jacobians = false;
  std::vector<std::vector<size_t>> parameter_block_sizes;
  std::vector<std::vector<size_t>> raw_jacobian_offsets;
  std::vector<std::vector<GlobalPositioningTraceParameterBlockDescriptor>>
      parameter_blocks;
  std::vector<std::vector<bool>> parameter_block_is_constant;
  std::vector<std::vector<std::vector<double>>> parameter_block_lower_bounds;
  std::vector<double> raw_jacobians;
};

struct GlobalPositioningRawBinaryTraceIterationArtifacts {
  int iteration = 0;
  bool has_frame_centers = false;
  bool has_point_xyz = false;
  bool has_scales = false;
  bool has_dmap_scales = false;
  bool has_cams_in_rig = false;
  bool has_residual_values = false;
};

struct GlobalPositioningRawBinaryResidualPointIndexEntry {
  int64_t min_frame_id = 0;
  int64_t max_frame_id = 0;
  std::vector<uint64_t> residual_ledger_offsets;
};

class GlobalPositioningTraceRecorder {
 public:
  explicit GlobalPositioningTraceRecorder(
      const GlobalPositioningTraceOptions& options);

  const std::string& RunId() const { return run_id_; }
  bool IsResidualLedgerEnabled() const;
  bool IsParameterSnapshotsEnabled() const;
  bool IsResidualValuesEnabled() const;
  bool IsResidualJacobiansEnabled() const;
  std::string AllocateResidualId();

  void WriteEvent(GlobalPositioningTraceRecord record);
  void WriteIteration(const ceres::IterationSummary& summary);
  void WriteResidualBlock(GlobalPositioningTraceRecord record);
  void WriteResidualSkip(GlobalPositioningTraceRecord record);
  void WriteResidualBucketSummary(GlobalPositioningTraceRecord record);
  void WriteParameterSnapshot(
      const GlobalPositioningTraceParameterSnapshot& snapshot);
  void WriteResidualValues(
      const GlobalPositioningTraceResidualValues& residual_values);
  void MarkFinished(std::string status);

 private:
  void WriteRecord(std::ofstream& stream, GlobalPositioningTraceRecord record);
  void WriteManifest(const std::string& status);
  void WriteRawBinaryManifest(const std::string& status);
  void WriteRawBinaryResidualLedgerHeader();
  void UpdateRawBinaryResidualLedgerHeader();
  void WriteRawBinaryResidualBlock(const GlobalPositioningTraceRecord& record);
  void WriteRawBinaryResidualPointIndex();
  void WriteRawBinaryParameterSnapshot(
      const GlobalPositioningTraceParameterSnapshot& snapshot);
  void WriteRawBinaryResidualValues(
      const GlobalPositioningTraceResidualValues& residual_values);
  GlobalPositioningRawBinaryTraceIterationArtifacts&
  RawBinaryIterationArtifacts(int iteration);

  GlobalPositioningTraceOptions options_;
  std::string run_id_;
  int64_t created_at_unix_ns_ = 0;
  uint64_t sequence_ = 0;
  uint64_t residual_sequence_ = 0;
  uint64_t raw_binary_residual_ledger_count_ = 0;
  std::ofstream events_stream_;
  std::ofstream iteration_metrics_stream_;
  std::ofstream residual_blocks_stream_;
  std::ofstream residual_skips_stream_;
  std::ofstream raw_binary_residual_ledger_stream_;
  std::map<uint64_t, GlobalPositioningRawBinaryResidualPointIndexEntry>
      raw_binary_residual_point_index_;
  std::vector<GlobalPositioningRawBinaryTraceIterationArtifacts>
      raw_binary_iterations_;
};

}  // namespace colmap
