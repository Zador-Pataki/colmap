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
};

std::string GlobalPositioningTraceLevelToString(
    GlobalPositioningTraceLevel level);

struct GlobalPositioningTraceOptions {
  GlobalPositioningTraceLevel level = GlobalPositioningTraceLevel::kOff;
  std::filesystem::path output_path;
  std::string run_label;
  int snapshot_every_n_iterations = 1;
  int max_snapshotted_points = -1;
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
  };

  static GlobalPositioningTraceValue Null();
  static GlobalPositioningTraceValue Bool(bool value);
  static GlobalPositioningTraceValue Int(int64_t value);
  static GlobalPositioningTraceValue UInt(uint64_t value);
  static GlobalPositioningTraceValue Double(double value);
  static GlobalPositioningTraceValue String(std::string value);
  static GlobalPositioningTraceValue StringArray(
      std::vector<std::string> value);

  Type type = Type::kNull;
  bool bool_value = false;
  int64_t int_value = 0;
  uint64_t uint_value = 0;
  double double_value = 0.0;
  std::string string_value;
  std::vector<std::string> string_array_value;
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

class GlobalPositioningTraceRecorder {
 public:
  explicit GlobalPositioningTraceRecorder(
      const GlobalPositioningTraceOptions& options);

  const std::string& RunId() const { return run_id_; }
  bool IsResidualLedgerEnabled() const;
  bool IsParameterSnapshotsEnabled() const;
  std::string AllocateResidualId();

  void WriteEvent(GlobalPositioningTraceRecord record);
  void WriteIteration(const ceres::IterationSummary& summary);
  void WriteResidualBlock(GlobalPositioningTraceRecord record);
  void WriteResidualSkip(GlobalPositioningTraceRecord record);
  void WriteResidualBucketSummary(GlobalPositioningTraceRecord record);
  void WriteParameterSnapshot(
      const GlobalPositioningTraceParameterSnapshot& snapshot);
  void MarkFinished(std::string status);

 private:
  void WriteRecord(std::ofstream& stream, GlobalPositioningTraceRecord record);
  void WriteManifest(const std::string& status);

  GlobalPositioningTraceOptions options_;
  std::string run_id_;
  int64_t created_at_unix_ns_ = 0;
  uint64_t sequence_ = 0;
  uint64_t residual_sequence_ = 0;
  std::ofstream events_stream_;
  std::ofstream iteration_metrics_stream_;
  std::ofstream residual_blocks_stream_;
  std::ofstream residual_skips_stream_;
};

}  // namespace colmap
