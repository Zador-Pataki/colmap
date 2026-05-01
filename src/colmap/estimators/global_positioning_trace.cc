#include "colmap/estimators/global_positioning_trace.h"

#include "colmap/util/endian.h"
#include "colmap/util/file.h"
#include "colmap/util/logging.h"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <system_error>
#include <utility>

#include <ceres/types.h>

namespace colmap {
namespace {

constexpr int kSchemaVersion = 1;

int64_t UnixNowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

int64_t SteadyNowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

std::string JsonEscape(const std::string& value) {
  std::ostringstream stream;
  stream << '"';
  for (const char c : value) {
    switch (c) {
      case '"':
        stream << "\\\"";
        break;
      case '\\':
        stream << "\\\\";
        break;
      case '\b':
        stream << "\\b";
        break;
      case '\f':
        stream << "\\f";
        break;
      case '\n':
        stream << "\\n";
        break;
      case '\r':
        stream << "\\r";
        break;
      case '\t':
        stream << "\\t";
        break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          stream << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                 << static_cast<int>(static_cast<unsigned char>(c)) << std::dec
                 << std::setfill(' ');
        } else {
          stream << c;
        }
    }
  }
  stream << '"';
  return stream.str();
}

std::string JsonDouble(const double value) {
  if (!std::isfinite(value)) {
    if (std::isnan(value)) {
      return JsonEscape("nan");
    }
    return JsonEscape(value < 0 ? "-inf" : "inf");
  }
  std::ostringstream stream;
  stream << std::setprecision(std::numeric_limits<double>::max_digits10)
         << value;
  return stream.str();
}

std::string JsonValue(const GlobalPositioningTraceValue& value) {
  switch (value.type) {
    case GlobalPositioningTraceValue::Type::kNull:
      return "null";
    case GlobalPositioningTraceValue::Type::kBool:
      return value.bool_value ? "true" : "false";
    case GlobalPositioningTraceValue::Type::kInt:
      return std::to_string(value.int_value);
    case GlobalPositioningTraceValue::Type::kUInt:
      return std::to_string(value.uint_value);
    case GlobalPositioningTraceValue::Type::kDouble:
      return JsonDouble(value.double_value);
    case GlobalPositioningTraceValue::Type::kString:
      return JsonEscape(value.string_value);
    case GlobalPositioningTraceValue::Type::kStringArray: {
      std::ostringstream stream;
      stream << "[";
      for (size_t i = 0; i < value.string_array_value.size(); ++i) {
        if (i > 0) {
          stream << ",";
        }
        stream << JsonEscape(value.string_array_value[i]);
      }
      stream << "]";
      return stream.str();
    }
  }
  return "null";
}

std::string JsonAttrs(
    const std::map<std::string, GlobalPositioningTraceValue>& attrs) {
  std::ostringstream stream;
  stream << "{";
  bool first = true;
  for (const auto& [key, value] : attrs) {
    if (!first) {
      stream << ",";
    }
    first = false;
    stream << JsonEscape(key) << ":" << JsonValue(value);
  }
  stream << "}";
  return stream.str();
}

std::string MakeRunId(const int64_t created_at_unix_ns,
                      const std::string& run_label) {
  std::ostringstream stream;
  stream << created_at_unix_ns;
  if (!run_label.empty()) {
    stream << "_" << run_label;
  }
  return stream.str();
}

bool IsResidualLedgerLevel(const GlobalPositioningTraceLevel level) {
  return static_cast<int>(level) >=
         static_cast<int>(GlobalPositioningTraceLevel::kResidualLedger);
}

bool IsParameterSnapshotsLevel(const GlobalPositioningTraceLevel level) {
  return static_cast<int>(level) >=
         static_cast<int>(GlobalPositioningTraceLevel::kParameterSnapshots);
}

std::string IterationPrefix(const int iteration) {
  std::ostringstream stream;
  stream << "iter_" << std::setw(6) << std::setfill('0') << iteration;
  return stream.str();
}

uint64_t ShapeElementCount(const std::vector<size_t>& shape) {
  uint64_t count = 1;
  for (const size_t dim : shape) {
    count *= static_cast<uint64_t>(dim);
  }
  return count;
}

std::string JsonUInt64Array(const std::vector<uint64_t>& values) {
  std::ostringstream stream;
  stream << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      stream << ",";
    }
    stream << values[i];
  }
  stream << "]";
  return stream.str();
}

std::string JsonSizeArray(const std::vector<size_t>& values) {
  std::ostringstream stream;
  stream << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      stream << ",";
    }
    stream << values[i];
  }
  stream << "]";
  return stream.str();
}

void ValidateSnapshotArray(const std::string& name,
                           const GlobalPositioningTraceSnapshotArray& array) {
  THROW_CHECK(!array.shape.empty())
      << "Snapshot array " << name << " must declare a non-empty shape.";
  THROW_CHECK_EQ(ShapeElementCount(array.shape), array.values.size())
      << "Snapshot array " << name
      << " shape does not match flattened value count.";
  THROW_CHECK_EQ(array.shape.front(), array.ids.size())
      << "Snapshot array " << name
      << " first shape dimension must match the number of IDs.";
}

void WriteSnapshotSidecar(const std::filesystem::path& path,
                          const GlobalPositioningTraceSnapshotArray& array) {
  std::ofstream sidecar_stream(
      path, std::ios::binary | std::ios::out | std::ios::trunc);
  THROW_CHECK_FILE_OPEN(sidecar_stream, path);
  for (const double value : array.values) {
    WriteBinaryLittleEndian<double>(&sidecar_stream, value);
  }
  THROW_CHECK(sidecar_stream.good())
      << "Failed while writing global positioning snapshot sidecar: " << path;
}

void WriteSnapshotArtifactMetadata(
    std::ostream& stream,
    const std::string& name,
    const GlobalPositioningTraceSnapshotArray& array,
    const std::string& filename) {
  stream << "    " << JsonEscape(name) << ": {\n"
         << "      \"file\": " << JsonEscape(filename) << ",\n"
         << "      \"dtype\": \"float64\",\n"
         << "      \"byte_order\": \"little_endian\",\n"
         << "      \"ids\": " << JsonUInt64Array(array.ids) << ",\n"
         << "      \"shape\": " << JsonSizeArray(array.shape) << "\n"
         << "    }";
}

void RemoveTraceArtifactIfExists(const std::filesystem::path& path) {
  std::error_code error;
  std::filesystem::remove_all(path, error);
  THROW_CHECK(!error) << "Failed to remove stale global positioning trace "
                         "artifact: "
                      << path << " error: " << error.message();
}

}  // namespace

std::string GlobalPositioningTraceLevelToString(
    const GlobalPositioningTraceLevel level) {
  switch (level) {
    case GlobalPositioningTraceLevel::kOff:
      return "off";
    case GlobalPositioningTraceLevel::kSummary:
      return "summary";
    case GlobalPositioningTraceLevel::kResidualLedger:
      return "residual_ledger";
    case GlobalPositioningTraceLevel::kParameterSnapshots:
      return "parameter_snapshots";
    case GlobalPositioningTraceLevel::kResidualValues:
      return "residual_values";
  }
  return "unknown";
}

GlobalPositioningTraceValue GlobalPositioningTraceValue::Null() {
  return GlobalPositioningTraceValue{};
}

GlobalPositioningTraceValue GlobalPositioningTraceValue::Bool(
    const bool value) {
  GlobalPositioningTraceValue trace_value;
  trace_value.type = Type::kBool;
  trace_value.bool_value = value;
  return trace_value;
}

GlobalPositioningTraceValue GlobalPositioningTraceValue::Int(
    const int64_t value) {
  GlobalPositioningTraceValue trace_value;
  trace_value.type = Type::kInt;
  trace_value.int_value = value;
  return trace_value;
}

GlobalPositioningTraceValue GlobalPositioningTraceValue::UInt(
    const uint64_t value) {
  GlobalPositioningTraceValue trace_value;
  trace_value.type = Type::kUInt;
  trace_value.uint_value = value;
  return trace_value;
}

GlobalPositioningTraceValue GlobalPositioningTraceValue::Double(
    const double value) {
  GlobalPositioningTraceValue trace_value;
  trace_value.type = Type::kDouble;
  trace_value.double_value = value;
  return trace_value;
}

GlobalPositioningTraceValue GlobalPositioningTraceValue::String(
    std::string value) {
  GlobalPositioningTraceValue trace_value;
  trace_value.type = Type::kString;
  trace_value.string_value = std::move(value);
  return trace_value;
}

GlobalPositioningTraceValue GlobalPositioningTraceValue::StringArray(
    std::vector<std::string> value) {
  GlobalPositioningTraceValue trace_value;
  trace_value.type = Type::kStringArray;
  trace_value.string_array_value = std::move(value);
  return trace_value;
}

GlobalPositioningTraceRecorder::GlobalPositioningTraceRecorder(
    const GlobalPositioningTraceOptions& options)
    : options_(options), created_at_unix_ns_(UnixNowNs()) {
  THROW_CHECK(options_.level != GlobalPositioningTraceLevel::kOff);
  THROW_CHECK(!options_.output_path.empty())
      << "Global positioning trace output_path must be set when tracing is "
         "enabled.";
  THROW_CHECK(!ExistsFile(options_.output_path))
      << "Global positioning trace output_path points to a file: "
      << options_.output_path;
  if (!ExistsDir(options_.output_path)) {
    CreateDirIfNotExists(options_.output_path, /*recursive=*/true);
  }
  THROW_CHECK(ExistsDir(options_.output_path))
      << "Global positioning trace output_path is not a directory: "
      << options_.output_path;

  RemoveTraceArtifactIfExists(options_.output_path / "manifest.json");
  RemoveTraceArtifactIfExists(options_.output_path / "events.jsonl");
  RemoveTraceArtifactIfExists(options_.output_path / "iteration_metrics.jsonl");
  RemoveTraceArtifactIfExists(options_.output_path / "residual_blocks.jsonl");
  RemoveTraceArtifactIfExists(options_.output_path / "residual_skips.jsonl");
  RemoveTraceArtifactIfExists(options_.output_path / "snapshots");

  run_id_ = MakeRunId(created_at_unix_ns_, options_.run_label);

  events_stream_.open(options_.output_path / "events.jsonl",
                      std::ios::out | std::ios::trunc);
  THROW_CHECK_FILE_OPEN(events_stream_, options_.output_path / "events.jsonl");
  iteration_metrics_stream_.open(
      options_.output_path / "iteration_metrics.jsonl",
      std::ios::out | std::ios::trunc);
  THROW_CHECK_FILE_OPEN(iteration_metrics_stream_,
                        options_.output_path / "iteration_metrics.jsonl");

  if (IsResidualLedgerEnabled()) {
    residual_blocks_stream_.open(options_.output_path / "residual_blocks.jsonl",
                                 std::ios::out | std::ios::trunc);
    THROW_CHECK_FILE_OPEN(residual_blocks_stream_,
                          options_.output_path / "residual_blocks.jsonl");
    residual_skips_stream_.open(options_.output_path / "residual_skips.jsonl",
                                std::ios::out | std::ios::trunc);
    THROW_CHECK_FILE_OPEN(residual_skips_stream_,
                          options_.output_path / "residual_skips.jsonl");
  }

  if (IsParameterSnapshotsEnabled()) {
    const std::filesystem::path snapshot_path =
        options_.output_path / "snapshots";
    THROW_CHECK(!ExistsFile(snapshot_path))
        << "Global positioning snapshot path points to a file: "
        << snapshot_path;
    if (!ExistsDir(snapshot_path)) {
      CreateDirIfNotExists(snapshot_path, /*recursive=*/true);
    }
    THROW_CHECK(ExistsDir(snapshot_path))
        << "Global positioning snapshot path is not a directory: "
        << snapshot_path;
  }

  WriteManifest("running");
}

bool GlobalPositioningTraceRecorder::IsResidualLedgerEnabled() const {
  return IsResidualLedgerLevel(options_.level);
}

bool GlobalPositioningTraceRecorder::IsParameterSnapshotsEnabled() const {
  return IsParameterSnapshotsLevel(options_.level);
}

std::string GlobalPositioningTraceRecorder::AllocateResidualId() {
  std::ostringstream stream;
  stream << "r" << std::setw(10) << std::setfill('0') << residual_sequence_++;
  return stream.str();
}

void GlobalPositioningTraceRecorder::WriteEvent(
    GlobalPositioningTraceRecord record) {
  WriteRecord(events_stream_, std::move(record));
}

void GlobalPositioningTraceRecorder::WriteIteration(
    const ceres::IterationSummary& summary) {
  GlobalPositioningTraceRecord record;
  record.event_type = "ceres_iteration";
  record.stage = "ceres_solve";
  record.iteration = summary.iteration;
  record.attrs = {
      {"step_is_successful",
       GlobalPositioningTraceValue::Bool(summary.step_is_successful)},
      {"cost", GlobalPositioningTraceValue::Double(summary.cost)},
      {"cost_change", GlobalPositioningTraceValue::Double(summary.cost_change)},
      {"gradient_max_norm",
       GlobalPositioningTraceValue::Double(summary.gradient_max_norm)},
      {"step_norm", GlobalPositioningTraceValue::Double(summary.step_norm)},
      {"trust_region_radius",
       GlobalPositioningTraceValue::Double(summary.trust_region_radius)},
      {"linear_solver_iterations",
       GlobalPositioningTraceValue::Int(summary.linear_solver_iterations)},
      {"iteration_time_sec",
       GlobalPositioningTraceValue::Double(summary.iteration_time_in_seconds)},
      {"cumulative_time_sec",
       GlobalPositioningTraceValue::Double(summary.cumulative_time_in_seconds)},
  };
  WriteRecord(iteration_metrics_stream_, std::move(record));
}

void GlobalPositioningTraceRecorder::WriteResidualBlock(
    GlobalPositioningTraceRecord record) {
  if (!IsResidualLedgerEnabled()) {
    return;
  }
  if (record.event_type.empty()) {
    record.event_type = "residual_added";
  }
  if (record.stage.empty()) {
    record.stage = "problem_build";
  }
  WriteRecord(residual_blocks_stream_, std::move(record));
}

void GlobalPositioningTraceRecorder::WriteResidualSkip(
    GlobalPositioningTraceRecord record) {
  if (!IsResidualLedgerEnabled()) {
    return;
  }
  if (record.event_type.empty()) {
    record.event_type = "residual_skipped";
  }
  if (record.stage.empty()) {
    record.stage = "problem_build";
  }
  WriteRecord(residual_skips_stream_, std::move(record));
}

void GlobalPositioningTraceRecorder::WriteResidualBucketSummary(
    GlobalPositioningTraceRecord record) {
  if (!IsResidualLedgerEnabled()) {
    return;
  }
  if (record.event_type.empty()) {
    record.event_type = "residual_bucket_summary";
  }
  if (record.stage.empty()) {
    record.stage = "problem_build";
  }
  WriteEvent(std::move(record));
}

void GlobalPositioningTraceRecorder::WriteParameterSnapshot(
    const GlobalPositioningTraceParameterSnapshot& snapshot) {
  if (!IsParameterSnapshotsEnabled()) {
    return;
  }

  THROW_CHECK_GE(snapshot.iteration, 0)
      << "Global positioning snapshot iteration must be non-negative.";
  THROW_CHECK_EQ(sizeof(double), 8)
      << "Global positioning snapshots require 64-bit doubles.";
  THROW_CHECK(std::numeric_limits<double>::is_iec559)
      << "Global positioning snapshots require IEEE-754 doubles.";
  THROW_CHECK(IsLittleEndian())
      << "Global positioning snapshot binaries are defined as little-endian.";

  ValidateSnapshotArray("frame_centers", snapshot.frame_centers);
  ValidateSnapshotArray("points3D", snapshot.points3D);
  ValidateSnapshotArray("scales", snapshot.scales);
  if (snapshot.dmap_scales.has_value()) {
    ValidateSnapshotArray("dmap_scales", *snapshot.dmap_scales);
  }
  if (snapshot.cams_in_rig.has_value()) {
    ValidateSnapshotArray("cams_in_rig", *snapshot.cams_in_rig);
  }

  const std::filesystem::path snapshot_path =
      options_.output_path / "snapshots";
  const std::string prefix = IterationPrefix(snapshot.iteration);
  const std::filesystem::path metadata_path =
      snapshot_path / (prefix + ".json");
  const std::string frame_centers_filename = prefix + "_frame_centers_f64.bin";
  const std::string points3D_filename = prefix + "_points3D_f64.bin";
  const std::string scales_filename = prefix + "_scales_f64.bin";

  WriteSnapshotSidecar(snapshot_path / frame_centers_filename,
                       snapshot.frame_centers);
  WriteSnapshotSidecar(snapshot_path / points3D_filename, snapshot.points3D);
  WriteSnapshotSidecar(snapshot_path / scales_filename, snapshot.scales);

  std::optional<std::string> dmap_scales_filename;
  if (snapshot.dmap_scales.has_value()) {
    dmap_scales_filename = prefix + "_dmap_scales_f64.bin";
    WriteSnapshotSidecar(snapshot_path / *dmap_scales_filename,
                         *snapshot.dmap_scales);
  }

  std::optional<std::string> cams_in_rig_filename;
  if (snapshot.cams_in_rig.has_value()) {
    cams_in_rig_filename = prefix + "_cams_in_rig_f64.bin";
    WriteSnapshotSidecar(snapshot_path / *cams_in_rig_filename,
                         *snapshot.cams_in_rig);
  }

  std::ofstream metadata_stream(metadata_path, std::ios::out | std::ios::trunc);
  THROW_CHECK_FILE_OPEN(metadata_stream, metadata_path);
  metadata_stream << "{\n"
                  << "  \"schema_version\": " << kSchemaVersion << ",\n"
                  << "  \"run_id\": " << JsonEscape(run_id_) << ",\n"
                  << "  \"iteration\": " << snapshot.iteration << ",\n"
                  << "  \"dtype\": \"float64\",\n"
                  << "  \"byte_order\": \"little_endian\",\n"
                  << "  \"coordinate_convention\": "
                  << JsonEscape(
                         "frame_centers are world-frame camera centers, not "
                         "cam_from_world.translation")
                  << ",\n"
                  << "  \"frame_ids\": "
                  << JsonUInt64Array(snapshot.frame_centers.ids) << ",\n"
                  << "  \"frame_centers_world_shape\": "
                  << JsonSizeArray(snapshot.frame_centers.shape) << ",\n"
                  << "  \"point3D_ids\": "
                  << JsonUInt64Array(snapshot.points3D.ids) << ",\n"
                  << "  \"points3D_world_shape\": "
                  << JsonSizeArray(snapshot.points3D.shape) << ",\n"
                  << "  \"bata_residual_ids\": [],\n"
                  << "  \"bata_scale_ids\": "
                  << JsonUInt64Array(snapshot.scales.ids) << ",\n"
                  << "  \"bata_scales_shape\": "
                  << JsonSizeArray(snapshot.scales.shape) << ",\n"
                  << "  \"dmap_image_ids\": "
                  << JsonUInt64Array(snapshot.dmap_scales.has_value()
                                         ? snapshot.dmap_scales->ids
                                         : std::vector<uint64_t>{})
                  << ",\n"
                  << "  \"dmap_scales_stored_shape\": "
                  << JsonSizeArray(snapshot.dmap_scales.has_value()
                                       ? snapshot.dmap_scales->shape
                                       : std::vector<size_t>{0})
                  << ",\n"
                  << "  \"coordinate_conventions\": {\n"
                  << "    \"frame_centers\": "
                  << JsonEscape(
                         "world-frame camera centers; not "
                         "cam_from_world.translation")
                  << ",\n"
                  << "    \"points3D\": " << JsonEscape("world-frame 3D points")
                  << ",\n"
                  << "    \"cams_in_rig\": "
                  << JsonEscape("camera offsets in rig frame") << "\n"
                  << "  },\n"
                  << "  \"artifacts\": {\n";
  WriteSnapshotArtifactMetadata(metadata_stream,
                                "frame_centers",
                                snapshot.frame_centers,
                                frame_centers_filename);
  metadata_stream << ",\n";
  WriteSnapshotArtifactMetadata(
      metadata_stream, "points3D", snapshot.points3D, points3D_filename);
  metadata_stream << ",\n";
  WriteSnapshotArtifactMetadata(
      metadata_stream, "scales", snapshot.scales, scales_filename);
  if (snapshot.dmap_scales.has_value()) {
    metadata_stream << ",\n";
    WriteSnapshotArtifactMetadata(metadata_stream,
                                  "dmap_scales",
                                  *snapshot.dmap_scales,
                                  *dmap_scales_filename);
  }
  if (snapshot.cams_in_rig.has_value()) {
    metadata_stream << ",\n";
    WriteSnapshotArtifactMetadata(metadata_stream,
                                  "cams_in_rig",
                                  *snapshot.cams_in_rig,
                                  *cams_in_rig_filename);
  }
  metadata_stream << "\n"
                  << "  }\n"
                  << "}\n";
  metadata_stream.flush();
}

void GlobalPositioningTraceRecorder::MarkFinished(std::string status) {
  WriteManifest(status);
}

void GlobalPositioningTraceRecorder::WriteRecord(
    std::ofstream& stream, GlobalPositioningTraceRecord record) {
  stream << "{\"schema_version\":" << kSchemaVersion
         << ",\"run_id\":" << JsonEscape(run_id_) << ",\"seq\":" << sequence_++
         << ",\"event_type\":" << JsonEscape(record.event_type)
         << ",\"stage\":" << JsonEscape(record.stage) << ",\"iteration\":";
  if (record.iteration.has_value()) {
    stream << *record.iteration;
  } else {
    stream << "null";
  }
  stream << ",\"timestamp_ns\":" << SteadyNowNs()
         << ",\"attrs\":" << JsonAttrs(record.attrs) << "}\n";
  stream.flush();
}

void GlobalPositioningTraceRecorder::WriteManifest(const std::string& status) {
  std::ofstream manifest_stream(options_.output_path / "manifest.json",
                                std::ios::out | std::ios::trunc);
  THROW_CHECK_FILE_OPEN(manifest_stream,
                        options_.output_path / "manifest.json");
  manifest_stream << "{\n"
                  << "  \"schema_version\": " << kSchemaVersion << ",\n"
                  << "  \"run_id\": " << JsonEscape(run_id_) << ",\n"
                  << "  \"run_label\": " << JsonEscape(options_.run_label)
                  << ",\n"
                  << "  \"status\": " << JsonEscape(status) << ",\n"
                  << "  \"trace_level\": "
                  << JsonEscape(
                         GlobalPositioningTraceLevelToString(options_.level))
                  << ",\n"
                  << "  \"created_at_unix_ns\": " << created_at_unix_ns_
                  << ",\n"
                  << "  \"non_finite_numeric_values\": \"strings\",\n"
                  << "  \"options\": {\n"
                  << "    \"snapshot_every_n_iterations\": "
                  << options_.snapshot_every_n_iterations << ",\n"
                  << "    \"max_snapshotted_points\": "
                  << options_.max_snapshotted_points << "\n"
                  << "  }\n"
                  << "}\n";
  manifest_stream.flush();
}

}  // namespace colmap
