#include "colmap/estimators/global_positioning_trace.h"

#include "colmap/util/endian.h"
#include "colmap/util/file.h"
#include "colmap/util/logging.h"

#include <algorithm>
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
constexpr char kRawBinaryStorageFormat[] = "global_positioning_raw_binary_v1";
constexpr char kRawBinaryLedgerMagic[] = "GPTRLGR1";
constexpr char kRawBinaryArrayMagic[] = "GPTRARR1";
constexpr char kRawBinaryResidualValuesMagic[] = "GPTRRSV1";
constexpr int64_t kRawBinaryNoneId = -1;

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

std::string JsonOptionalDouble(const std::optional<double>& value) {
  return value.has_value() ? JsonDouble(*value) : "null";
}

std::string JsonDoubleArray(const std::vector<double>& values) {
  std::ostringstream stream;
  stream << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      stream << ",";
    }
    stream << JsonDouble(values[i]);
  }
  stream << "]";
  return stream.str();
}

std::string JsonParameterBlockDescriptors(
    const std::vector<GlobalPositioningTraceParameterBlockDescriptor>& values) {
  std::ostringstream stream;
  stream << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      stream << ",";
    }
    stream << "{"
           << "\"role\":" << JsonEscape(values[i].role) << ","
           << "\"kind\":" << JsonEscape(values[i].kind) << ","
           << "\"id\":" << values[i].id;
    if (values[i].size.has_value()) {
      stream << ",\"size\":" << *values[i].size;
    }
    stream << "}";
  }
  stream << "]";
  return stream.str();
}

std::string JsonLossConfig(const GlobalPositioningTraceLossConfig& value) {
  std::ostringstream stream;
  stream << "{"
         << "\"bucket\":" << JsonEscape(value.bucket) << ","
         << "\"type\":" << JsonEscape(value.type) << ","
         << "\"scale\":" << JsonOptionalDouble(value.scale) << ","
         << "\"weight\":" << JsonOptionalDouble(value.weight) << ","
         << "\"source\":" << JsonEscape(value.source);
  if (value.observation_count_weight.has_value()) {
    stream << ",\"observation_count_weight\":"
           << JsonDouble(*value.observation_count_weight);
  }
  stream << "}";
  return stream.str();
}

void WriteOptionalDoubleArray(std::ostringstream& stream,
                              bool& first,
                              const std::string& key,
                              const std::optional<std::vector<double>>& value) {
  if (!value.has_value()) {
    return;
  }
  if (!first) {
    stream << ",";
  }
  first = false;
  stream << JsonEscape(key) << ":" << JsonDoubleArray(*value);
}

void WriteOptionalBool(std::ostringstream& stream,
                       bool& first,
                       const std::string& key,
                       const std::optional<bool>& value) {
  if (!value.has_value()) {
    return;
  }
  if (!first) {
    stream << ",";
  }
  first = false;
  stream << JsonEscape(key) << ":" << (*value ? "true" : "false");
}

void WriteOptionalString(std::ostringstream& stream,
                         bool& first,
                         const std::string& key,
                         const std::optional<std::string>& value) {
  if (!value.has_value()) {
    return;
  }
  if (!first) {
    stream << ",";
  }
  first = false;
  stream << JsonEscape(key) << ":" << JsonEscape(*value);
}

void WriteOptionalDouble(std::ostringstream& stream,
                         bool& first,
                         const std::string& key,
                         const std::optional<double>& value) {
  if (!value.has_value()) {
    return;
  }
  if (!first) {
    stream << ",";
  }
  first = false;
  stream << JsonEscape(key) << ":" << JsonDouble(*value);
}

std::string JsonFixedParameters(
    const GlobalPositioningTraceFixedParameters& value) {
  std::ostringstream stream;
  stream << "{";
  bool first = true;
  WriteOptionalDoubleArray(
      stream, first, "cam_from_point3D_dir", value.cam_from_point3D_dir);
  WriteOptionalDoubleArray(stream,
                           first,
                           "keypoint_covariance_world_row_major",
                           value.keypoint_covariance_world_row_major);
  WriteOptionalDoubleArray(
      stream, first, "cam_from_rig_dir", value.cam_from_rig_dir);
  WriteOptionalDoubleArray(stream,
                           first,
                           "rig_from_world_rotation_wxyz",
                           value.rig_from_world_rotation_wxyz);
  WriteOptionalDoubleArray(stream,
                           first,
                           "world_from_rig_rotation_wxyz",
                           value.world_from_rig_rotation_wxyz);
  WriteOptionalDoubleArray(
      stream, first, "camera_rotation_wxyz", value.camera_rotation_wxyz);
  WriteOptionalBool(stream,
                    first,
                    "metric_depth_use_log_scale",
                    value.metric_depth_use_log_scale);
  WriteOptionalString(stream,
                      first,
                      "metric_depth_residual_type",
                      value.metric_depth_residual_type);
  WriteOptionalBool(stream,
                    first,
                    "metric_depth_zero_residual_behind",
                    value.metric_depth_zero_residual_behind);
  WriteOptionalDouble(stream,
                      first,
                      "metric_depth_log_linear_threshold",
                      value.metric_depth_log_linear_threshold);
  WriteOptionalDouble(
      stream, first, "scale_prior_target", value.scale_prior_target);
  WriteOptionalDouble(
      stream, first, "scale_prior_stddev", value.scale_prior_stddev);
  stream << "}";
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
    case GlobalPositioningTraceValue::Type::kParameterBlockArray:
      return JsonParameterBlockDescriptors(value.parameter_block_array_value);
    case GlobalPositioningTraceValue::Type::kLossConfig:
      return JsonLossConfig(value.loss_config_value);
    case GlobalPositioningTraceValue::Type::kFixedParameters:
      return JsonFixedParameters(value.fixed_parameters_value);
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

bool IsResidualValuesLevel(const GlobalPositioningTraceLevel level) {
  return static_cast<int>(level) >=
         static_cast<int>(GlobalPositioningTraceLevel::kResidualValues);
}

bool IsResidualJacobiansLevel(const GlobalPositioningTraceLevel level) {
  return static_cast<int>(level) >=
         static_cast<int>(GlobalPositioningTraceLevel::kResidualJacobians);
}

std::string IterationPrefix(const int iteration) {
  std::ostringstream stream;
  stream << "iter_" << std::setw(6) << std::setfill('0') << iteration;
  return stream.str();
}

std::filesystem::path RawBinaryPath(
    const GlobalPositioningTraceOptions& options) {
  return options.output_path / "raw_binary";
}

std::filesystem::path RawBinaryStaticPath(
    const GlobalPositioningTraceOptions& options) {
  return RawBinaryPath(options) / "static";
}

std::filesystem::path RawBinaryIterationsPath(
    const GlobalPositioningTraceOptions& options) {
  return RawBinaryPath(options) / "iterations";
}

std::filesystem::path RawBinaryIterationPath(
    const GlobalPositioningTraceOptions& options, const int iteration) {
  return RawBinaryIterationsPath(options) / IterationPrefix(iteration);
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

std::string JsonNestedSizeArray(
    const std::vector<std::vector<size_t>>& values) {
  std::ostringstream stream;
  stream << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      stream << ",";
    }
    stream << JsonSizeArray(values[i]);
  }
  stream << "]";
  return stream.str();
}

std::string JsonNestedBoolArray(const std::vector<std::vector<bool>>& values) {
  std::ostringstream stream;
  stream << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      stream << ",";
    }
    stream << "[";
    for (size_t j = 0; j < values[i].size(); ++j) {
      if (j > 0) {
        stream << ",";
      }
      stream << (values[i][j] ? "true" : "false");
    }
    stream << "]";
  }
  stream << "]";
  return stream.str();
}

std::string JsonNestedDoubleArray(
    const std::vector<std::vector<std::vector<double>>>& values) {
  std::ostringstream stream;
  stream << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      stream << ",";
    }
    stream << "[";
    for (size_t j = 0; j < values[i].size(); ++j) {
      if (j > 0) {
        stream << ",";
      }
      stream << JsonDoubleArray(values[i][j]);
    }
    stream << "]";
  }
  stream << "]";
  return stream.str();
}

std::string JsonNestedParameterBlockDescriptors(
    const std::vector<
        std::vector<GlobalPositioningTraceParameterBlockDescriptor>>& values) {
  std::ostringstream stream;
  stream << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      stream << ",";
    }
    stream << JsonParameterBlockDescriptors(values[i]);
  }
  stream << "]";
  return stream.str();
}

std::string JsonStringArray(const std::vector<std::string>& values) {
  std::ostringstream stream;
  stream << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      stream << ",";
    }
    stream << JsonEscape(values[i]);
  }
  stream << "]";
  return stream.str();
}

std::string JsonBoolArray(const std::vector<bool>& values) {
  std::ostringstream stream;
  stream << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      stream << ",";
    }
    stream << (values[i] ? "true" : "false");
  }
  stream << "]";
  return stream.str();
}

void WriteRawBytes(std::ofstream& stream,
                   const char* data,
                   const size_t size,
                   const std::filesystem::path& path) {
  stream.write(data, static_cast<std::streamsize>(size));
  THROW_CHECK(stream.good())
      << "Failed while writing global positioning raw binary trace: " << path;
}

template <typename T>
void WriteRawLittleEndian(std::ofstream& stream,
                          const T value,
                          const std::filesystem::path& path) {
  WriteBinaryLittleEndian<T>(&stream, value);
  THROW_CHECK(stream.good())
      << "Failed while writing global positioning raw binary trace: " << path;
}

void WriteRawBool(std::ofstream& stream,
                  const bool value,
                  const std::filesystem::path& path) {
  const char byte = value ? 1 : 0;
  WriteRawBytes(stream, &byte, 1, path);
}

void WriteRawString(std::ofstream& stream,
                    const std::string& value,
                    const std::filesystem::path& path) {
  THROW_CHECK_LE(value.size(), std::numeric_limits<uint32_t>::max())
      << "Raw binary trace strings are length-prefixed with uint32.";
  WriteRawLittleEndian<uint32_t>(
      stream, static_cast<uint32_t>(value.size()), path);
  if (!value.empty()) {
    WriteRawBytes(stream, value.data(), value.size(), path);
  }
}

void WriteRawParameterBlockDescriptor(
    std::ofstream& stream,
    const GlobalPositioningTraceParameterBlockDescriptor& descriptor,
    const std::filesystem::path& path) {
  THROW_CHECK(!descriptor.role.empty())
      << "Raw binary parameter block role must be non-empty.";
  THROW_CHECK(!descriptor.kind.empty())
      << "Raw binary parameter block kind must be non-empty.";
  WriteRawString(stream, descriptor.role, path);
  WriteRawString(stream, descriptor.kind, path);
  WriteRawLittleEndian<uint64_t>(stream, descriptor.id, path);
}

std::string TraceAttrString(
    const std::map<std::string, GlobalPositioningTraceValue>& attrs,
    const std::string& key) {
  const auto it = attrs.find(key);
  THROW_CHECK(it != attrs.end()) << "Missing trace attr: " << key;
  THROW_CHECK(it->second.type == GlobalPositioningTraceValue::Type::kString)
      << "Trace attr must be a string: " << key;
  return it->second.string_value;
}

int64_t TraceAttrOptionalId(
    const std::map<std::string, GlobalPositioningTraceValue>& attrs,
    const std::string& key) {
  const auto it = attrs.find(key);
  THROW_CHECK(it != attrs.end()) << "Missing trace attr: " << key;
  if (it->second.type == GlobalPositioningTraceValue::Type::kNull) {
    return kRawBinaryNoneId;
  }
  THROW_CHECK(it->second.type == GlobalPositioningTraceValue::Type::kUInt)
      << "Trace attr must be an unsigned ID or null: " << key;
  THROW_CHECK_LE(it->second.uint_value,
                 static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
      << "Raw binary trace IDs are stored as signed int64.";
  return static_cast<int64_t>(it->second.uint_value);
}

bool TraceAttrBool(
    const std::map<std::string, GlobalPositioningTraceValue>& attrs,
    const std::string& key) {
  const auto it = attrs.find(key);
  THROW_CHECK(it != attrs.end()) << "Missing trace attr: " << key;
  THROW_CHECK(it->second.type == GlobalPositioningTraceValue::Type::kBool)
      << "Trace attr must be a bool: " << key;
  return it->second.bool_value;
}

void ValidateSnapshotArray(const std::string& name,
                           const GlobalPositioningTraceSnapshotArray& array);

void WriteRawSnapshotArray(const std::filesystem::path& path,
                           const std::string& name,
                           const GlobalPositioningTraceSnapshotArray& array) {
  ValidateSnapshotArray(name, array);
  THROW_CHECK_LE(array.shape.size(), 2)
      << "Raw binary snapshot arrays support rank-1 or rank-2 arrays.";
  const uint64_t rows = static_cast<uint64_t>(array.shape.front());
  const uint64_t cols =
      array.shape.size() == 1 ? 1 : static_cast<uint64_t>(array.shape[1]);
  std::ofstream stream(path,
                       std::ios::binary | std::ios::out | std::ios::trunc);
  THROW_CHECK_FILE_OPEN(stream, path);
  WriteRawBytes(stream, kRawBinaryArrayMagic, 8, path);
  WriteRawLittleEndian<uint32_t>(stream, kSchemaVersion, path);
  WriteRawLittleEndian<uint64_t>(stream, rows, path);
  WriteRawLittleEndian<uint64_t>(stream, cols, path);
  WriteRawString(stream, name, path);
  for (const uint64_t id : array.ids) {
    THROW_CHECK_LE(id,
                   static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
        << "Raw binary snapshot IDs are stored as signed int64.";
    WriteRawLittleEndian<int64_t>(stream, static_cast<int64_t>(id), path);
  }
  for (const double value : array.values) {
    WriteRawLittleEndian<double>(stream, value, path);
  }
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

void WriteDoubleSidecar(const std::filesystem::path& path,
                        const std::vector<double>& values) {
  std::ofstream sidecar_stream(
      path, std::ios::binary | std::ios::out | std::ios::trunc);
  THROW_CHECK_FILE_OPEN(sidecar_stream, path);
  for (const double value : values) {
    WriteBinaryLittleEndian<double>(&sidecar_stream, value);
  }
  THROW_CHECK(sidecar_stream.good())
      << "Failed while writing global positioning trace sidecar: " << path;
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

void WriteResidualValueArtifactMetadata(std::ostream& stream,
                                        const std::string& name,
                                        const std::string& filename,
                                        const std::vector<size_t>& shape) {
  stream << "    " << JsonEscape(name) << ": {\n"
         << "      \"file\": " << JsonEscape(filename) << ",\n"
         << "      \"dtype\": \"float64\",\n"
         << "      \"byte_order\": \"little_endian\",\n"
         << "      \"shape\": " << JsonSizeArray(shape) << "\n"
         << "    }";
}

void WriteResidualValueArtifactMetadata(std::ostream& stream,
                                        const std::string& name,
                                        const std::string& filename,
                                        const size_t element_count) {
  WriteResidualValueArtifactMetadata(
      stream, name, filename, std::vector<size_t>{element_count});
}

size_t ValidateResidualValues(
    const GlobalPositioningTraceResidualValues& residual_values) {
  THROW_CHECK_GE(residual_values.iteration, 0)
      << "Global positioning residual values iteration must be non-negative.";
  const size_t num_residual_blocks = residual_values.residual_ids.size();
  THROW_CHECK_EQ(residual_values.residual_dims.size(), num_residual_blocks)
      << "Residual value residual_dims count must match residual_ids count.";
  THROW_CHECK_EQ(residual_values.residual_offsets.size(), num_residual_blocks)
      << "Residual value residual_offsets count must match residual_ids count.";
  THROW_CHECK_EQ(residual_values.evaluation_success.size(), num_residual_blocks)
      << "Residual value evaluation_success count must match residual_ids "
         "count.";
  THROW_CHECK_EQ(residual_values.raw_costs.size(), num_residual_blocks)
      << "Residual value raw_costs count must match residual_ids count.";
  THROW_CHECK_EQ(residual_values.robust_costs.size(), num_residual_blocks)
      << "Residual value robust_costs count must match residual_ids count.";
  THROW_CHECK_EQ(residual_values.loss_rho_values.size(),
                 num_residual_blocks * 3)
      << "Residual value loss_rho_values count must be three per residual "
         "block.";

  size_t total_scalar_residuals = 0;
  for (size_t i = 0; i < num_residual_blocks; ++i) {
    THROW_CHECK(!residual_values.residual_ids[i].empty())
        << "Residual value residual_ids entries must be non-empty.";
    THROW_CHECK_EQ(residual_values.residual_offsets[i], total_scalar_residuals)
        << "Residual value residual_offsets must be contiguous cumulative "
           "offsets.";
    total_scalar_residuals += residual_values.residual_dims[i];
  }
  THROW_CHECK_EQ(residual_values.raw_residuals.size(), total_scalar_residuals)
      << "Residual value raw_residuals count must match sum(residual_dims).";
  if (residual_values.has_raw_jacobians) {
    THROW_CHECK_EQ(residual_values.parameter_block_sizes.size(),
                   num_residual_blocks)
        << "Residual value parameter_block_sizes count must match "
           "residual_ids count.";
    THROW_CHECK_EQ(residual_values.raw_jacobian_offsets.size(),
                   num_residual_blocks)
        << "Residual value raw_jacobian_offsets count must match "
           "residual_ids count.";
    THROW_CHECK_EQ(residual_values.parameter_blocks.size(), num_residual_blocks)
        << "Residual value parameter_blocks count must match residual_ids "
           "count.";
    THROW_CHECK_EQ(residual_values.parameter_block_is_constant.size(),
                   num_residual_blocks)
        << "Residual value parameter_block_is_constant count must match "
           "residual_ids count.";
    THROW_CHECK_EQ(residual_values.parameter_block_lower_bounds.size(),
                   num_residual_blocks)
        << "Residual value parameter_block_lower_bounds count must match "
           "residual_ids count.";
    size_t total_jacobian_scalars = 0;
    for (size_t i = 0; i < num_residual_blocks; ++i) {
      THROW_CHECK_EQ(residual_values.raw_jacobian_offsets[i].size(),
                     residual_values.parameter_block_sizes[i].size())
          << "Residual value raw_jacobian_offsets inner count must match "
             "parameter_block_sizes inner count.";
      THROW_CHECK_EQ(residual_values.parameter_blocks[i].size(),
                     residual_values.parameter_block_sizes[i].size())
          << "Residual value parameter_blocks inner count must match "
             "parameter_block_sizes inner count.";
      THROW_CHECK_EQ(residual_values.parameter_block_is_constant[i].size(),
                     residual_values.parameter_block_sizes[i].size())
          << "Residual value parameter_block_is_constant inner count must "
             "match parameter_block_sizes inner count.";
      THROW_CHECK_EQ(residual_values.parameter_block_lower_bounds[i].size(),
                     residual_values.parameter_block_sizes[i].size())
          << "Residual value parameter_block_lower_bounds inner count must "
             "match parameter_block_sizes inner count.";
      for (size_t block_idx = 0;
           block_idx < residual_values.parameter_block_sizes[i].size();
           ++block_idx) {
        THROW_CHECK(
            !residual_values.parameter_blocks[i][block_idx].role.empty())
            << "Residual value parameter block role must be non-empty.";
        THROW_CHECK(
            !residual_values.parameter_blocks[i][block_idx].kind.empty())
            << "Residual value parameter block kind must be non-empty.";
        THROW_CHECK_EQ(
            residual_values.parameter_block_lower_bounds[i][block_idx].size(),
            residual_values.parameter_block_sizes[i][block_idx])
            << "Residual value parameter_block_lower_bounds block shape must "
               "match parameter_block_sizes.";
        THROW_CHECK_EQ(residual_values.raw_jacobian_offsets[i][block_idx],
                       total_jacobian_scalars)
            << "Residual value raw_jacobian_offsets must be contiguous "
               "cumulative offsets.";
        total_jacobian_scalars +=
            residual_values.residual_dims[i] *
            residual_values.parameter_block_sizes[i][block_idx];
      }
    }
    THROW_CHECK_EQ(residual_values.raw_jacobians.size(), total_jacobian_scalars)
        << "Residual value raw_jacobians count must match residual_dims times "
           "parameter_block_sizes.";
  } else {
    THROW_CHECK(residual_values.parameter_block_sizes.empty())
        << "Residual value parameter_block_sizes require has_raw_jacobians.";
    THROW_CHECK(residual_values.raw_jacobian_offsets.empty())
        << "Residual value raw_jacobian_offsets require has_raw_jacobians.";
    THROW_CHECK(residual_values.parameter_blocks.empty())
        << "Residual value parameter_blocks require has_raw_jacobians.";
    THROW_CHECK(residual_values.parameter_block_is_constant.empty())
        << "Residual value parameter_block_is_constant require "
           "has_raw_jacobians.";
    THROW_CHECK(residual_values.parameter_block_lower_bounds.empty())
        << "Residual value parameter_block_lower_bounds require "
           "has_raw_jacobians.";
    THROW_CHECK(residual_values.raw_jacobians.empty())
        << "Residual value raw_jacobians require has_raw_jacobians.";
  }
  return total_scalar_residuals;
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
    case GlobalPositioningTraceLevel::kResidualJacobians:
      return "residual_jacobians";
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

GlobalPositioningTraceValue GlobalPositioningTraceValue::ParameterBlockArray(
    std::vector<GlobalPositioningTraceParameterBlockDescriptor> value) {
  GlobalPositioningTraceValue trace_value;
  trace_value.type = Type::kParameterBlockArray;
  trace_value.parameter_block_array_value = std::move(value);
  return trace_value;
}

GlobalPositioningTraceValue GlobalPositioningTraceValue::LossConfig(
    GlobalPositioningTraceLossConfig value) {
  GlobalPositioningTraceValue trace_value;
  trace_value.type = Type::kLossConfig;
  trace_value.loss_config_value = std::move(value);
  return trace_value;
}

GlobalPositioningTraceValue GlobalPositioningTraceValue::FixedParameters(
    GlobalPositioningTraceFixedParameters value) {
  GlobalPositioningTraceValue trace_value;
  trace_value.type = Type::kFixedParameters;
  trace_value.fixed_parameters_value = std::move(value);
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
  RemoveTraceArtifactIfExists(options_.output_path / "residual_values");
  RemoveTraceArtifactIfExists(options_.output_path / "raw_binary");

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

    const std::filesystem::path raw_binary_static_path =
        RawBinaryStaticPath(options_);
    CreateDirIfNotExists(raw_binary_static_path, /*recursive=*/true);
    THROW_CHECK(ExistsDir(raw_binary_static_path))
        << "Global positioning raw binary static path is not a directory: "
        << raw_binary_static_path;
    const std::filesystem::path raw_binary_iterations_path =
        RawBinaryIterationsPath(options_);
    CreateDirIfNotExists(raw_binary_iterations_path, /*recursive=*/true);
    THROW_CHECK(ExistsDir(raw_binary_iterations_path))
        << "Global positioning raw binary iterations path is not a directory: "
        << raw_binary_iterations_path;
    raw_binary_residual_ledger_stream_.open(
        raw_binary_static_path / "residual_ledger.bin",
        std::ios::binary | std::ios::out | std::ios::trunc);
    THROW_CHECK_FILE_OPEN(raw_binary_residual_ledger_stream_,
                          raw_binary_static_path / "residual_ledger.bin");
    WriteRawBinaryResidualLedgerHeader();
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

  if (IsResidualValuesEnabled()) {
    const std::filesystem::path residual_values_path =
        options_.output_path / "residual_values";
    THROW_CHECK(!ExistsFile(residual_values_path))
        << "Global positioning residual values path points to a file: "
        << residual_values_path;
    if (!ExistsDir(residual_values_path)) {
      CreateDirIfNotExists(residual_values_path, /*recursive=*/true);
    }
    THROW_CHECK(ExistsDir(residual_values_path))
        << "Global positioning residual values path is not a directory: "
        << residual_values_path;
  }

  WriteManifest("running");
}

bool GlobalPositioningTraceRecorder::IsResidualLedgerEnabled() const {
  return IsResidualLedgerLevel(options_.level);
}

bool GlobalPositioningTraceRecorder::IsParameterSnapshotsEnabled() const {
  return IsParameterSnapshotsLevel(options_.level);
}

bool GlobalPositioningTraceRecorder::IsResidualValuesEnabled() const {
  return IsResidualValuesLevel(options_.level);
}

bool GlobalPositioningTraceRecorder::IsResidualJacobiansEnabled() const {
  return IsResidualJacobiansLevel(options_.level);
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
  WriteRawBinaryResidualBlock(record);
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
  WriteRawBinaryParameterSnapshot(snapshot);

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

void GlobalPositioningTraceRecorder::WriteResidualValues(
    const GlobalPositioningTraceResidualValues& residual_values) {
  if (!IsResidualValuesEnabled()) {
    return;
  }

  THROW_CHECK_EQ(sizeof(double), 8)
      << "Global positioning residual value sidecars require 64-bit doubles.";
  THROW_CHECK(std::numeric_limits<double>::is_iec559)
      << "Global positioning residual value sidecars require IEEE-754 doubles.";
  THROW_CHECK(IsLittleEndian())
      << "Global positioning residual value binaries are defined as "
         "little-endian.";

  const size_t total_scalar_residuals = ValidateResidualValues(residual_values);
  WriteRawBinaryResidualValues(residual_values);
  const size_t num_residual_blocks = residual_values.residual_ids.size();
  const std::filesystem::path residual_values_path =
      options_.output_path / "residual_values";
  THROW_CHECK(ExistsDir(residual_values_path))
      << "Global positioning residual values path is not a directory: "
      << residual_values_path;

  const std::string prefix = IterationPrefix(residual_values.iteration);
  const std::filesystem::path metadata_path =
      residual_values_path / (prefix + ".json");
  const std::string raw_residuals_filename = prefix + "_raw_residuals_f64.bin";
  const std::string raw_costs_filename = prefix + "_raw_costs_f64.bin";
  const std::string robust_costs_filename = prefix + "_robust_costs_f64.bin";
  const std::string loss_rho_values_filename =
      prefix + "_loss_rho_values_f64.bin";
  const std::string raw_jacobians_filename = prefix + "_raw_jacobians_f64.bin";

  WriteDoubleSidecar(residual_values_path / raw_residuals_filename,
                     residual_values.raw_residuals);
  WriteDoubleSidecar(residual_values_path / raw_costs_filename,
                     residual_values.raw_costs);
  WriteDoubleSidecar(residual_values_path / robust_costs_filename,
                     residual_values.robust_costs);
  WriteDoubleSidecar(residual_values_path / loss_rho_values_filename,
                     residual_values.loss_rho_values);
  if (residual_values.has_raw_jacobians) {
    WriteDoubleSidecar(residual_values_path / raw_jacobians_filename,
                       residual_values.raw_jacobians);
  }

  std::ofstream metadata_stream(metadata_path, std::ios::out | std::ios::trunc);
  THROW_CHECK_FILE_OPEN(metadata_stream, metadata_path);
  metadata_stream << "{\n"
                  << "  \"schema_version\": " << kSchemaVersion << ",\n"
                  << "  \"run_id\": " << JsonEscape(run_id_) << ",\n"
                  << "  \"iteration\": " << residual_values.iteration << ",\n"
                  << "  \"dtype\": \"float64\",\n"
                  << "  \"byte_order\": \"little_endian\",\n"
                  << "  \"num_residual_blocks\": " << num_residual_blocks
                  << ",\n"
                  << "  \"total_scalar_residuals\": " << total_scalar_residuals
                  << ",\n"
                  << "  \"has_raw_jacobians\": "
                  << (residual_values.has_raw_jacobians ? "true" : "false")
                  << ",\n";
  if (residual_values.has_raw_jacobians) {
    metadata_stream << "  \"total_jacobian_scalars\": "
                    << residual_values.raw_jacobians.size() << ",\n";
  }
  metadata_stream << "  \"residual_ids\": "
                  << JsonStringArray(residual_values.residual_ids) << ",\n"
                  << "  \"residual_dims\": "
                  << JsonSizeArray(residual_values.residual_dims) << ",\n"
                  << "  \"residual_offsets\": "
                  << JsonSizeArray(residual_values.residual_offsets) << ",\n"
                  << "  \"evaluation_success\": "
                  << JsonBoolArray(residual_values.evaluation_success) << ",\n"
                  << "  \"loss_rho_layout\": "
                  << JsonEscape("residual_block_major/rho0_rho1_rho2") << ",\n";
  if (residual_values.has_raw_jacobians) {
    metadata_stream
        << "  \"parameter_block_sizes\": "
        << JsonNestedSizeArray(residual_values.parameter_block_sizes) << ",\n"
        << "  \"raw_jacobian_offsets\": "
        << JsonNestedSizeArray(residual_values.raw_jacobian_offsets) << ",\n"
        << "  \"parameter_blocks\": "
        << JsonNestedParameterBlockDescriptors(residual_values.parameter_blocks)
        << ",\n"
        << "  \"parameter_block_is_constant\": "
        << JsonNestedBoolArray(residual_values.parameter_block_is_constant)
        << ",\n"
        << "  \"parameter_block_lower_bounds\": "
        << JsonNestedDoubleArray(residual_values.parameter_block_lower_bounds)
        << ",\n"
        << "  \"raw_jacobian_layout\": "
        << JsonEscape(
               "residual_block_major/parameter_block_major/"
               "row_major")
        << ",\n"
        << "  \"jacobian_domain\": "
        << JsonEscape("raw_cost_function_ambient_parameters") << ",\n"
        << "  \"loss_applied_to_jacobians\": false,\n"
        << "  \"manifold_applied_to_jacobians\": false,\n"
        << "  \"constant_parameter_blocks_included\": true"
        << ",\n";
  }
  metadata_stream << "  \"artifacts\": {\n";
  WriteResidualValueArtifactMetadata(metadata_stream,
                                     "raw_residuals",
                                     raw_residuals_filename,
                                     total_scalar_residuals);
  metadata_stream << ",\n";
  WriteResidualValueArtifactMetadata(
      metadata_stream, "raw_costs", raw_costs_filename, num_residual_blocks);
  metadata_stream << ",\n";
  WriteResidualValueArtifactMetadata(metadata_stream,
                                     "robust_costs",
                                     robust_costs_filename,
                                     num_residual_blocks);
  metadata_stream << ",\n";
  WriteResidualValueArtifactMetadata(metadata_stream,
                                     "loss_rho_values",
                                     loss_rho_values_filename,
                                     {num_residual_blocks, 3});
  if (residual_values.has_raw_jacobians) {
    metadata_stream << ",\n";
    WriteResidualValueArtifactMetadata(metadata_stream,
                                       "raw_jacobians",
                                       raw_jacobians_filename,
                                       residual_values.raw_jacobians.size());
  }
  metadata_stream << "\n"
                  << "  }\n"
                  << "}\n";
  metadata_stream.flush();
}

void GlobalPositioningTraceRecorder::MarkFinished(std::string status) {
  WriteManifest(status);
}

GlobalPositioningRawBinaryTraceIterationArtifacts&
GlobalPositioningTraceRecorder::RawBinaryIterationArtifacts(
    const int iteration) {
  for (GlobalPositioningRawBinaryTraceIterationArtifacts& artifacts :
       raw_binary_iterations_) {
    if (artifacts.iteration == iteration) {
      return artifacts;
    }
  }
  raw_binary_iterations_.push_back({iteration});
  std::sort(raw_binary_iterations_.begin(),
            raw_binary_iterations_.end(),
            [](const GlobalPositioningRawBinaryTraceIterationArtifacts& a,
               const GlobalPositioningRawBinaryTraceIterationArtifacts& b) {
              return a.iteration < b.iteration;
            });
  for (GlobalPositioningRawBinaryTraceIterationArtifacts& artifacts :
       raw_binary_iterations_) {
    if (artifacts.iteration == iteration) {
      return artifacts;
    }
  }
  THROW_CHECK(false) << "Failed to register raw binary iteration.";
  return raw_binary_iterations_.front();
}

void GlobalPositioningTraceRecorder::WriteRawBinaryResidualLedgerHeader() {
  if (!raw_binary_residual_ledger_stream_.is_open()) {
    return;
  }
  const std::filesystem::path path =
      RawBinaryStaticPath(options_) / "residual_ledger.bin";
  raw_binary_residual_ledger_stream_.seekp(0, std::ios::beg);
  WriteRawBytes(
      raw_binary_residual_ledger_stream_, kRawBinaryLedgerMagic, 8, path);
  WriteRawLittleEndian<uint32_t>(
      raw_binary_residual_ledger_stream_, kSchemaVersion, path);
  WriteRawLittleEndian<uint64_t>(raw_binary_residual_ledger_stream_,
                                 raw_binary_residual_ledger_count_,
                                 path);
  raw_binary_residual_ledger_stream_.seekp(0, std::ios::end);
}

void GlobalPositioningTraceRecorder::UpdateRawBinaryResidualLedgerHeader() {
  if (!raw_binary_residual_ledger_stream_.is_open()) {
    return;
  }
  WriteRawBinaryResidualLedgerHeader();
  raw_binary_residual_ledger_stream_.flush();
}

void GlobalPositioningTraceRecorder::WriteRawBinaryResidualBlock(
    const GlobalPositioningTraceRecord& record) {
  if (!raw_binary_residual_ledger_stream_.is_open()) {
    return;
  }
  const std::filesystem::path path =
      RawBinaryStaticPath(options_) / "residual_ledger.bin";
  raw_binary_residual_ledger_stream_.seekp(0, std::ios::end);
  WriteRawString(raw_binary_residual_ledger_stream_,
                 TraceAttrString(record.attrs, "residual_id"),
                 path);
  WriteRawString(raw_binary_residual_ledger_stream_,
                 TraceAttrString(record.attrs, "residual_type"),
                 path);
  WriteRawString(raw_binary_residual_ledger_stream_,
                 TraceAttrString(record.attrs, "loss_bucket"),
                 path);
  WriteRawLittleEndian<int64_t>(raw_binary_residual_ledger_stream_,
                                TraceAttrOptionalId(record.attrs, "frame_id"),
                                path);
  WriteRawLittleEndian<int64_t>(raw_binary_residual_ledger_stream_,
                                TraceAttrOptionalId(record.attrs, "image_id"),
                                path);
  WriteRawLittleEndian<int64_t>(raw_binary_residual_ledger_stream_,
                                TraceAttrOptionalId(record.attrs, "point3D_id"),
                                path);
  WriteRawBool(raw_binary_residual_ledger_stream_,
               TraceAttrBool(record.attrs, "is_lc_observation"),
               path);
  WriteRawString(
      raw_binary_residual_ledger_stream_, JsonAttrs(record.attrs), path);
  ++raw_binary_residual_ledger_count_;
}

void GlobalPositioningTraceRecorder::WriteRawBinaryParameterSnapshot(
    const GlobalPositioningTraceParameterSnapshot& snapshot) {
  if (!IsParameterSnapshotsEnabled() || !IsResidualLedgerEnabled()) {
    return;
  }
  const std::filesystem::path iteration_path =
      RawBinaryIterationPath(options_, snapshot.iteration);
  CreateDirIfNotExists(iteration_path, /*recursive=*/true);
  THROW_CHECK(ExistsDir(iteration_path))
      << "Global positioning raw binary iteration path is not a directory: "
      << iteration_path;

  WriteRawSnapshotArray(iteration_path / "frame_centers.bin",
                        "frame_centers",
                        snapshot.frame_centers);
  WriteRawSnapshotArray(
      iteration_path / "point_xyz.bin", "point_xyz", snapshot.points3D);
  WriteRawSnapshotArray(
      iteration_path / "scales.bin", "scales", snapshot.scales);
  GlobalPositioningRawBinaryTraceIterationArtifacts& artifacts =
      RawBinaryIterationArtifacts(snapshot.iteration);
  artifacts.has_frame_centers = true;
  artifacts.has_point_xyz = true;
  artifacts.has_scales = true;
  if (snapshot.dmap_scales.has_value()) {
    WriteRawSnapshotArray(iteration_path / "dmap_scales.bin",
                          "dmap_scales",
                          *snapshot.dmap_scales);
    artifacts.has_dmap_scales = true;
  }
  if (snapshot.cams_in_rig.has_value()) {
    WriteRawSnapshotArray(iteration_path / "cams_in_rig.bin",
                          "cams_in_rig",
                          *snapshot.cams_in_rig);
    artifacts.has_cams_in_rig = true;
  }
}

void GlobalPositioningTraceRecorder::WriteRawBinaryResidualValues(
    const GlobalPositioningTraceResidualValues& residual_values) {
  if (!IsResidualValuesEnabled() || !IsResidualLedgerEnabled()) {
    return;
  }
  const size_t total_scalar_residuals = ValidateResidualValues(residual_values);
  const size_t num_residual_blocks = residual_values.residual_ids.size();
  const std::filesystem::path iteration_path =
      RawBinaryIterationPath(options_, residual_values.iteration);
  CreateDirIfNotExists(iteration_path, /*recursive=*/true);
  THROW_CHECK(ExistsDir(iteration_path))
      << "Global positioning raw binary iteration path is not a directory: "
      << iteration_path;
  const std::filesystem::path path = iteration_path / "residual_values.bin";
  std::ofstream stream(path,
                       std::ios::binary | std::ios::out | std::ios::trunc);
  THROW_CHECK_FILE_OPEN(stream, path);
  WriteRawBytes(stream, kRawBinaryResidualValuesMagic, 8, path);
  WriteRawLittleEndian<uint32_t>(
      stream,
      residual_values.has_raw_jacobians ? uint32_t{2} : uint32_t{1},
      path);
  WriteRawLittleEndian<int64_t>(stream, residual_values.iteration, path);
  WriteRawLittleEndian<uint64_t>(
      stream, static_cast<uint64_t>(num_residual_blocks), path);
  WriteRawLittleEndian<uint64_t>(
      stream, static_cast<uint64_t>(total_scalar_residuals), path);
  WriteRawBool(stream, true, path);
  if (residual_values.has_raw_jacobians) {
    WriteRawBool(stream, true, path);
  }
  for (size_t i = 0; i < num_residual_blocks; ++i) {
    THROW_CHECK_LE(residual_values.residual_dims[i],
                   std::numeric_limits<uint32_t>::max())
        << "Raw binary residual dimensions are stored as uint32.";
    WriteRawString(stream, residual_values.residual_ids[i], path);
    WriteRawLittleEndian<uint32_t>(
        stream, static_cast<uint32_t>(residual_values.residual_dims[i]), path);
    WriteRawLittleEndian<uint64_t>(
        stream,
        static_cast<uint64_t>(residual_values.residual_offsets[i]),
        path);
    WriteRawBool(stream, residual_values.evaluation_success[i], path);
  }
  for (const double value : residual_values.raw_residuals) {
    WriteRawLittleEndian<double>(stream, value, path);
  }
  for (const double value : residual_values.raw_costs) {
    WriteRawLittleEndian<double>(stream, value, path);
  }
  for (const double value : residual_values.robust_costs) {
    WriteRawLittleEndian<double>(stream, value, path);
  }
  for (const double value : residual_values.loss_rho_values) {
    WriteRawLittleEndian<double>(stream, value, path);
  }
  if (residual_values.has_raw_jacobians) {
    WriteRawLittleEndian<uint64_t>(
        stream,
        static_cast<uint64_t>(residual_values.raw_jacobians.size()),
        path);
    for (size_t residual_idx = 0; residual_idx < num_residual_blocks;
         ++residual_idx) {
      const size_t num_parameter_blocks =
          residual_values.parameter_block_sizes[residual_idx].size();
      THROW_CHECK_LE(num_parameter_blocks, std::numeric_limits<uint32_t>::max())
          << "Raw binary parameter block counts are stored as uint32.";
      WriteRawLittleEndian<uint32_t>(
          stream, static_cast<uint32_t>(num_parameter_blocks), path);
      for (size_t block_idx = 0; block_idx < num_parameter_blocks;
           ++block_idx) {
        THROW_CHECK_LE(
            residual_values.parameter_block_sizes[residual_idx][block_idx],
            std::numeric_limits<uint32_t>::max())
            << "Raw binary parameter block sizes are stored as uint32.";
        WriteRawParameterBlockDescriptor(
            stream,
            residual_values.parameter_blocks[residual_idx][block_idx],
            path);
        WriteRawLittleEndian<uint32_t>(
            stream,
            static_cast<uint32_t>(
                residual_values.parameter_block_sizes[residual_idx][block_idx]),
            path);
        WriteRawLittleEndian<uint64_t>(
            stream,
            static_cast<uint64_t>(
                residual_values.raw_jacobian_offsets[residual_idx][block_idx]),
            path);
        WriteRawBool(stream,
                     residual_values
                         .parameter_block_is_constant[residual_idx][block_idx],
                     path);
        for (const double value :
             residual_values
                 .parameter_block_lower_bounds[residual_idx][block_idx]) {
          WriteRawLittleEndian<double>(stream, value, path);
        }
      }
    }
    for (const double value : residual_values.raw_jacobians) {
      WriteRawLittleEndian<double>(stream, value, path);
    }
  }
  RawBinaryIterationArtifacts(residual_values.iteration).has_residual_values =
      true;
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
  WriteRawBinaryManifest(status);
}

void GlobalPositioningTraceRecorder::WriteRawBinaryManifest(
    const std::string& status) {
  if (!IsResidualLedgerEnabled()) {
    return;
  }
  UpdateRawBinaryResidualLedgerHeader();
  const std::filesystem::path raw_binary_path = RawBinaryPath(options_);
  CreateDirIfNotExists(raw_binary_path, /*recursive=*/true);
  const std::filesystem::path manifest_path = raw_binary_path / "manifest.json";
  std::ofstream manifest_stream(manifest_path, std::ios::out | std::ios::trunc);
  THROW_CHECK_FILE_OPEN(manifest_stream, manifest_path);
  manifest_stream
      << "{\n"
      << "  \"schema_version\": " << kSchemaVersion << ",\n"
      << "  \"storage_format\": " << JsonEscape(kRawBinaryStorageFormat)
      << ",\n"
      << "  \"run_id\": " << JsonEscape(run_id_) << ",\n"
      << "  \"run_label\": " << JsonEscape(options_.run_label) << ",\n"
      << "  \"status\": " << JsonEscape(status) << ",\n"
      << "  \"trace_level\": "
      << JsonEscape(GlobalPositioningTraceLevelToString(options_.level))
      << ",\n"
      << "  \"byte_order\": \"little_endian\",\n"
      << "  \"dtype\": \"float64\",\n"
      << "  \"static\": {\n"
      << "    \"residual_ledger\": " << JsonEscape("static/residual_ledger.bin")
      << "\n"
      << "  },\n"
      << "  \"iterations\": [\n";
  bool first = true;
  for (const GlobalPositioningRawBinaryTraceIterationArtifacts& artifacts :
       raw_binary_iterations_) {
    if (!first) {
      manifest_stream << ",\n";
    }
    first = false;
    const std::string prefix = IterationPrefix(artifacts.iteration);
    manifest_stream << "    {\n"
                    << "      \"iteration\": " << artifacts.iteration << ",\n"
                    << "      \"directory\": "
                    << JsonEscape("iterations/" + prefix);
    if (artifacts.has_frame_centers) {
      manifest_stream << ",\n      \"frame_centers\": "
                      << JsonEscape("frame_centers.bin");
    }
    if (artifacts.has_point_xyz) {
      manifest_stream << ",\n      \"point_xyz\": "
                      << JsonEscape("point_xyz.bin");
    }
    if (artifacts.has_scales) {
      manifest_stream << ",\n      \"scales\": " << JsonEscape("scales.bin");
    }
    if (artifacts.has_dmap_scales) {
      manifest_stream << ",\n      \"dmap_scales\": "
                      << JsonEscape("dmap_scales.bin");
    }
    if (artifacts.has_cams_in_rig) {
      manifest_stream << ",\n      \"cams_in_rig\": "
                      << JsonEscape("cams_in_rig.bin");
    }
    if (artifacts.has_residual_values) {
      manifest_stream << ",\n      \"residual_values\": "
                      << JsonEscape("residual_values.bin");
    }
    manifest_stream << "\n"
                    << "    }";
  }
  manifest_stream << "\n"
                  << "  ]\n"
                  << "}\n";
  manifest_stream.flush();
}

}  // namespace colmap
