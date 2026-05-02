// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "colmap/estimators/global_positioning.h"

#include "colmap/estimators/cost_functions/metric_depth.h"
#include "colmap/math/random.h"
#include "colmap/scene/database_cache.h"
#include "colmap/scene/pose_graph.h"
#include "colmap/scene/reconstruction_matchers.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/endian.h"
#include "colmap/util/file.h"
#include "colmap/util/testing.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iterator>
#include <limits>
#include <optional>
#include <sstream>
#include <tuple>

#include <ceres/loss_function.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(GlobalPositioning, Nominal) {
  SetPRNGSeed(0);

  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 10;
  synthetic_dataset_options.num_points3D = 200;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  DatabaseCache database_cache;
  DatabaseCache::Options cache_options;
  database_cache.Load(*database, cache_options);

  PoseGraph pose_graph;
  pose_graph.Load(*database_cache.CorrespondenceGraph());

  // Copy GT reconstruction and keep only rotations (reset translations).
  Reconstruction reconstruction = gt_reconstruction;
  for (const auto& [frame_id, _] : reconstruction.Frames()) {
    Frame& frame = reconstruction.Frame(frame_id);
    frame.SetRigFromWorld(
        Rigid3d(frame.RigFromWorld().rotation(), Eigen::Vector3d::Zero()));
  }

  GlobalPositionerOptions options;
  options.use_gpu = false;
  options.random_seed = 42;
  options.solver_options.minimizer_progress_to_stdout = false;

  const bool success =
      RunGlobalPositioning(options, pose_graph, reconstruction);
  ASSERT_TRUE(success);

  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(reconstruction,
                                 /*max_rotation_error_deg=*/0.1,
                                 /*max_proj_center_error=*/0.5,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.0));
}

TEST(GlobalPositioning, MultiCameraRig) {
  SetPRNGSeed(0);

  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 3;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 200;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  DatabaseCache database_cache;
  DatabaseCache::Options cache_options;
  database_cache.Load(*database, cache_options);

  PoseGraph pose_graph;
  pose_graph.Load(*database_cache.CorrespondenceGraph());

  // Copy GT reconstruction and keep only rotations (reset translations).
  Reconstruction reconstruction = gt_reconstruction;
  for (const auto& [frame_id, _] : reconstruction.Frames()) {
    Frame& frame = reconstruction.Frame(frame_id);
    frame.SetRigFromWorld(
        Rigid3d(frame.RigFromWorld().rotation(), Eigen::Vector3d::Zero()));
  }

  GlobalPositionerOptions options;
  options.use_gpu = false;
  options.random_seed = 42;
  options.solver_options.minimizer_progress_to_stdout = false;

  const bool success =
      RunGlobalPositioning(options, pose_graph, reconstruction);
  ASSERT_TRUE(success);

  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(reconstruction,
                                 /*max_rotation_error_deg=*/0.1,
                                 /*max_proj_center_error=*/0.5,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.0));
}

// ---- LossConfig::CreateLossFunction() dispatch ----

TEST(LossConfig, TrivialDispatch) {
  LossConfig config;
  config.type = LossFunctionType::TRIVIAL;
  config.scale = 1.0;
  config.weight = 1.0;
  std::shared_ptr<ceres::LossFunction> loss = config.CreateLossFunction();
  ASSERT_NE(loss, nullptr);

  // TrivialLoss: rho = (s, 1, 0) for any s.
  double rho[3];
  loss->Evaluate(/*sq_norm=*/4.0, rho);
  EXPECT_NEAR(rho[0], 4.0, 1e-12);
  EXPECT_NEAR(rho[1], 1.0, 1e-12);
  EXPECT_NEAR(rho[2], 0.0, 1e-12);

  // dynamic_cast confirms exact type when no ScaledLoss wrap.
  EXPECT_NE(dynamic_cast<ceres::TrivialLoss*>(loss.get()), nullptr);
}

TEST(LossConfig, HuberDispatchSemantics) {
  LossConfig config;
  config.type = LossFunctionType::HUBER;
  config.scale = 0.5;
  config.weight = 1.0;
  std::shared_ptr<ceres::LossFunction> loss = config.CreateLossFunction();
  ASSERT_NE(loss, nullptr);
  EXPECT_NE(dynamic_cast<ceres::HuberLoss*>(loss.get()), nullptr);

  // HuberLoss(a) with sq_norm s:
  //   s <= a^2: rho[0] = s
  //   s >  a^2: rho[0] = 2*a*sqrt(s) - a^2
  const double a = 0.5;
  const double a2 = a * a;

  double rho_below[3];
  loss->Evaluate(/*sq_norm=*/0.1, rho_below);  // 0.1 < 0.25 -> quadratic
  EXPECT_NEAR(rho_below[0], 0.1, 1e-12);

  double rho_above[3];
  loss->Evaluate(/*sq_norm=*/4.0, rho_above);  // 4.0 > 0.25 -> linear
  const double expected = 2.0 * a * std::sqrt(4.0) - a2;
  EXPECT_NEAR(rho_above[0], expected, 1e-12);
}

TEST(LossConfig, ScaledLossWrapWhenWeightNonOne) {
  // weight=2.0 over a TrivialLoss → rho[0] = 2 * sq_norm.
  LossConfig config;
  config.type = LossFunctionType::TRIVIAL;
  config.scale = 1.0;
  config.weight = 2.0;
  std::shared_ptr<ceres::LossFunction> loss = config.CreateLossFunction();
  ASSERT_NE(loss, nullptr);

  // After ScaledLoss wrap, the outer pointer is no longer a TrivialLoss.
  EXPECT_EQ(dynamic_cast<ceres::TrivialLoss*>(loss.get()), nullptr);

  double rho[3];
  loss->Evaluate(/*sq_norm=*/3.0, rho);
  EXPECT_NEAR(rho[0], 6.0, 1e-12);  // 2 * sq_norm
  EXPECT_NEAR(rho[1], 2.0, 1e-12);  // 2 * 1.0
  EXPECT_NEAR(rho[2], 0.0, 1e-12);

  // Weight=2 over Huber: rho should be exactly 2x the unweighted Huber rho.
  LossConfig huber_unweighted;
  huber_unweighted.type = LossFunctionType::HUBER;
  huber_unweighted.scale = 0.7;
  huber_unweighted.weight = 1.0;
  LossConfig huber_weighted = huber_unweighted;
  huber_weighted.weight = 2.0;
  auto loss_u = huber_unweighted.CreateLossFunction();
  auto loss_w = huber_weighted.CreateLossFunction();

  double rho_u[3];
  double rho_w[3];
  loss_u->Evaluate(/*sq_norm=*/2.5, rho_u);
  loss_w->Evaluate(/*sq_norm=*/2.5, rho_w);
  EXPECT_NEAR(rho_w[0], 2.0 * rho_u[0], 1e-12);
  EXPECT_NEAR(rho_w[1], 2.0 * rho_u[1], 1e-12);
  EXPECT_NEAR(rho_w[2], 2.0 * rho_u[2], 1e-12);
}

TEST(LossConfig, CauchyAndSoftL1Smoke) {
  for (LossFunctionType type :
       {LossFunctionType::CAUCHY, LossFunctionType::SOFT_L1}) {
    LossConfig config;
    config.type = type;
    config.scale = 0.5;
    config.weight = 1.0;
    std::shared_ptr<ceres::LossFunction> loss = config.CreateLossFunction();
    ASSERT_NE(loss, nullptr);
    double rho[3];
    loss->Evaluate(/*sq_norm=*/1.0, rho);
    EXPECT_TRUE(std::isfinite(rho[0]));
    EXPECT_TRUE(std::isfinite(rho[1]));
    EXPECT_TRUE(std::isfinite(rho[2]));
  }
}

// ---- MetricDepthError log-linear C¹ continuity at threshold ----

TEST(MetricDepthError, SmoothLogLinearTransitionC1Continuity) {
  const Eigen::Quaterniond identity = Eigen::Quaterniond::Identity();
  const double depth_prior = 1.0;
  const double sigma_depth = 0.2;
  const double threshold = 0.5;

  MetricDepthOptions options;
  options.residual_type = MetricDepthResidualType::kLogLinear;
  options.log_linear_threshold = threshold;

  MetricDepthError functor(identity, depth_prior, sigma_depth, options);

  // Helper: evaluate residual at z_est with camera at origin, identity
  // rotation, point at (0, 0, z_est), dmap_scale = 1.0.
  auto eval = [&](double z_est) -> double {
    const double c_i[3] = {0.0, 0.0, 0.0};
    const double X_k[3] = {0.0, 0.0, z_est};
    const double dmap_scale[1] = {1.0};
    double residual[1] = {0.0};
    functor(c_i, X_k, dmap_scale, residual);
    return residual[0];
  };

  // --- C⁰: residual values match at threshold ---
  {
    const double eps = 1e-8;
    const double r_above = eval(threshold + eps);
    const double r_below = eval(threshold - eps);
    EXPECT_NEAR(r_above, r_below, 1e-5)
        << "C0 violated: residual discontinuity at threshold";
  }

  // --- C¹: slopes match at threshold ---
  // Finite-difference the slope on each side of the boundary.
  // Use a probe point close to the boundary, then finite-diff around it.
  {
    const double delta = 1e-5;  // offset from boundary
    const double h = 1e-8;      // finite-difference step

    // Slope on the log side (above threshold)
    const double z_above = threshold + delta;
    const double slope_above =
        (eval(z_above + h) - eval(z_above - h)) / (2.0 * h);

    // Slope on the linear side (below threshold)
    const double z_below = threshold - delta;
    const double slope_below =
        (eval(z_below + h) - eval(z_below - h)) / (2.0 * h);

    EXPECT_NEAR(slope_above, slope_below, 1e-3)
        << "C1 violated: slope discontinuity at threshold";
  }

  // --- Verify exact residual at threshold matches analytic expectation ---
  // At z_est = threshold: r_depth = log(threshold / scaled_prior)
  // weight = 1 / (sigma_depth / depth_prior) = depth_prior / sigma_depth
  {
    const double r_at_threshold = eval(threshold);
    const double expected_weight = depth_prior / sigma_depth;
    const double expected_r_depth = std::log(threshold / (1.0 * depth_prior));
    EXPECT_NEAR(r_at_threshold, expected_weight * expected_r_depth, 1e-10);
  }
}

// ---- Backward-compat gate: use_lc_observations.
//
// Default-constructed ``GlobalPositionerOptions`` keeps the gate ``false``
// so vanilla pycolmap callers get vanilla colmap4 GP behaviour even when
// ``track.lc_elements`` carry non-empty values left over from upstream code
// paths. These tests verify both directions of the gate by counting BATA
// residual blocks added during ``Solve``.

void StampGtDepthPriors(Reconstruction& reconstruction);

namespace {

// Test-only subclass exposing ``scales_`` (one entry per BATA residual
// block added by ``AddObservationToProblem``) so we can count residuals
// without instrumenting the production class.
class TestableGlobalPositioner : public GlobalPositioner {
 public:
  using GlobalPositioner::GlobalPositioner;
  size_t NumScales() const { return scales_.size(); }
  size_t NumFrameCenters() const { return frame_centers_.size(); }
  size_t NumReplayEntries() const {
    return ResidualReplayEntriesForTest().size();
  }
  const std::vector<GlobalPositioningResidualReplayEntry>&
  ResidualReplayEntries() const {
    return ResidualReplayEntriesForTest();
  }
  void SetupOnlyForTest(const PoseGraph& pose_graph,
                        Reconstruction& reconstruction) {
    SetupProblem(pose_graph, reconstruction);
    InitializeRandomPositions(pose_graph, reconstruction);
    AddPointToCameraConstraints(reconstruction);
  }
};

// Build a small synthetic GP problem with rigs of a single ref camera so
// every observation routes through the BATA branch in
// ``AddObservationToProblem``. Returns a freshly loaded ``Reconstruction``
// with rotations from GT and zero translations (mirrors the nominal test).
struct GpTestData {
  std::shared_ptr<Database> database;
  Reconstruction gt_reconstruction;
  Reconstruction reconstruction;
  PoseGraph pose_graph;
  DatabaseCache database_cache;
};

GpTestData BuildGpTestData(const int num_rigs = 1,
                           const int num_cameras_per_rig = 1,
                           const int num_frames_per_rig = 5,
                           const int num_points3D = 30) {
  GpTestData data;
  const auto database_path = CreateTestDir() / "database.db";
  data.database = Database::Open(database_path);

  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = num_rigs;
  synthetic_dataset_options.num_cameras_per_rig = num_cameras_per_rig;
  synthetic_dataset_options.num_frames_per_rig = num_frames_per_rig;
  synthetic_dataset_options.num_points3D = num_points3D;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &data.gt_reconstruction, data.database.get());

  DatabaseCache::Options cache_options;
  data.database_cache.Load(*data.database, cache_options);

  data.pose_graph.Load(*data.database_cache.CorrespondenceGraph());

  data.reconstruction = data.gt_reconstruction;
  for (const auto& [frame_id, _] : data.reconstruction.Frames()) {
    Frame& frame = data.reconstruction.Frame(frame_id);
    frame.SetRigFromWorld(
        Rigid3d(frame.RigFromWorld().rotation(), Eigen::Vector3d::Zero()));
  }
  return data;
}

void ForceNonRefRigTranslationsUnknownForTest(Reconstruction& reconstruction) {
  std::vector<rig_t> rig_ids;
  rig_ids.reserve(reconstruction.NumRigs());
  for (const auto& [rig_id, _] : reconstruction.Rigs()) {
    rig_ids.push_back(rig_id);
  }

  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (const rig_t rig_id : rig_ids) {
    Rig& rig = reconstruction.Rig(rig_id);
    for (auto& [_, sensor_from_rig] : rig.NonRefSensors()) {
      ASSERT_TRUE(sensor_from_rig.has_value());
      sensor_from_rig->translation() = Eigen::Vector3d(nan, nan, nan);
    }
  }
}

GlobalPositionerOptions BaselineGpOptions() {
  GlobalPositionerOptions options;
  options.use_gpu = false;
  options.random_seed = 42;
  options.solver_options.minimizer_progress_to_stdout = false;
  // Solve with a single iteration: we only care about residual blocks
  // attached during setup, not convergence.
  options.solver_options.max_num_iterations = 1;
  return options;
}

std::string ReadFileForTest(const std::filesystem::path& path) {
  std::ifstream file(path);
  THROW_CHECK_FILE_OPEN(file, path);
  return std::string((std::istreambuf_iterator<char>(file)),
                     std::istreambuf_iterator<char>());
}

std::string ReadBinaryPrefixForTest(const std::filesystem::path& path,
                                    const size_t size) {
  std::ifstream file(path, std::ios::binary);
  THROW_CHECK_FILE_OPEN(file, path);
  std::string prefix(size, '\0');
  file.read(prefix.data(), static_cast<std::streamsize>(size));
  THROW_CHECK_EQ(static_cast<size_t>(file.gcount()), size)
      << "Could not read binary prefix from " << path;
  return prefix;
}

template <typename T>
T ReadRawLittleEndianForTest(std::ifstream& file,
                             const std::filesystem::path& path) {
  const T value = ReadBinaryLittleEndian<T>(&file);
  THROW_CHECK(file.good()) << "Could not read raw binary trace value from "
                           << path;
  return value;
}

std::string ReadRawStringForTest(std::ifstream& file,
                                 const std::filesystem::path& path) {
  const uint32_t size = ReadRawLittleEndianForTest<uint32_t>(file, path);
  std::string value(size, '\0');
  if (size > 0) {
    file.read(value.data(), static_cast<std::streamsize>(size));
    THROW_CHECK(file.good())
        << "Could not read raw binary trace string from " << path;
  }
  return value;
}

uint64_t ReadRawLedgerRecordCountForTest(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  THROW_CHECK_FILE_OPEN(file, path);
  EXPECT_EQ(ReadBinaryPrefixForTest(path, 8), "GPTRLGR1");
  file.seekg(8, std::ios::beg);
  EXPECT_EQ(ReadRawLittleEndianForTest<uint32_t>(file, path), 1);
  return ReadRawLittleEndianForTest<uint64_t>(file, path);
}

std::pair<uint64_t, uint64_t> ReadRawArrayHeaderForTest(
    const std::filesystem::path& path, const std::string& expected_name) {
  std::ifstream file(path, std::ios::binary);
  THROW_CHECK_FILE_OPEN(file, path);
  EXPECT_EQ(ReadBinaryPrefixForTest(path, 8), "GPTRARR1");
  file.seekg(8, std::ios::beg);
  EXPECT_EQ(ReadRawLittleEndianForTest<uint32_t>(file, path), 1);
  const uint64_t rows = ReadRawLittleEndianForTest<uint64_t>(file, path);
  const uint64_t cols = ReadRawLittleEndianForTest<uint64_t>(file, path);
  EXPECT_EQ(ReadRawStringForTest(file, path), expected_name);
  return {rows, cols};
}

std::tuple<uint32_t, int64_t, uint64_t, uint64_t, bool>
ReadRawResidualValuesHeaderForTest(const std::filesystem::path& path,
                                   const bool expect_raw_jacobians) {
  std::ifstream file(path, std::ios::binary);
  THROW_CHECK_FILE_OPEN(file, path);
  EXPECT_EQ(ReadBinaryPrefixForTest(path, 8), "GPTRRSV1");
  file.seekg(8, std::ios::beg);
  const uint32_t version = ReadRawLittleEndianForTest<uint32_t>(file, path);
  EXPECT_TRUE(version == 1 || version == 2);
  const int64_t iteration = ReadRawLittleEndianForTest<int64_t>(file, path);
  const uint64_t num_residual_blocks =
      ReadRawLittleEndianForTest<uint64_t>(file, path);
  const uint64_t total_scalar_residuals =
      ReadRawLittleEndianForTest<uint64_t>(file, path);
  char has_loss_rho = 0;
  file.read(&has_loss_rho, 1);
  THROW_CHECK(file.good())
      << "Could not read raw binary residual-values loss-rho flag from "
      << path;
  EXPECT_EQ(has_loss_rho, 1);
  bool has_raw_jacobians = false;
  if (version >= 2) {
    char has_raw_jacobians_byte = 0;
    file.read(&has_raw_jacobians_byte, 1);
    THROW_CHECK(file.good())
        << "Could not read raw binary residual-values Jacobian flag from "
        << path;
    has_raw_jacobians = has_raw_jacobians_byte != 0;
  }
  EXPECT_EQ(has_raw_jacobians, expect_raw_jacobians);
  return {version,
          iteration,
          num_residual_blocks,
          total_scalar_residuals,
          has_raw_jacobians};
}

size_t CountJsonlRecordsForTest(const std::string& text) {
  size_t count = 0;
  std::istringstream stream(text);
  std::string line;
  while (std::getline(stream, line)) {
    if (!line.empty()) {
      ++count;
    }
  }
  return count;
}

bool ContainsJsonStringValueForTest(const std::string& text,
                                    const std::string& key,
                                    const std::string& value) {
  const std::string key_token = "\"" + key + "\"";
  const std::string value_token = "\"" + value + "\"";
  size_t pos = 0;
  while ((pos = text.find(key_token, pos)) != std::string::npos) {
    const size_t colon = text.find(':', pos + key_token.size());
    const size_t line_end = text.find('\n', pos);
    const size_t search_end =
        line_end == std::string::npos ? text.size() : line_end;
    if (colon != std::string::npos && colon < search_end &&
        text.find(value_token, colon) < search_end) {
      return true;
    }
    pos += key_token.size();
  }
  return false;
}

std::optional<std::string> FindJsonlRecordWithStringValueForTest(
    const std::string& text, const std::string& key, const std::string& value) {
  std::istringstream stream(text);
  std::string line;
  while (std::getline(stream, line)) {
    if (ContainsJsonStringValueForTest(line, key, value)) {
      return line;
    }
  }
  return std::nullopt;
}

void ExpectEveryResidualBlockHasReplayLedgerFieldsForTest(
    const std::string& residual_blocks) {
  std::istringstream stream(residual_blocks);
  std::string line;
  size_t count = 0;
  while (std::getline(stream, line)) {
    if (line.empty()) {
      continue;
    }
    ++count;
    EXPECT_NE(line.find("\"replay_schema_version\":1"), std::string::npos)
        << line;
    EXPECT_NE(line.find("\"parameter_blocks\":[{"), std::string::npos) << line;
    EXPECT_NE(line.find("\"role\":"), std::string::npos) << line;
    EXPECT_NE(line.find("\"kind\":"), std::string::npos) << line;
    EXPECT_NE(line.find("\"id\":"), std::string::npos) << line;
    EXPECT_NE(line.find("\"size\":"), std::string::npos) << line;
    EXPECT_NE(line.find("\"loss\":{\"bucket\":"), std::string::npos) << line;
    EXPECT_NE(line.find("\"type\":"), std::string::npos) << line;
    EXPECT_NE(line.find("\"scale\":"), std::string::npos) << line;
    EXPECT_NE(line.find("\"weight\":"), std::string::npos) << line;
    EXPECT_NE(line.find("\"fixed_parameters_status\":\"serialized\""),
              std::string::npos)
        << line;
    EXPECT_NE(line.find("\"fixed_parameters\":{"), std::string::npos) << line;
    EXPECT_EQ(line.find("\"fixed_parameters_status\":"
                        "\"deferred_not_serialized\""),
              std::string::npos)
        << line;
    EXPECT_EQ(line.find("\"fixed_parameters_missing\":["), std::string::npos)
        << line;
    EXPECT_EQ(line.find("\"fixed_parameters_todo\":"), std::string::npos)
        << line;
  }
  EXPECT_GT(count, 0u);
}

std::optional<int64_t> FindEventAttrIntForTest(const std::string& events,
                                               const std::string& event_type,
                                               const std::string& attr_key) {
  std::istringstream stream(events);
  std::string line;
  while (std::getline(stream, line)) {
    if (!ContainsJsonStringValueForTest(line, "event_type", event_type)) {
      continue;
    }
    const std::string key_token = "\"" + attr_key + "\"";
    const size_t key_pos = line.find(key_token);
    if (key_pos == std::string::npos) {
      continue;
    }
    size_t value_begin = line.find(':', key_pos + key_token.size());
    if (value_begin == std::string::npos) {
      continue;
    }
    ++value_begin;
    while (value_begin < line.size() &&
           std::isspace(static_cast<unsigned char>(line[value_begin]))) {
      ++value_begin;
    }
    size_t value_end = value_begin;
    if (value_end < line.size() && line[value_end] == '-') {
      ++value_end;
    }
    while (value_end < line.size() &&
           std::isdigit(static_cast<unsigned char>(line[value_end]))) {
      ++value_end;
    }
    if (value_end > value_begin) {
      return std::stoll(line.substr(value_begin, value_end - value_begin));
    }
  }
  return std::nullopt;
}

std::optional<int64_t> FindJsonIntForTest(const std::string& text,
                                          const std::string& key) {
  const std::string key_token = "\"" + key + "\"";
  const size_t key_pos = text.find(key_token);
  if (key_pos == std::string::npos) {
    return std::nullopt;
  }
  size_t value_begin = text.find(':', key_pos + key_token.size());
  if (value_begin == std::string::npos) {
    return std::nullopt;
  }
  ++value_begin;
  while (value_begin < text.size() &&
         std::isspace(static_cast<unsigned char>(text[value_begin]))) {
    ++value_begin;
  }
  size_t value_end = value_begin;
  if (value_end < text.size() && text[value_end] == '-') {
    ++value_end;
  }
  while (value_end < text.size() &&
         std::isdigit(static_cast<unsigned char>(text[value_end]))) {
    ++value_end;
  }
  if (value_end == value_begin) {
    return std::nullopt;
  }
  return std::stoll(text.substr(value_begin, value_end - value_begin));
}

std::optional<double> FindJsonDoubleForTest(const std::string& text,
                                            const std::string& key) {
  const std::string key_token = "\"" + key + "\"";
  const size_t key_pos = text.find(key_token);
  if (key_pos == std::string::npos) {
    return std::nullopt;
  }
  size_t value_begin = text.find(':', key_pos + key_token.size());
  if (value_begin == std::string::npos) {
    return std::nullopt;
  }
  ++value_begin;
  while (value_begin < text.size() &&
         std::isspace(static_cast<unsigned char>(text[value_begin]))) {
    ++value_begin;
  }
  size_t value_end = value_begin;
  while (value_end < text.size()) {
    const char c = text[value_end];
    if (!(std::isdigit(static_cast<unsigned char>(c)) || c == '-' || c == '+' ||
          c == '.' || c == 'e' || c == 'E')) {
      break;
    }
    ++value_end;
  }
  if (value_end == value_begin) {
    return std::nullopt;
  }
  return std::stod(text.substr(value_begin, value_end - value_begin));
}

std::optional<std::string> FindJsonStringForTest(const std::string& text,
                                                 const std::string& key) {
  const std::string key_token = "\"" + key + "\"";
  const size_t key_pos = text.find(key_token);
  if (key_pos == std::string::npos) {
    return std::nullopt;
  }
  size_t value_begin = text.find(':', key_pos + key_token.size());
  if (value_begin == std::string::npos) {
    return std::nullopt;
  }
  ++value_begin;
  while (value_begin < text.size() &&
         std::isspace(static_cast<unsigned char>(text[value_begin]))) {
    ++value_begin;
  }
  if (value_begin >= text.size() || text[value_begin] != '"') {
    return std::nullopt;
  }
  ++value_begin;
  std::string value;
  for (size_t i = value_begin; i < text.size(); ++i) {
    if (text[i] == '\\' && i + 1 < text.size()) {
      value.push_back(text[i + 1]);
      ++i;
    } else if (text[i] == '"') {
      return value;
    } else {
      value.push_back(text[i]);
    }
  }
  return std::nullopt;
}

std::string ExtractJsonArrayForTest(const std::string& text,
                                    const std::string& key) {
  const std::string key_token = "\"" + key + "\"";
  const size_t key_pos = text.find(key_token);
  THROW_CHECK_NE(key_pos, std::string::npos) << "Missing JSON key: " << key;
  const size_t colon = text.find(':', key_pos + key_token.size());
  THROW_CHECK_NE(colon, std::string::npos) << "Missing JSON colon for: " << key;
  const size_t array_begin = text.find('[', colon);
  THROW_CHECK_NE(array_begin, std::string::npos)
      << "Missing JSON array for: " << key;

  bool in_string = false;
  bool escaped = false;
  int depth = 0;
  for (size_t i = array_begin; i < text.size(); ++i) {
    const char c = text[i];
    if (in_string) {
      if (escaped) {
        escaped = false;
      } else if (c == '\\') {
        escaped = true;
      } else if (c == '"') {
        in_string = false;
      }
      continue;
    }
    if (c == '"') {
      in_string = true;
    } else if (c == '[') {
      ++depth;
    } else if (c == ']') {
      --depth;
      if (depth == 0) {
        return text.substr(array_begin, i - array_begin + 1);
      }
    }
  }
  THROW_CHECK(false) << "Unterminated JSON array for: " << key;
  return "";
}

std::vector<std::string> FindJsonStringArrayForTest(const std::string& text,
                                                    const std::string& key) {
  const std::string array = ExtractJsonArrayForTest(text, key);
  std::vector<std::string> values;
  bool in_string = false;
  bool escaped = false;
  std::string value;
  for (size_t i = 1; i + 1 < array.size(); ++i) {
    const char c = array[i];
    if (!in_string) {
      if (c == '"') {
        in_string = true;
        value.clear();
      }
      continue;
    }
    if (escaped) {
      value.push_back(c);
      escaped = false;
    } else if (c == '\\') {
      escaped = true;
    } else if (c == '"') {
      values.push_back(value);
      in_string = false;
    } else {
      value.push_back(c);
    }
  }
  THROW_CHECK(!in_string) << "Unterminated JSON string array for: " << key;
  return values;
}

std::vector<size_t> FindJsonSizeArrayForTest(const std::string& text,
                                             const std::string& key) {
  const std::string array = ExtractJsonArrayForTest(text, key);
  std::vector<size_t> values;
  size_t pos = 1;
  while (pos + 1 < array.size()) {
    while (pos + 1 < array.size() &&
           !std::isdigit(static_cast<unsigned char>(array[pos]))) {
      ++pos;
    }
    if (pos + 1 >= array.size()) {
      break;
    }
    size_t end = pos;
    while (end < array.size() &&
           std::isdigit(static_cast<unsigned char>(array[end]))) {
      ++end;
    }
    values.push_back(
        static_cast<size_t>(std::stoull(array.substr(pos, end - pos))));
    pos = end;
  }
  return values;
}

std::vector<bool> FindJsonBoolArrayForTest(const std::string& text,
                                           const std::string& key) {
  const std::string array = ExtractJsonArrayForTest(text, key);
  std::vector<bool> values;
  size_t pos = 1;
  while (pos + 1 < array.size()) {
    while (pos + 1 < array.size() &&
           std::isspace(static_cast<unsigned char>(array[pos]))) {
      ++pos;
    }
    if (array.compare(pos, 4, "true") == 0) {
      values.push_back(true);
      pos += 4;
    } else if (array.compare(pos, 5, "false") == 0) {
      values.push_back(false);
      pos += 5;
    } else {
      ++pos;
    }
  }
  return values;
}

std::vector<std::string> ResidualIdsFromBlocksJsonlForTest(
    const std::string& residual_blocks) {
  std::vector<std::string> residual_ids;
  std::istringstream stream(residual_blocks);
  std::string line;
  while (std::getline(stream, line)) {
    if (line.empty()) {
      continue;
    }
    std::optional<std::string> residual_id =
        FindJsonStringForTest(line, "residual_id");
    THROW_CHECK(residual_id.has_value())
        << "Residual block record missing residual_id: " << line;
    residual_ids.push_back(*residual_id);
  }
  return residual_ids;
}

std::optional<double> FindIterationMetricDoubleForTest(
    const std::string& iteration_metrics,
    const int64_t iteration,
    const std::string& key) {
  std::istringstream stream(iteration_metrics);
  std::string line;
  while (std::getline(stream, line)) {
    if (!ContainsJsonStringValueForTest(
            line, "event_type", "ceres_iteration")) {
      continue;
    }
    const std::optional<int64_t> line_iteration =
        FindJsonIntForTest(line, "iteration");
    if (!line_iteration.has_value() || *line_iteration != iteration) {
      continue;
    }
    return FindJsonDoubleForTest(line, key);
  }
  return std::nullopt;
}

std::vector<double> ReadDoubleSidecarForTest(const std::filesystem::path& path,
                                             const size_t expected_count) {
  std::ifstream file(path, std::ios::binary);
  THROW_CHECK_FILE_OPEN(file, path);
  std::vector<double> values(expected_count);
  ReadBinaryLittleEndian<double>(&file, &values);
  THROW_CHECK(file.good() || file.eof())
      << "Failed while reading binary sidecar: " << path;
  return values;
}

void ExpectContainsAllSnapshotMetadataKeysForTest(const std::string& metadata) {
  for (const char* key : {"\"iteration\"",
                          "\"frame_ids\"",
                          "\"frame_centers_world_shape\"",
                          "\"point3D_ids\"",
                          "\"points3D_world_shape\"",
                          "\"bata_residual_ids\"",
                          "\"bata_scales_shape\"",
                          "\"dmap_image_ids\"",
                          "\"dmap_scales_stored_shape\"",
                          "\"coordinate_convention\""}) {
    EXPECT_NE(metadata.find(key), std::string::npos)
        << "Missing snapshot metadata key: " << key;
  }
}

std::optional<TrackElement> FindFirstValidDepthObservationForTest(
    const Reconstruction& reconstruction) {
  for (const auto& [_, point3D] : reconstruction.Points3D()) {
    for (const TrackElement& observation : point3D.track.Elements()) {
      const Image& image = reconstruction.Image(observation.image_id);
      if (observation.point2D_idx < image.depth_prior_validity.size() &&
          image.depth_prior_validity[observation.point2D_idx]) {
        return observation;
      }
    }
  }
  return std::nullopt;
}

// Sum of valid track elements across tracks long enough to be added by
// ``AddPointToCameraConstraints`` (i.e. ``track.Length() >=
// min_num_view_per_track``).
size_t CountValidObservations(const Reconstruction& reconstruction,
                              int min_num_view_per_track) {
  size_t total = 0;
  for (const auto& [_, point3D] : reconstruction.Points3D()) {
    if (static_cast<int>(point3D.track.Length()) < min_num_view_per_track) {
      continue;
    }
    total += point3D.track.Length();
  }
  return total;
}

// Sum of LC elements in tracks long enough to participate in GP.
size_t CountLcObservations(const Reconstruction& reconstruction,
                           int min_num_view_per_track) {
  size_t total = 0;
  for (const auto& [_, point3D] : reconstruction.Points3D()) {
    if (static_cast<int>(point3D.track.Length()) < min_num_view_per_track) {
      continue;
    }
    total += point3D.track.lc_elements.size();
  }
  return total;
}

// Copy regular track elements into ``lc_elements`` for every track that
// participates in GP and return the total LC elements added.
size_t DuplicateElementsAsLc(Reconstruction& reconstruction,
                             int min_num_view_per_track) {
  size_t total_lc = 0;
  std::vector<point3D_t> point3D_ids;
  point3D_ids.reserve(reconstruction.NumPoints3D());
  for (const auto& [point3D_id, _] : reconstruction.Points3D()) {
    point3D_ids.push_back(point3D_id);
  }
  for (point3D_t point3D_id : point3D_ids) {
    Point3D& point3D = reconstruction.Point3D(point3D_id);
    if (static_cast<int>(point3D.track.Length()) < min_num_view_per_track) {
      continue;
    }
    point3D.track.lc_elements = point3D.track.Elements();
    total_lc += point3D.track.lc_elements.size();
  }
  return total_lc;
}

// Keep exactly one point with one regular observation plus two LC observations.
// LC residuals may augment admitted points, but they must not satisfy the
// min-view admission gate.
void KeepSingleRegularPlusLcPoint(Reconstruction& reconstruction) {
  std::vector<point3D_t> point3D_ids;
  point3D_ids.reserve(reconstruction.NumPoints3D());
  for (const point3D_t point3D_id : reconstruction.Point3DIds()) {
    point3D_ids.push_back(point3D_id);
  }
  for (const point3D_t point3D_id : point3D_ids) {
    reconstruction.DeletePoint3D(point3D_id);
  }

  const std::vector<image_t> image_ids = reconstruction.RegImageIds();
  THROW_CHECK_GE(image_ids.size(), 3);
  THROW_CHECK_GT(reconstruction.Image(image_ids[0]).NumPoints2D(), 0);
  THROW_CHECK_GT(reconstruction.Image(image_ids[1]).NumPoints2D(), 0);
  THROW_CHECK_GT(reconstruction.Image(image_ids[2]).NumPoints2D(), 0);

  Point3D point3D;
  point3D.xyz = Eigen::Vector3d(0.1, 0.2, 4.0);
  point3D.track.AddElement(image_ids[0], 0);
  point3D.track.lc_elements.emplace_back(image_ids[1], 0);
  point3D.track.lc_elements.emplace_back(image_ids[2], 0);
  reconstruction.AddPoint3D(0, std::move(point3D));
}

}  // namespace

TEST(GlobalPositioning, DefaultTraceDisabledWritesNoFiles) {
  SetPRNGSeed(0);
  GpTestData data = BuildGpTestData();
  const std::filesystem::path trace_dir = CreateTestDir() / "gp_trace";

  GlobalPositionerOptions options = BaselineGpOptions();
  ASSERT_EQ(options.trace.level, GlobalPositioningTraceLevel::kOff);

  TestableGlobalPositioner positioner(options);
  ASSERT_TRUE(positioner.Solve(data.pose_graph, data.reconstruction));
  EXPECT_FALSE(ExistsPath(trace_dir));
}

TEST(GlobalPositioning, SummaryTraceWritesLifecycleAndIterationFiles) {
  SetPRNGSeed(0);
  GpTestData data = BuildGpTestData();
  const std::filesystem::path trace_dir = CreateTestDir() / "gp_trace";
  ASSERT_TRUE(std::filesystem::create_directories(trace_dir / "snapshots"));
  ASSERT_TRUE(
      std::filesystem::create_directories(trace_dir / "residual_values"));
  {
    std::ofstream residual_blocks(trace_dir / "residual_blocks.jsonl");
    ASSERT_TRUE(residual_blocks.is_open());
    residual_blocks << "stale";
    std::ofstream residual_skips(trace_dir / "residual_skips.jsonl");
    ASSERT_TRUE(residual_skips.is_open());
    residual_skips << "stale";
    std::ofstream snapshot(trace_dir / "snapshots" / "iter_000000.json");
    ASSERT_TRUE(snapshot.is_open());
    snapshot << "stale";
    std::ofstream residual_values(trace_dir / "residual_values" /
                                  "iter_000000.json");
    ASSERT_TRUE(residual_values.is_open());
    residual_values << "stale";
  }

  GlobalPositionerOptions options = BaselineGpOptions();
  options.trace.level = GlobalPositioningTraceLevel::kSummary;
  options.trace.output_path = trace_dir;
  options.trace.run_label = "gp_test";

  TestableGlobalPositioner positioner(options);
  ASSERT_TRUE(positioner.Solve(data.pose_graph, data.reconstruction));

  const std::filesystem::path manifest_path = trace_dir / "manifest.json";
  const std::filesystem::path events_path = trace_dir / "events.jsonl";
  const std::filesystem::path iteration_path =
      trace_dir / "iteration_metrics.jsonl";
  ASSERT_TRUE(ExistsFile(manifest_path));
  ASSERT_TRUE(ExistsFile(events_path));
  ASSERT_TRUE(ExistsFile(iteration_path));
  EXPECT_FALSE(ExistsFile(trace_dir / "residual_blocks.jsonl"));
  EXPECT_FALSE(ExistsFile(trace_dir / "residual_skips.jsonl"));
  EXPECT_FALSE(ExistsDir(trace_dir / "snapshots"));
  EXPECT_FALSE(ExistsDir(trace_dir / "residual_values"));

  const std::string manifest = ReadFileForTest(manifest_path);
  EXPECT_NE(manifest.find("\"status\": \"finished\""), std::string::npos);
  EXPECT_NE(manifest.find("\"trace_level\": \"summary\""), std::string::npos);

  const std::string events = ReadFileForTest(events_path);
  EXPECT_NE(events.find("\"event_type\":\"run_started\""), std::string::npos);
  EXPECT_NE(events.find("\"event_type\":\"problem_built\""), std::string::npos);
  EXPECT_NE(events.find("\"event_type\":\"solve_started\""), std::string::npos);
  EXPECT_NE(events.find("\"event_type\":\"solve_finished\""),
            std::string::npos);
  EXPECT_NE(events.find("\"event_type\":\"results_converted\""),
            std::string::npos);
  EXPECT_NE(events.find("\"num_residual_blocks\""), std::string::npos);

  const std::string iterations = ReadFileForTest(iteration_path);
  EXPECT_NE(iterations.find("\"event_type\":\"ceres_iteration\""),
            std::string::npos);
  EXPECT_NE(iterations.find("\"cost\""), std::string::npos);
  EXPECT_NE(iterations.find("\"gradient_max_norm\""), std::string::npos);
}

TEST(GlobalPositioning, TraceOutputPathFileFailsLoudly) {
  SetPRNGSeed(0);
  GpTestData data = BuildGpTestData();
  const std::filesystem::path trace_path = CreateTestDir() / "gp_trace_file";
  {
    std::ofstream file(trace_path);
    ASSERT_TRUE(file.is_open());
    file << "not a directory";
  }

  GlobalPositionerOptions options = BaselineGpOptions();
  options.trace.level = GlobalPositioningTraceLevel::kSummary;
  options.trace.output_path = trace_path;

  TestableGlobalPositioner positioner(options);
  EXPECT_ANY_THROW(positioner.Solve(data.pose_graph, data.reconstruction));
}

TEST(GlobalPositioning, ParameterSnapshotsTraceWritesMetadataAndSidecars) {
  SetPRNGSeed(0);
  GpTestData data = BuildGpTestData();
  const std::filesystem::path trace_dir = CreateTestDir() / "gp_snapshots";

  GlobalPositionerOptions options = BaselineGpOptions();
  options.trace.level = GlobalPositioningTraceLevel::kParameterSnapshots;
  options.trace.output_path = trace_dir;
  options.trace.snapshot_every_n_iterations = 1;
  options.trace.max_snapshotted_points = 2;

  TestableGlobalPositioner positioner(options);
  ASSERT_TRUE(positioner.Solve(data.pose_graph, data.reconstruction));

  const std::filesystem::path snapshot_dir = trace_dir / "snapshots";
  ASSERT_TRUE(ExistsDir(snapshot_dir));
  size_t snapshot_metadata_count = 0;
  for (const std::filesystem::directory_entry& entry :
       std::filesystem::directory_iterator(snapshot_dir)) {
    if (entry.path().extension() == ".json") {
      ++snapshot_metadata_count;
    }
  }
  EXPECT_EQ(snapshot_metadata_count,
            CountJsonlRecordsForTest(
                ReadFileForTest(trace_dir / "iteration_metrics.jsonl")));

  const std::filesystem::path metadata_path = snapshot_dir / "iter_000000.json";
  const std::filesystem::path frame_centers_path =
      snapshot_dir / "iter_000000_frame_centers_f64.bin";
  const std::filesystem::path points3D_path =
      snapshot_dir / "iter_000000_points3D_f64.bin";
  const std::filesystem::path scales_path =
      snapshot_dir / "iter_000000_scales_f64.bin";
  ASSERT_TRUE(ExistsFile(metadata_path));
  ASSERT_TRUE(ExistsFile(frame_centers_path));
  ASSERT_TRUE(ExistsFile(points3D_path));
  ASSERT_TRUE(ExistsFile(scales_path));

  const std::string metadata = ReadFileForTest(metadata_path);
  ExpectContainsAllSnapshotMetadataKeysForTest(metadata);
  EXPECT_NE(metadata.find("iter_000000_frame_centers_f64.bin"),
            std::string::npos);
  EXPECT_NE(metadata.find("iter_000000_points3D_f64.bin"), std::string::npos);
  EXPECT_NE(metadata.find("iter_000000_scales_f64.bin"), std::string::npos);
  EXPECT_NE(metadata.find("world"), std::string::npos);
  EXPECT_NE(metadata.find("cam_from_world.translation"), std::string::npos);

  EXPECT_EQ(std::filesystem::file_size(frame_centers_path),
            positioner.NumFrameCenters() * 3 * sizeof(double));

  const uintmax_t points3D_size = std::filesystem::file_size(points3D_path);
  EXPECT_LE(points3D_size, 2u * 3u * sizeof(double));
  EXPECT_EQ(points3D_size % (3u * sizeof(double)), 0u);

  const uintmax_t scales_size = std::filesystem::file_size(scales_path);
  EXPECT_GT(scales_size, 0u);
  EXPECT_EQ(scales_size % sizeof(double), 0u);

  const std::filesystem::path raw_binary_dir = trace_dir / "raw_binary";
  ASSERT_TRUE(ExistsFile(raw_binary_dir / "manifest.json"));
  ASSERT_TRUE(ExistsFile(raw_binary_dir / "static" / "residual_ledger.bin"));
  ASSERT_TRUE(ExistsFile(raw_binary_dir / "iterations" / "iter_000000" /
                         "frame_centers.bin"));
  ASSERT_TRUE(ExistsFile(raw_binary_dir / "iterations" / "iter_000000" /
                         "point_xyz.bin"));
  ASSERT_TRUE(
      ExistsFile(raw_binary_dir / "iterations" / "iter_000000" / "scales.bin"));
  EXPECT_EQ(ReadRawLedgerRecordCountForTest(raw_binary_dir / "static" /
                                            "residual_ledger.bin"),
            CountJsonlRecordsForTest(
                ReadFileForTest(trace_dir / "residual_blocks.jsonl")));
  EXPECT_EQ(ReadRawArrayHeaderForTest(raw_binary_dir / "iterations" /
                                          "iter_000000" / "frame_centers.bin",
                                      "frame_centers"),
            std::make_pair(static_cast<uint64_t>(positioner.NumFrameCenters()),
                           uint64_t{3}));
  EXPECT_EQ(ReadRawArrayHeaderForTest(
                raw_binary_dir / "iterations" / "iter_000000" / "point_xyz.bin",
                "point_xyz")
                .second,
            uint64_t{3});
  EXPECT_GT(ReadRawArrayHeaderForTest(
                raw_binary_dir / "iterations" / "iter_000000" / "scales.bin",
                "scales")
                .first,
            uint64_t{0});

  const std::string raw_manifest =
      ReadFileForTest(raw_binary_dir / "manifest.json");
  EXPECT_NE(raw_manifest.find("\"storage_format\": "
                              "\"global_positioning_raw_binary_v1\""),
            std::string::npos);
  EXPECT_NE(raw_manifest.find("\"residual_ledger\": "
                              "\"static/residual_ledger.bin\""),
            std::string::npos);
  EXPECT_NE(raw_manifest.find("\"frame_centers\": \"frame_centers.bin\""),
            std::string::npos);
  EXPECT_NE(raw_manifest.find("\"point_xyz\": \"point_xyz.bin\""),
            std::string::npos);
}

TEST(GlobalPositioning, ResidualValuesTraceWritesMetadataAndSidecars) {
  SetPRNGSeed(0);
  GpTestData data = BuildGpTestData();
  StampGtDepthPriors(data.reconstruction);
  const std::filesystem::path trace_dir =
      CreateTestDir() / "gp_residual_values";

  GlobalPositionerOptions options = BaselineGpOptions();
  options.use_metric_depth_constraint = true;
  options.trace.level = GlobalPositioningTraceLevel::kResidualValues;
  options.trace.output_path = trace_dir;
  options.trace.snapshot_every_n_iterations = 1;
  options.trace.max_snapshotted_points = -1;

  TestableGlobalPositioner positioner(options);
  ASSERT_TRUE(positioner.Solve(data.pose_graph, data.reconstruction));
  EXPECT_GT(positioner.NumReplayEntries(), 0u);

  const std::filesystem::path residual_values_dir =
      trace_dir / "residual_values";
  ASSERT_TRUE(ExistsDir(residual_values_dir));

  const std::filesystem::path metadata_path =
      residual_values_dir / "iter_000000.json";
  const std::filesystem::path raw_residuals_path =
      residual_values_dir / "iter_000000_raw_residuals_f64.bin";
  const std::filesystem::path raw_costs_path =
      residual_values_dir / "iter_000000_raw_costs_f64.bin";
  const std::filesystem::path robust_costs_path =
      residual_values_dir / "iter_000000_robust_costs_f64.bin";
  const std::filesystem::path loss_rho_values_path =
      residual_values_dir / "iter_000000_loss_rho_values_f64.bin";
  const std::filesystem::path raw_jacobians_path =
      residual_values_dir / "iter_000000_raw_jacobians_f64.bin";
  ASSERT_TRUE(ExistsFile(metadata_path));
  ASSERT_TRUE(ExistsFile(raw_residuals_path));
  ASSERT_TRUE(ExistsFile(raw_costs_path));
  ASSERT_TRUE(ExistsFile(robust_costs_path));
  ASSERT_TRUE(ExistsFile(loss_rho_values_path));
  ASSERT_FALSE(ExistsFile(raw_jacobians_path));

  const std::string manifest = ReadFileForTest(trace_dir / "manifest.json");
  EXPECT_NE(manifest.find("\"trace_level\": \"residual_values\""),
            std::string::npos);

  const std::string metadata = ReadFileForTest(metadata_path);
  for (const char* key : {"\"num_residual_blocks\"",
                          "\"total_scalar_residuals\"",
                          "\"has_raw_jacobians\"",
                          "\"residual_ids\"",
                          "\"residual_dims\"",
                          "\"residual_offsets\"",
                          "\"evaluation_success\"",
                          "\"loss_rho_layout\"",
                          "\"artifacts\""}) {
    EXPECT_NE(metadata.find(key), std::string::npos)
        << "Missing residual-values metadata key: " << key;
  }
  EXPECT_NE(metadata.find("iter_000000_raw_residuals_f64.bin"),
            std::string::npos);
  EXPECT_NE(metadata.find("iter_000000_raw_costs_f64.bin"), std::string::npos);
  EXPECT_NE(metadata.find("iter_000000_robust_costs_f64.bin"),
            std::string::npos);
  EXPECT_NE(metadata.find("iter_000000_loss_rho_values_f64.bin"),
            std::string::npos);
  EXPECT_NE(metadata.find("\"loss_rho_values\": {"), std::string::npos);
  EXPECT_NE(metadata.find("\"shape\": ["), std::string::npos);
  EXPECT_NE(metadata.find("\"residual_block_major/rho0_rho1_rho2\""),
            std::string::npos);
  EXPECT_NE(metadata.find("\"has_raw_jacobians\": false"), std::string::npos);
  EXPECT_EQ(metadata.find("\"raw_jacobians\": {"), std::string::npos);
  EXPECT_EQ(metadata.find("iter_000000_raw_jacobians_f64.bin"),
            std::string::npos);

  const std::optional<int64_t> num_residual_blocks =
      FindJsonIntForTest(metadata, "num_residual_blocks");
  const std::optional<int64_t> total_scalar_residuals =
      FindJsonIntForTest(metadata, "total_scalar_residuals");
  ASSERT_TRUE(num_residual_blocks.has_value());
  ASSERT_TRUE(total_scalar_residuals.has_value());
  ASSERT_GT(*num_residual_blocks, 0);
  ASSERT_GE(*total_scalar_residuals, *num_residual_blocks);
  ASSERT_EQ(positioner.NumReplayEntries(),
            static_cast<size_t>(*num_residual_blocks));

  const std::string residual_blocks =
      ReadFileForTest(trace_dir / "residual_blocks.jsonl");
  EXPECT_EQ(CountJsonlRecordsForTest(residual_blocks),
            static_cast<size_t>(*num_residual_blocks));
  const std::vector<std::string> block_residual_ids =
      ResidualIdsFromBlocksJsonlForTest(residual_blocks);
  const std::vector<std::string> metadata_residual_ids =
      FindJsonStringArrayForTest(metadata, "residual_ids");
  EXPECT_EQ(metadata_residual_ids, block_residual_ids);

  std::vector<std::string> replay_residual_ids;
  replay_residual_ids.reserve(positioner.ResidualReplayEntries().size());
  for (const GlobalPositioningResidualReplayEntry& entry :
       positioner.ResidualReplayEntries()) {
    EXPECT_FALSE(entry.residual_id.empty());
    ASSERT_NE(entry.cost_function, nullptr) << entry.residual_id;
    EXPECT_GT(entry.residual_dimension, 0u) << entry.residual_id;
    EXPECT_EQ(entry.residual_dimension,
              static_cast<size_t>(entry.cost_function->num_residuals()))
        << entry.residual_id;
    EXPECT_EQ(entry.parameter_blocks.size(),
              entry.cost_function->parameter_block_sizes().size())
        << entry.residual_id;
    for (const double* parameter_block : entry.parameter_blocks) {
      EXPECT_NE(parameter_block, nullptr) << entry.residual_id;
    }
    replay_residual_ids.push_back(entry.residual_id);
  }
  EXPECT_EQ(replay_residual_ids, metadata_residual_ids);

  const std::vector<size_t> residual_dims =
      FindJsonSizeArrayForTest(metadata, "residual_dims");
  const std::vector<size_t> residual_offsets =
      FindJsonSizeArrayForTest(metadata, "residual_offsets");
  const std::vector<bool> evaluation_success =
      FindJsonBoolArrayForTest(metadata, "evaluation_success");
  ASSERT_EQ(residual_dims.size(), static_cast<size_t>(*num_residual_blocks));
  ASSERT_EQ(residual_offsets.size(), static_cast<size_t>(*num_residual_blocks));
  ASSERT_EQ(evaluation_success.size(),
            static_cast<size_t>(*num_residual_blocks));
  EXPECT_TRUE(std::all_of(evaluation_success.begin(),
                          evaluation_success.end(),
                          [](const bool success) { return success; }));

  EXPECT_EQ(std::filesystem::file_size(raw_residuals_path),
            static_cast<uintmax_t>(*total_scalar_residuals) * sizeof(double));
  EXPECT_EQ(std::filesystem::file_size(raw_costs_path),
            static_cast<uintmax_t>(*num_residual_blocks) * sizeof(double));
  EXPECT_EQ(std::filesystem::file_size(robust_costs_path),
            static_cast<uintmax_t>(*num_residual_blocks) * sizeof(double));
  EXPECT_EQ(std::filesystem::file_size(loss_rho_values_path),
            static_cast<uintmax_t>(*num_residual_blocks) * 3u * sizeof(double));

  const std::filesystem::path raw_binary_dir = trace_dir / "raw_binary";
  ASSERT_TRUE(ExistsFile(raw_binary_dir / "manifest.json"));
  ASSERT_TRUE(ExistsFile(raw_binary_dir / "static" / "residual_ledger.bin"));
  ASSERT_TRUE(ExistsFile(raw_binary_dir / "iterations" / "iter_000000" /
                         "residual_values.bin"));
  EXPECT_EQ(ReadRawLedgerRecordCountForTest(raw_binary_dir / "static" /
                                            "residual_ledger.bin"),
            static_cast<uint64_t>(*num_residual_blocks));
  EXPECT_EQ(
      ReadRawResidualValuesHeaderForTest(
          raw_binary_dir / "iterations" / "iter_000000" / "residual_values.bin",
          /*expect_raw_jacobians=*/false),
      std::make_tuple(uint32_t{1},
                      int64_t{0},
                      static_cast<uint64_t>(*num_residual_blocks),
                      static_cast<uint64_t>(*total_scalar_residuals),
                      false));
  const std::string raw_manifest =
      ReadFileForTest(raw_binary_dir / "manifest.json");
  EXPECT_NE(raw_manifest.find("\"residual_values\": "
                              "\"residual_values.bin\""),
            std::string::npos);

  const std::vector<double> raw_residuals = ReadDoubleSidecarForTest(
      raw_residuals_path, static_cast<size_t>(*total_scalar_residuals));
  const std::vector<double> raw_costs = ReadDoubleSidecarForTest(
      raw_costs_path, static_cast<size_t>(*num_residual_blocks));
  const std::vector<double> robust_costs = ReadDoubleSidecarForTest(
      robust_costs_path, static_cast<size_t>(*num_residual_blocks));
  const std::vector<double> loss_rho_values = ReadDoubleSidecarForTest(
      loss_rho_values_path, static_cast<size_t>(*num_residual_blocks) * 3u);

  size_t expected_offset = 0;
  double robust_cost_sum = 0.0;
  for (size_t i = 0; i < residual_dims.size(); ++i) {
    EXPECT_EQ(residual_offsets[i], expected_offset);
    ASSERT_LE(residual_offsets[i] + residual_dims[i], raw_residuals.size());

    double squared_norm = 0.0;
    for (size_t j = 0; j < residual_dims[i]; ++j) {
      const double residual = raw_residuals[residual_offsets[i] + j];
      ASSERT_TRUE(std::isfinite(residual));
      squared_norm += residual * residual;
    }
    ASSERT_TRUE(std::isfinite(raw_costs[i]));
    ASSERT_TRUE(std::isfinite(robust_costs[i]));
    ASSERT_TRUE(std::isfinite(loss_rho_values[3 * i]));
    ASSERT_TRUE(std::isfinite(loss_rho_values[3 * i + 1]));
    ASSERT_TRUE(std::isfinite(loss_rho_values[3 * i + 2]));
    EXPECT_NEAR(raw_costs[i], 0.5 * squared_norm, 1e-12)
        << metadata_residual_ids[i];
    EXPECT_NEAR(robust_costs[i], 0.5 * loss_rho_values[3 * i], 1e-12)
        << metadata_residual_ids[i];

    robust_cost_sum += robust_costs[i];
    expected_offset += residual_dims[i];
  }
  EXPECT_EQ(expected_offset, static_cast<size_t>(*total_scalar_residuals));

  const std::optional<int64_t> iteration =
      FindJsonIntForTest(metadata, "iteration");
  ASSERT_TRUE(iteration.has_value());
  const std::optional<double> ceres_iteration_cost =
      FindIterationMetricDoubleForTest(
          ReadFileForTest(trace_dir / "iteration_metrics.jsonl"),
          *iteration,
          "cost");
  ASSERT_TRUE(ceres_iteration_cost.has_value());
  EXPECT_NEAR(robust_cost_sum,
              *ceres_iteration_cost,
              std::max(1e-9, std::abs(*ceres_iteration_cost) * 1e-12));
}

TEST(GlobalPositioning, ResidualJacobiansTraceWritesMetadataAndSidecars) {
  SetPRNGSeed(0);
  GpTestData data = BuildGpTestData();
  StampGtDepthPriors(data.reconstruction);
  const std::filesystem::path trace_dir =
      CreateTestDir() / "gp_residual_jacobians";

  GlobalPositionerOptions options = BaselineGpOptions();
  options.use_metric_depth_constraint = true;
  options.trace.level = GlobalPositioningTraceLevel::kResidualJacobians;
  options.trace.output_path = trace_dir;
  options.trace.snapshot_every_n_iterations = 1;
  options.trace.max_snapshotted_points = -1;

  TestableGlobalPositioner positioner(options);
  ASSERT_TRUE(positioner.Solve(data.pose_graph, data.reconstruction));
  EXPECT_GT(positioner.NumReplayEntries(), 0u);

  const std::filesystem::path residual_values_dir =
      trace_dir / "residual_values";
  ASSERT_TRUE(ExistsDir(residual_values_dir));

  const std::filesystem::path metadata_path =
      residual_values_dir / "iter_000000.json";
  const std::filesystem::path raw_jacobians_path =
      residual_values_dir / "iter_000000_raw_jacobians_f64.bin";
  ASSERT_TRUE(ExistsFile(metadata_path));
  ASSERT_TRUE(ExistsFile(raw_jacobians_path));

  const std::string manifest = ReadFileForTest(trace_dir / "manifest.json");
  EXPECT_NE(manifest.find("\"trace_level\": \"residual_jacobians\""),
            std::string::npos);

  const std::string metadata = ReadFileForTest(metadata_path);
  for (const char* key : {"\"has_raw_jacobians\"",
                          "\"total_jacobian_scalars\"",
                          "\"parameter_block_sizes\"",
                          "\"raw_jacobian_offsets\"",
                          "\"parameter_blocks\"",
                          "\"parameter_block_is_constant\"",
                          "\"parameter_block_lower_bounds\"",
                          "\"raw_jacobian_layout\"",
                          "\"jacobian_domain\"",
                          "\"loss_applied_to_jacobians\"",
                          "\"manifold_applied_to_jacobians\"",
                          "\"constant_parameter_blocks_included\"",
                          "\"raw_jacobians\""}) {
    EXPECT_NE(metadata.find(key), std::string::npos)
        << "Missing residual-jacobians metadata key: " << key;
  }
  EXPECT_NE(metadata.find("\"has_raw_jacobians\": true"), std::string::npos);
  EXPECT_NE(
      metadata.find("\"raw_jacobian_layout\": "
                    "\"residual_block_major/parameter_block_major/row_major\""),
      std::string::npos);
  EXPECT_NE(metadata.find("\"jacobian_domain\": "
                          "\"raw_cost_function_ambient_parameters\""),
            std::string::npos);
  EXPECT_NE(metadata.find("\"loss_applied_to_jacobians\": false"),
            std::string::npos);
  EXPECT_NE(metadata.find("\"manifold_applied_to_jacobians\": false"),
            std::string::npos);
  EXPECT_NE(metadata.find("\"constant_parameter_blocks_included\": true"),
            std::string::npos);
  EXPECT_NE(metadata.find("iter_000000_raw_jacobians_f64.bin"),
            std::string::npos);
  EXPECT_NE(metadata.find("\"role\":\"frame_center\""), std::string::npos);
  EXPECT_NE(metadata.find("\"role\":\"point3D\""), std::string::npos);
  EXPECT_NE(metadata.find("\"role\":\"bata_scale\""), std::string::npos);
  EXPECT_NE(metadata.find("\"role\":\"dmap_scale\""), std::string::npos);

  const std::optional<int64_t> num_residual_blocks =
      FindJsonIntForTest(metadata, "num_residual_blocks");
  const std::optional<int64_t> total_scalar_residuals =
      FindJsonIntForTest(metadata, "total_scalar_residuals");
  const std::optional<int64_t> total_jacobian_scalars =
      FindJsonIntForTest(metadata, "total_jacobian_scalars");
  ASSERT_TRUE(num_residual_blocks.has_value());
  ASSERT_TRUE(total_scalar_residuals.has_value());
  ASSERT_TRUE(total_jacobian_scalars.has_value());
  ASSERT_GT(*num_residual_blocks, 0);
  ASSERT_GT(*total_scalar_residuals, 0);
  ASSERT_GT(*total_jacobian_scalars, 0);

  std::vector<size_t> expected_parameter_block_sizes;
  std::vector<size_t> expected_raw_jacobian_offsets;
  size_t expected_jacobian_scalars = 0;
  for (const GlobalPositioningResidualReplayEntry& entry :
       positioner.ResidualReplayEntries()) {
    ASSERT_EQ(entry.parameter_blocks.size(), entry.parameter_block_sizes.size())
        << entry.residual_id;
    for (const int parameter_block_size : entry.parameter_block_sizes) {
      expected_parameter_block_sizes.push_back(
          static_cast<size_t>(parameter_block_size));
      expected_raw_jacobian_offsets.push_back(expected_jacobian_scalars);
      expected_jacobian_scalars +=
          entry.residual_dimension * static_cast<size_t>(parameter_block_size);
    }
  }

  EXPECT_EQ(static_cast<size_t>(*total_jacobian_scalars),
            expected_jacobian_scalars);
  EXPECT_EQ(FindJsonSizeArrayForTest(metadata, "parameter_block_sizes"),
            expected_parameter_block_sizes);
  EXPECT_EQ(FindJsonSizeArrayForTest(metadata, "raw_jacobian_offsets"),
            expected_raw_jacobian_offsets);
  EXPECT_EQ(std::filesystem::file_size(raw_jacobians_path),
            static_cast<uintmax_t>(*total_jacobian_scalars) * sizeof(double));

  const std::filesystem::path raw_binary_residual_values_path =
      trace_dir / "raw_binary" / "iterations" / "iter_000000" /
      "residual_values.bin";
  ASSERT_TRUE(ExistsFile(raw_binary_residual_values_path));
  EXPECT_EQ(ReadRawResidualValuesHeaderForTest(raw_binary_residual_values_path,
                                               /*expect_raw_jacobians=*/true),
            std::make_tuple(uint32_t{2},
                            int64_t{0},
                            static_cast<uint64_t>(*num_residual_blocks),
                            static_cast<uint64_t>(*total_scalar_residuals),
                            true));

  const std::vector<bool> evaluation_success =
      FindJsonBoolArrayForTest(metadata, "evaluation_success");
  ASSERT_EQ(evaluation_success.size(),
            static_cast<size_t>(*num_residual_blocks));
  EXPECT_TRUE(std::all_of(evaluation_success.begin(),
                          evaluation_success.end(),
                          [](const bool success) { return success; }));

  const std::vector<double> raw_jacobians = ReadDoubleSidecarForTest(
      raw_jacobians_path, static_cast<size_t>(*total_jacobian_scalars));
  EXPECT_TRUE(std::all_of(
      raw_jacobians.begin(), raw_jacobians.end(), [](const double value) {
        return std::isfinite(value);
      }));
}

TEST(GlobalPositioning, LowerTraceLevelsDoNotCreateResidualValues) {
  SetPRNGSeed(0);
  GpTestData data = BuildGpTestData();
  const std::vector<std::pair<GlobalPositioningTraceLevel, std::string>>
      lower_levels = {
          {GlobalPositioningTraceLevel::kSummary, "summary"},
          {GlobalPositioningTraceLevel::kResidualLedger, "residual_ledger"},
          {GlobalPositioningTraceLevel::kParameterSnapshots,
           "parameter_snapshots"},
      };

  for (const auto& [level, label] : lower_levels) {
    Reconstruction reconstruction = data.reconstruction;
    StampGtDepthPriors(reconstruction);
    const std::filesystem::path trace_dir =
        CreateTestDir() / ("gp_no_residual_values_" + label);

    GlobalPositionerOptions options = BaselineGpOptions();
    options.use_metric_depth_constraint = true;
    options.trace.level = level;
    options.trace.output_path = trace_dir;
    options.trace.snapshot_every_n_iterations = 1;
    options.trace.max_snapshotted_points = 2;

    TestableGlobalPositioner positioner(options);
    ASSERT_TRUE(positioner.Solve(data.pose_graph, reconstruction))
        << "trace level: " << label;
    EXPECT_EQ(positioner.NumReplayEntries(), 0u) << "trace level: " << label;
    EXPECT_FALSE(ExistsDir(trace_dir / "residual_values"))
        << "trace level: " << label;
  }
}

TEST(GlobalPositioning, MetricDepthConstraintConverges) {
  SetPRNGSeed(0);

  GpTestData data = BuildGpTestData();

  // Stamp ground-truth z-depths as depth priors on every image.
  for (const auto& [image_id, _] : data.gt_reconstruction.Images()) {
    Image& image = data.gt_reconstruction.Image(image_id);
    const size_t num_points2D = image.NumPoints2D();
    image.depth_priors.assign(num_points2D, 0.0);
    image.depth_prior_stddevs.assign(num_points2D, 0.0);
    image.depth_prior_validity.assign(num_points2D, false);

    for (point2D_t idx = 0; idx < num_points2D; ++idx) {
      if (!image.Point2D(idx).HasPoint3D()) continue;
      const auto& point3D =
          data.gt_reconstruction.Point3D(image.Point2D(idx).point3D_id);
      const Eigen::Vector3d point_cam = image.CamFromWorld() * point3D.xyz;
      const double z = point_cam[2];
      if (z > 0) {
        image.depth_priors[idx] = z;
        image.depth_prior_stddevs[idx] = 0.1 * z;
        image.depth_prior_validity[idx] = true;
      }
    }
  }

  // Copy depth priors into the working reconstruction (which has zero
  // translations but GT rotations -- mirrors BuildGpTestData).
  for (const auto& [image_id, _] : data.reconstruction.Images()) {
    const Image& gt_image = data.gt_reconstruction.Image(image_id);
    Image& image = data.reconstruction.Image(image_id);
    image.depth_priors = gt_image.depth_priors;
    image.depth_prior_stddevs = gt_image.depth_prior_stddevs;
    image.depth_prior_validity = gt_image.depth_prior_validity;
  }

  GlobalPositionerOptions options = BaselineGpOptions();
  options.use_metric_depth_constraint = true;
  options.use_init = false;
  options.generate_random_positions = true;
  options.generate_random_points = true;
  options.solver_options.max_num_iterations = 100;

  GlobalPositioner positioner(options);
  ASSERT_TRUE(positioner.Solve(data.pose_graph, data.reconstruction));

  const auto& dmap_scales = positioner.GetDmapScales();
  ASSERT_FALSE(dmap_scales.empty());

  for (const auto& [image_id, scale] : dmap_scales) {
    EXPECT_NEAR(scale, 1.0, 0.5) << "dmap_scale for image " << image_id << " = "
                                 << scale << ", expected ~1.0";
  }
}

TEST(GlobalPositioning, Gate_UseLcObservations_Off_IgnoresLcElements) {
  SetPRNGSeed(0);
  GpTestData data = BuildGpTestData();

  GlobalPositionerOptions options = BaselineGpOptions();
  ASSERT_FALSE(options.use_lc_observations);

  // Duplicate the regular elements as LC elements so every track has both
  // sets populated. Off-gate -> only regular elements added.
  const size_t expected_lc = DuplicateElementsAsLc(
      data.reconstruction, options.min_num_view_per_track);
  ASSERT_GT(expected_lc, 0u);

  const size_t expected_regular = CountValidObservations(
      data.reconstruction, options.min_num_view_per_track);

  TestableGlobalPositioner positioner(options);
  ASSERT_TRUE(positioner.Solve(data.pose_graph, data.reconstruction));

  EXPECT_EQ(positioner.NumScales(), expected_regular);
}

TEST(GlobalPositioning, Gate_UseLcObservations_On_IteratesLcElements) {
  SetPRNGSeed(0);
  GpTestData data = BuildGpTestData();

  GlobalPositionerOptions options = BaselineGpOptions();
  options.use_lc_observations = true;

  // Same setup as the OFF test; ON should now add both passes.
  DuplicateElementsAsLc(data.reconstruction, options.min_num_view_per_track);

  const size_t expected_regular = CountValidObservations(
      data.reconstruction, options.min_num_view_per_track);
  const size_t expected_lc =
      CountLcObservations(data.reconstruction, options.min_num_view_per_track);
  ASSERT_GT(expected_lc, 0u);

  TestableGlobalPositioner positioner(options);
  ASSERT_TRUE(positioner.Solve(data.pose_graph, data.reconstruction));

  EXPECT_EQ(positioner.NumScales(), expected_regular + expected_lc);
}

TEST(GlobalPositioning, MinViewGate_UseLcObservationsOff_IgnoresLcElements) {
  SetPRNGSeed(0);
  GpTestData data = BuildGpTestData();
  KeepSingleRegularPlusLcPoint(data.reconstruction);

  GlobalPositionerOptions options = BaselineGpOptions();
  options.min_num_view_per_track = 3;
  ASSERT_FALSE(options.use_lc_observations);

  PoseGraph empty_pose_graph;
  TestableGlobalPositioner positioner(options);
  positioner.SetupOnlyForTest(empty_pose_graph, data.reconstruction);

  EXPECT_EQ(positioner.NumFrameCenters(), 0u);
  EXPECT_EQ(positioner.NumScales(), 0u);
}

TEST(GlobalPositioning,
     MinViewGate_UseLcObservationsOn_RequiresRegularElements) {
  SetPRNGSeed(0);
  GpTestData data = BuildGpTestData();
  KeepSingleRegularPlusLcPoint(data.reconstruction);

  GlobalPositionerOptions options = BaselineGpOptions();
  options.min_num_view_per_track = 3;
  options.use_lc_observations = true;

  PoseGraph empty_pose_graph;
  TestableGlobalPositioner positioner(options);
  positioner.SetupOnlyForTest(empty_pose_graph, data.reconstruction);

  EXPECT_EQ(positioner.NumFrameCenters(), 0u);
  EXPECT_EQ(positioner.NumScales(), 0u);
}

// ---- Depth-prior integration tests ----

// Stamp GT z-depth priors on every image in the reconstruction.
void StampGtDepthPriors(Reconstruction& reconstruction) {
  for (const auto& [image_id, _] : reconstruction.Images()) {
    Image& image = reconstruction.Image(image_id);
    const size_t n = image.NumPoints2D();
    image.depth_priors.assign(n, 0.0);
    image.depth_prior_stddevs.assign(n, 0.0);
    image.depth_prior_validity.assign(n, false);
    for (point2D_t idx = 0; idx < static_cast<point2D_t>(n); ++idx) {
      if (!image.Point2D(idx).HasPoint3D()) continue;
      const auto& p3d = reconstruction.Point3D(image.Point2D(idx).point3D_id);
      const Eigen::Vector3d pc = image.CamFromWorld() * p3d.xyz;
      if (pc[2] > 0) {
        image.depth_priors[idx] = pc[2];
        image.depth_prior_stddevs[idx] = 0.1 * pc[2];
        image.depth_prior_validity[idx] = true;
      }
    }
  }
}

TEST(GlobalPositioning, ResidualLedgerTraceWritesBlocksAndMatchesProblemBuilt) {
  SetPRNGSeed(0);
  GpTestData data = BuildGpTestData();
  StampGtDepthPriors(data.reconstruction);

  const std::filesystem::path trace_dir =
      CreateTestDir() / "gp_residual_ledger";

  GlobalPositionerOptions options = BaselineGpOptions();
  options.use_metric_depth_constraint = true;
  options.trace.level = GlobalPositioningTraceLevel::kResidualLedger;
  options.trace.output_path = trace_dir;

  TestableGlobalPositioner positioner(options);
  ASSERT_TRUE(positioner.Solve(data.pose_graph, data.reconstruction));

  const std::filesystem::path residual_blocks_path =
      trace_dir / "residual_blocks.jsonl";
  ASSERT_TRUE(ExistsFile(residual_blocks_path));

  const std::string residual_blocks = ReadFileForTest(residual_blocks_path);
  ExpectEveryResidualBlockHasReplayLedgerFieldsForTest(residual_blocks);
  EXPECT_TRUE(ContainsJsonStringValueForTest(
      residual_blocks, "residual_type", "bata_ref_frame"));
  EXPECT_TRUE(ContainsJsonStringValueForTest(
      residual_blocks, "residual_type", "metric_depth"));
  EXPECT_TRUE(ContainsJsonStringValueForTest(
      residual_blocks, "residual_type", "scale_prior"));

  const std::string events = ReadFileForTest(trace_dir / "events.jsonl");
  const std::optional<int64_t> expected_num_residual_blocks =
      FindEventAttrIntForTest(events, "problem_built", "num_residual_blocks");
  ASSERT_TRUE(expected_num_residual_blocks.has_value());
  EXPECT_EQ(CountJsonlRecordsForTest(residual_blocks),
            static_cast<size_t>(*expected_num_residual_blocks));
}

TEST(GlobalPositioning, ResidualLedgerTraceWritesReplayDescriptorsAndLoss) {
  SetPRNGSeed(0);
  GpTestData data = BuildGpTestData();
  StampGtDepthPriors(data.reconstruction);

  const std::filesystem::path trace_dir =
      CreateTestDir() / "gp_residual_ledger_replay_fields";

  GlobalPositionerOptions options = BaselineGpOptions();
  options.use_metric_depth_constraint = true;
  options.trace.level = GlobalPositioningTraceLevel::kResidualLedger;
  options.trace.output_path = trace_dir;

  TestableGlobalPositioner positioner(options);
  ASSERT_TRUE(positioner.Solve(data.pose_graph, data.reconstruction));

  const std::string residual_blocks =
      ReadFileForTest(trace_dir / "residual_blocks.jsonl");
  ExpectEveryResidualBlockHasReplayLedgerFieldsForTest(residual_blocks);

  const std::optional<std::string> bata_ref_frame =
      FindJsonlRecordWithStringValueForTest(
          residual_blocks, "residual_type", "bata_ref_frame");
  ASSERT_TRUE(bata_ref_frame.has_value());
  EXPECT_NE(bata_ref_frame->find("\"role\":\"frame_center\""),
            std::string::npos);
  EXPECT_NE(bata_ref_frame->find("\"role\":\"point3D\""), std::string::npos);
  EXPECT_NE(bata_ref_frame->find("\"role\":\"bata_scale\""), std::string::npos);
  EXPECT_NE(bata_ref_frame->find("\"loss\":{\"bucket\":\"geometry_"),
            std::string::npos);
  EXPECT_NE(bata_ref_frame->find("\"cam_from_point3D_dir\""),
            std::string::npos);

  const std::optional<std::string> metric_depth =
      FindJsonlRecordWithStringValueForTest(
          residual_blocks, "residual_type", "metric_depth");
  ASSERT_TRUE(metric_depth.has_value());
  EXPECT_NE(metric_depth->find("\"role\":\"dmap_scale\""), std::string::npos);
  EXPECT_NE(metric_depth->find("\"loss\":{\"bucket\":\"depth_"),
            std::string::npos);
  EXPECT_NE(metric_depth->find("\"camera_rotation_wxyz\""), std::string::npos);
  EXPECT_NE(metric_depth->find("\"metric_depth_use_log_scale\""),
            std::string::npos);
  EXPECT_NE(metric_depth->find("\"metric_depth_residual_type\""),
            std::string::npos);
  EXPECT_NE(metric_depth->find("\"metric_depth_zero_residual_behind\""),
            std::string::npos);
  EXPECT_NE(metric_depth->find("\"metric_depth_log_linear_threshold\""),
            std::string::npos);

  const std::optional<std::string> scale_prior =
      FindJsonlRecordWithStringValueForTest(
          residual_blocks, "residual_type", "scale_prior");
  ASSERT_TRUE(scale_prior.has_value());
  EXPECT_NE(scale_prior->find("\"role\":\"dmap_scale\""), std::string::npos);
  EXPECT_NE(scale_prior->find("\"loss\":{\"bucket\":\"scale_prior\""),
            std::string::npos);
  EXPECT_NE(scale_prior->find("\"observation_count_weight\":"),
            std::string::npos);
  EXPECT_NE(scale_prior->find("\"scale_prior_target\""), std::string::npos);
  EXPECT_NE(scale_prior->find("\"scale_prior_stddev\""), std::string::npos);
}

TEST(GlobalPositioning, ResidualLedgerTraceCanForceBataConstantRigFamily) {
  SetPRNGSeed(0);
  GpTestData data = BuildGpTestData(/*num_rigs=*/1,
                                    /*num_cameras_per_rig=*/2,
                                    /*num_frames_per_rig=*/4,
                                    /*num_points3D=*/40);

  const std::filesystem::path trace_dir =
      CreateTestDir() / "gp_constant_rig_family";

  GlobalPositionerOptions options = BaselineGpOptions();
  options.trace.level = GlobalPositioningTraceLevel::kResidualLedger;
  options.trace.output_path = trace_dir;

  TestableGlobalPositioner positioner(options);
  ASSERT_TRUE(positioner.Solve(data.pose_graph, data.reconstruction));

  const std::string residual_blocks =
      ReadFileForTest(trace_dir / "residual_blocks.jsonl");
  ExpectEveryResidualBlockHasReplayLedgerFieldsForTest(residual_blocks);
  const std::optional<std::string> constant_rig =
      FindJsonlRecordWithStringValueForTest(
          residual_blocks, "residual_type", "bata_constant_rig");
  ASSERT_TRUE(constant_rig.has_value());
  EXPECT_NE(constant_rig->find("\"cam_from_point3D_dir\""), std::string::npos);
  EXPECT_NE(constant_rig->find("\"cam_from_rig_dir\""), std::string::npos);
}

TEST(GlobalPositioning, ResidualLedgerTraceCanForceBataVariableRigFamily) {
  SetPRNGSeed(0);
  GpTestData data = BuildGpTestData(/*num_rigs=*/1,
                                    /*num_cameras_per_rig=*/2,
                                    /*num_frames_per_rig=*/4,
                                    /*num_points3D=*/40);
  ForceNonRefRigTranslationsUnknownForTest(data.reconstruction);

  const std::filesystem::path trace_dir =
      CreateTestDir() / "gp_variable_rig_family";

  GlobalPositionerOptions options = BaselineGpOptions();
  options.trace.level = GlobalPositioningTraceLevel::kResidualLedger;
  options.trace.output_path = trace_dir;

  TestableGlobalPositioner positioner(options);
  ASSERT_TRUE(positioner.Solve(data.pose_graph, data.reconstruction));

  const std::string residual_blocks =
      ReadFileForTest(trace_dir / "residual_blocks.jsonl");
  ExpectEveryResidualBlockHasReplayLedgerFieldsForTest(residual_blocks);
  const std::optional<std::string> variable_rig =
      FindJsonlRecordWithStringValueForTest(
          residual_blocks, "residual_type", "bata_variable_rig");
  ASSERT_TRUE(variable_rig.has_value());
  EXPECT_NE(variable_rig->find("\"cam_from_point3D_dir\""), std::string::npos);
  EXPECT_NE(variable_rig->find("\"rig_from_world_rotation_wxyz\""),
            std::string::npos);
  EXPECT_NE(variable_rig->find("\"world_from_rig_rotation_wxyz\""),
            std::string::npos);
}

TEST(GlobalPositioning, ResidualLedgerTraceWritesMissingDepthPriorSkip) {
  SetPRNGSeed(0);
  GpTestData data = BuildGpTestData();
  StampGtDepthPriors(data.reconstruction);

  const std::optional<TrackElement> observation =
      FindFirstValidDepthObservationForTest(data.reconstruction);
  ASSERT_TRUE(observation.has_value());

  Image& image = data.reconstruction.Image(observation->image_id);
  ASSERT_LT(observation->point2D_idx, image.depth_prior_validity.size());
  image.depth_prior_validity[observation->point2D_idx] = false;

  const std::filesystem::path trace_dir = CreateTestDir() / "gp_residual_skip";

  GlobalPositionerOptions options = BaselineGpOptions();
  options.use_metric_depth_constraint = true;
  options.trace.level = GlobalPositioningTraceLevel::kResidualLedger;
  options.trace.output_path = trace_dir;

  TestableGlobalPositioner positioner(options);
  ASSERT_TRUE(positioner.Solve(data.pose_graph, data.reconstruction));

  const std::filesystem::path residual_skips_path =
      trace_dir / "residual_skips.jsonl";
  ASSERT_TRUE(ExistsFile(residual_skips_path));

  const std::string residual_skips = ReadFileForTest(residual_skips_path);
  EXPECT_TRUE(ContainsJsonStringValueForTest(
      residual_skips, "event_type", "residual_skipped"));
  EXPECT_TRUE(ContainsJsonStringValueForTest(
      residual_skips, "skip_reason", "missing_depth_validity"));
}

TEST(GlobalPositioning, FilterDepthOutliersRoutesSoftFallback) {
  SetPRNGSeed(0);
  GpTestData data = BuildGpTestData();

  // Stamp GT depth priors on every image.
  StampGtDepthPriors(data.reconstruction);

  // Corrupt one valid observation to create a depth outlier.
  {
    bool corrupted = false;
    for (const auto& [image_id, _] : data.reconstruction.Images()) {
      Image& image = data.reconstruction.Image(image_id);
      for (point2D_t idx = 0; idx < static_cast<point2D_t>(image.NumPoints2D());
           ++idx) {
        if (image.depth_prior_validity[idx]) {
          image.depth_priors[idx] *= 100.0;  // wildly wrong
          corrupted = true;
          break;
        }
      }
      if (corrupted) break;
    }
    ASSERT_TRUE(corrupted) << "No valid depth prior found to corrupt";
  }

  GlobalPositionerOptions options = BaselineGpOptions();
  options.use_metric_depth_constraint = true;
  options.filter_depth_outliers = true;
  options.filter_depth_outlier_sigma = 3.0;
  options.use_init = false;

  TestableGlobalPositioner positioner(options);
  ASSERT_TRUE(positioner.Solve(data.pose_graph, data.reconstruction));

  // BATA residual blocks should have been added.
  EXPECT_GT(positioner.NumScales(), 0u);

  // dmap_scales must be populated (one entry per image with valid depth obs).
  EXPECT_FALSE(positioner.GetDmapScales().empty());
}

TEST(GlobalPositioning, CallerSuppliedInitialDmapScales) {
  SetPRNGSeed(0);
  GpTestData data = BuildGpTestData();

  // Stamp GT depth priors.
  StampGtDepthPriors(data.reconstruction);

  // Build initial dmap_scales with a non-default value.
  std::unordered_map<image_t, double> init_scales;
  for (const auto& [image_id, _] : data.reconstruction.Images()) {
    init_scales[image_id] = 2.5;
  }

  GlobalPositionerOptions options = BaselineGpOptions();
  options.use_metric_depth_constraint = true;
  options.initial_dmap_scales = init_scales;
  options.optimize_scales = false;                // freeze BATA scales
  options.solver_options.max_num_iterations = 0;  // no optimization
  options.use_init = false;

  TestableGlobalPositioner positioner(options);
  ASSERT_TRUE(positioner.Solve(data.pose_graph, data.reconstruction));

  // With 0 iterations and frozen scales, dmap_scales should preserve the
  // initial 2.5 value exactly.
  const auto& dmap_scales = positioner.GetDmapScales();
  EXPECT_FALSE(dmap_scales.empty());
  for (const auto& [image_id, scale] : dmap_scales) {
    EXPECT_NEAR(scale, 2.5, 0.01) << "dmap_scale for image " << image_id
                                  << " deviated from initial value";
  }
}

}  // namespace
}  // namespace colmap
