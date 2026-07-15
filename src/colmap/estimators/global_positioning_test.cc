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
#include "colmap/util/testing.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
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

  MetricDepthError functor(identity,
                           depth_prior,
                           sigma_depth,
                           /*use_log_scale=*/false,
                           MetricDepthResidualType::kLogLinear,
                           /*zero_residual_behind=*/false,
                           threshold);

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

namespace {

// Test-only subclass exposing ``scales_`` (one entry per BATA residual
// block added by ``AddObservationToProblem``) so we can count residuals
// without instrumenting the production class.
class TestableGlobalPositioner : public GlobalPositioner {
 public:
  using GlobalPositioner::GlobalPositioner;
  size_t NumScales() const { return scales_.size(); }
  size_t NumFrameCenters() const { return frame_centers_.size(); }
  size_t NumNonUnitScales(double eps = 1e-9) const {
    return std::count_if(scales_.begin(), scales_.end(), [eps](double scale) {
      return std::abs(scale - 1.0) > eps;
    });
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

GpTestData BuildGpTestData() {
  GpTestData data;
  const auto database_path = CreateTestDir() / "database.db";
  data.database = Database::Open(database_path);

  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 1;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 5;
  synthetic_dataset_options.num_points3D = 30;
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
    const auto& regular = point3D.track.Elements();
    for (size_t index = 0; index < point3D.track.lc_elements.size(); ++index) {
      TrackElement& lc = point3D.track.lc_elements[index];
      const TrackElement& anchor = regular[(index + 1) % regular.size()];
      lc.lc_anchor_image_id = anchor.image_id;
      lc.lc_anchor_point2D_idx = anchor.point2D_idx;
    }
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
  point3D.track.lc_elements.back().lc_anchor_image_id = image_ids[0];
  point3D.track.lc_elements.back().lc_anchor_point2D_idx = 0;
  point3D.track.lc_elements.emplace_back(image_ids[2], 0);
  point3D.track.lc_elements.back().lc_anchor_image_id = image_ids[0];
  point3D.track.lc_elements.back().lc_anchor_point2D_idx = 0;
  reconstruction.AddPoint3D(0, std::move(point3D));
}

TEST(GlobalPositioning, PlaybackCapturesCompactTimeline) {
  GpTestData data = BuildGpTestData();
  GlobalPositionerOptions options = BaselineGpOptions();
  options.use_lc_observations = true;
  options.solver_options.max_num_iterations = 2;
  DuplicateElementsAsLc(data.reconstruction, options.min_num_view_per_track);
  using Observation = std::pair<image_t, point2D_t>;
  using Match = std::pair<Observation, Observation>;
  std::map<std::pair<image_t, image_t>, std::set<Match>> expected_edges;
  for (const auto& [_, point3D] : data.reconstruction.Points3D()) {
    for (const TrackElement& observation : point3D.track.lc_elements) {
      const auto image_pair =
          std::minmax(observation.image_id, observation.lc_anchor_image_id);
      const Observation endpoint(observation.image_id, observation.point2D_idx);
      const Observation anchor(observation.lc_anchor_image_id,
                               observation.lc_anchor_point2D_idx);
      expected_edges[image_pair].insert(std::minmax(endpoint, anchor));
    }
  }

  std::vector<GlobalPositioningPlaybackCapture> captures;
  options.playback.callback =
      [&captures](const GlobalPositioningPlaybackCapture& capture) {
        captures.push_back(capture);
      };

  GlobalPositioner positioner(options);
  ASSERT_TRUE(positioner.Solve(data.pose_graph, data.reconstruction));
  ASSERT_GE(captures.size(), 2);
  EXPECT_EQ(captures.front().phase, "initial");
  EXPECT_EQ(captures.back().phase, "final");
  ASSERT_GE(captures.size(), 3);
  EXPECT_EQ(captures[1].phase, "iteration");
  EXPECT_EQ(captures[1].iteration, 0);
  EXPECT_EQ(captures.back().iteration, captures[captures.size() - 2].iteration);
  for (const GlobalPositioningPlaybackCapture& capture : captures) {
    EXPECT_EQ(capture.image_centers.size(), 3 * capture.image_ids.size());
    EXPECT_EQ(capture.points3D.size(), 3 * capture.point3D_ids.size());
    EXPECT_EQ(capture.lc_pairs.size(), 2 * capture.lc_support_count.size());
    EXPECT_EQ(capture.lc_raw_score.size(), capture.lc_support_count.size());
    EXPECT_LE(capture.point3D_ids.size(), 200000);
    EXPECT_FALSE(capture.image_ids.empty());
    EXPECT_FALSE(capture.point3D_ids.empty());
    EXPECT_FALSE(capture.lc_pairs.empty());
    ASSERT_EQ(capture.lc_support_count.size(), expected_edges.size());
    for (size_t index = 0; index < capture.lc_support_count.size(); ++index) {
      const image_t image_id1 =
          static_cast<image_t>(capture.lc_pairs[2 * index]);
      const image_t image_id2 =
          static_cast<image_t>(capture.lc_pairs[2 * index + 1]);
      const auto pair = std::minmax(image_id1, image_id2);
      const auto expected = expected_edges.find(pair);
      ASSERT_NE(expected, expected_edges.end());
      EXPECT_EQ(capture.lc_support_count[index], expected->second.size());
    }
    EXPECT_TRUE(std::all_of(capture.lc_raw_score.begin(),
                            capture.lc_raw_score.end(),
                            [](const double score) {
                              return std::isfinite(score) && score >= 0.0;
                            }));
  }
}

TEST(GlobalPositioning, PlaybackRetainsEquivalentSolverResult) {
  GpTestData data = BuildGpTestData();
  Reconstruction disabled = data.reconstruction;
  Reconstruction enabled = data.reconstruction;
  GlobalPositionerOptions disabled_options = BaselineGpOptions();
  disabled_options.solver_options.max_num_iterations = 5;
  GlobalPositionerOptions enabled_options = disabled_options;
  int capture_count = 0;
  enabled_options.playback.callback =
      [&capture_count](const GlobalPositioningPlaybackCapture&) {
        ++capture_count;
      };

  GlobalPositioner disabled_positioner(disabled_options);
  ASSERT_TRUE(disabled_positioner.Solve(data.pose_graph, disabled));
  GlobalPositioner enabled_positioner(enabled_options);
  ASSERT_TRUE(enabled_positioner.Solve(data.pose_graph, enabled));
  EXPECT_GT(capture_count, 0);
  EXPECT_THAT(disabled,
              ReconstructionNear(enabled,
                                 /*max_rotation_error_deg=*/1e-9,
                                 /*max_proj_center_error=*/1e-8,
                                 /*max_scale_error=*/1e-8,
                                 /*num_obs_tolerance=*/0.0));
}

}  // namespace

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

TEST(GlobalPositioning, WarmStartGenerateScalesFalseInitializesBataScales) {
  SetPRNGSeed(0);
  GpTestData data = BuildGpTestData();

  GlobalPositionerOptions options = BaselineGpOptions();
  options.use_init = true;
  options.generate_scales = false;
  options.initialize_warm_start_scales = true;

  TestableGlobalPositioner positioner(options);
  positioner.SetupOnlyForTest(data.pose_graph, data.gt_reconstruction);

  EXPECT_GT(positioner.NumScales(), 0u);
  EXPECT_GT(positioner.NumNonUnitScales(), 0u)
      << "Warm-started GP with generate_scales=false must initialize BATA "
         "scales from current camera/point geometry instead of leaving every "
         "scale at the constructor default 1.0.";
}

TEST(GlobalPositioning, LegacyWarmStartScaleInitializationCanBeReproduced) {
  SetPRNGSeed(0);
  GpTestData data = BuildGpTestData();

  GlobalPositionerOptions options = BaselineGpOptions();
  options.use_init = true;
  options.generate_scales = false;
  options.initialize_warm_start_scales = false;

  TestableGlobalPositioner positioner(options);
  positioner.SetupOnlyForTest(data.pose_graph, data.gt_reconstruction);

  EXPECT_GT(positioner.NumScales(), 0u);
  EXPECT_EQ(positioner.NumNonUnitScales(), 0u)
      << "Legacy warm-started GP behavior should leave BATA scales at 1.0 "
         "when generate_scales=false.";
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
