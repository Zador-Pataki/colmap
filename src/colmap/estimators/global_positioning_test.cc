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

#include "colmap/math/random.h"
#include "colmap/scene/database_cache.h"
#include "colmap/scene/pose_graph.h"
#include "colmap/scene/reconstruction_matchers.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/testing.h"

#include <cmath>

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

}  // namespace
}  // namespace colmap
