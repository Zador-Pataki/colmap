// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
// (BSD-3-Clause license, see LICENSE)

#include "colmap/estimators/bundle_adjustment.h"
#include "colmap/estimators/global_positioning.h"
#include "colmap/estimators/loss_function_config.h"
#include "colmap/estimators/rotation_averaging.h"

#include <ceres/loss_function.h>
#include <gtest/gtest.h>

#include <memory>

namespace colmap {
namespace {

// Suite 1: LossFunctionConfig

TEST(LossFunctionConfigTest, DefaultsAreTrivialOneOne) {
  LossFunctionConfig cfg;
  EXPECT_EQ(cfg.name, "trivial");
  EXPECT_EQ(cfg.scale, 1.0);
  EXPECT_EQ(cfg.weight, 1.0);
}

TEST(LossFunctionConfigTest, CreateHuber) {
  auto loss = LossFunctionConfig{"huber", 0.5, 1.0}.Create();
  ASSERT_NE(loss, nullptr);
  EXPECT_NE(dynamic_cast<ceres::HuberLoss*>(loss.get()), nullptr);
}

TEST(LossFunctionConfigTest, CreateCauchy) {
  auto loss = LossFunctionConfig{"cauchy", 0.5, 1.0}.Create();
  ASSERT_NE(loss, nullptr);
  EXPECT_NE(dynamic_cast<ceres::CauchyLoss*>(loss.get()), nullptr);
}

TEST(LossFunctionConfigTest, CreateArctan) {
  auto loss = LossFunctionConfig{"arctan", 0.5, 1.0}.Create();
  ASSERT_NE(loss, nullptr);
  EXPECT_NE(dynamic_cast<ceres::ArctanLoss*>(loss.get()), nullptr);
}

TEST(LossFunctionConfigTest, CreateTrivial) {
  auto loss = LossFunctionConfig{"trivial", 1.0, 1.0}.Create();
  ASSERT_NE(loss, nullptr);
  EXPECT_NE(dynamic_cast<ceres::TrivialLoss*>(loss.get()), nullptr);
}

TEST(LossFunctionConfigTest, CreateScaledWhenWeightNotOne) {
  auto loss = LossFunctionConfig{"huber", 0.1, 2.0}.Create();
  ASSERT_NE(loss, nullptr);
  EXPECT_NE(dynamic_cast<ceres::ScaledLoss*>(loss.get()), nullptr);
}

TEST(LossFunctionConfigTest, UnknownNameFatals) {
  EXPECT_DEATH(LossFunctionConfig{"xyz", 1.0, 1.0}.Create(), "");
}

// Suite 2: GlobalPositionerOptions

TEST(GlobalPositionerOptionsTest, ForkDefaults) {
  GlobalPositionerOptions opt;
  EXPECT_EQ(opt.random_init_scale, 100.0);
  EXPECT_FALSE(opt.use_init);
  EXPECT_TRUE(opt.optimize_depth_map_scales);
  EXPECT_FALSE(opt.optimize_rotations);
  EXPECT_TRUE(opt.regularize_rotations);
  EXPECT_EQ(opt.rotation_prior_sigma, 0.01);
  EXPECT_EQ(opt.scale_regularization_sigma, 1.0);
  EXPECT_EQ(opt.scale_prior_stddev, 0.1);
  EXPECT_TRUE(opt.regularize_depth_map_scales);
  EXPECT_TRUE(opt.use_log_scale_for_depth_map_scales);
  EXPECT_FALSE(opt.use_log_residual_for_depth);
  EXPECT_FALSE(opt.zero_residual_behind_camera);
  EXPECT_FALSE(opt.smooth_log_linear_transition);
  EXPECT_EQ(opt.log_linear_threshold, 0.1);
  EXPECT_FALSE(opt.filter_depth_outliers);
  EXPECT_TRUE(opt.consecutive_trivial);
  EXPECT_FALSE(opt.use_relative_pose_constraints);
  EXPECT_EQ(opt.relative_pose_default_stddev, 0.1);
  EXPECT_FALSE(opt.debug_only_relative_pose);
  EXPECT_EQ(opt.seed, 1u);
  EXPECT_EQ(opt.constraint_type, ConstraintType::ONLY_POINTS);
  EXPECT_EQ(opt.point_constraint_type, PointConstraintType::GEOMETRY_ONLY);
}

TEST(GlobalPositionerOptionsTest, LossChannelDefaults) {
  GlobalPositionerOptions opt;
  EXPECT_EQ(opt.loss_normal_geometry.name, "huber");
  EXPECT_EQ(opt.loss_normal_geometry.scale, 0.1);
  EXPECT_EQ(opt.loss_normal_geometry.weight, 1.0);

  EXPECT_EQ(opt.loss_normal_depth.name, "huber");
  EXPECT_EQ(opt.loss_normal_depth.scale, 0.1);

  EXPECT_EQ(opt.loss_lc_geometry.name, "huber");
  EXPECT_EQ(opt.loss_lc_depth.name, "huber");
  EXPECT_EQ(opt.loss_scale_prior.name, "huber");

  EXPECT_EQ(opt.loss_normal_geometry_inlier.name, "trivial");
  EXPECT_EQ(opt.loss_normal_geometry_inlier.scale, 1.0);
  EXPECT_EQ(opt.loss_normal_geometry_inlier.weight, 1.0);

  EXPECT_EQ(opt.loss_normal_depth_inlier.name, "trivial");
  EXPECT_EQ(opt.loss_normal_depth_outlier.name, "huber");
  EXPECT_EQ(opt.loss_normal_depth_outlier.scale, 1.0);

  EXPECT_EQ(opt.loss_normal_geometry_trackstart.name, "trivial");
  EXPECT_EQ(opt.loss_normal_depth_trackstart.name, "huber");
  EXPECT_EQ(opt.loss_normal_depth_trackstart.scale, 1.0);

  EXPECT_EQ(opt.loss_relative_pose.name, "huber");
  EXPECT_EQ(opt.loss_relative_pose.scale, 0.1);

  EXPECT_EQ(opt.loss_rotation_prior.name, "trivial");
  EXPECT_EQ(opt.loss_rotation_prior.scale, 1.0);
  EXPECT_EQ(opt.loss_rotation_prior.weight, 1.0);
}

TEST(GlobalPositionerOptionsTest, EnumsAreDistinct) {
  EXPECT_NE(static_cast<int>(ConstraintType::ONLY_POINTS),
            static_cast<int>(ConstraintType::ALL));
  EXPECT_NE(static_cast<int>(PointConstraintType::GEOMETRY_ONLY),
            static_cast<int>(PointConstraintType::SPLIT_METRIC_DEPTH));
}

// Suite 3: RotationEstimatorOptions

TEST(RotationEstimatorOptionsTest, ForkDefaults) {
  RotationEstimatorOptions opt;
  EXPECT_NEAR(opt.axis.x(), 0.0, 1e-12);
  EXPECT_NEAR(opt.axis.y(), 1.0, 1e-12);
  EXPECT_NEAR(opt.axis.z(), 0.0, 1e-12);
  EXPECT_FALSE(opt.use_precomputed_weights);
  EXPECT_FALSE(opt.fix_non_lc_weights);
  EXPECT_EQ(opt.fixed_non_lc_weight, 138.0);
  EXPECT_FALSE(opt.skip_risky_LC_pairs);
  EXPECT_FALSE(opt.use_video_constraints);
  EXPECT_NEAR(opt.video_tracking_huber_scale, 0.1, 1e-9);
  EXPECT_NEAR(opt.video_lc_cauchy_scale, 0.05, 1e-9);
  EXPECT_FALSE(opt.use_gravity_prior);
  EXPECT_NEAR(opt.gravity_world.x(), 0.0, 1e-12);
  EXPECT_NEAR(opt.gravity_world.y(), -1.0, 1e-12);
  EXPECT_NEAR(opt.gravity_world.z(), 0.0, 1e-12);
  EXPECT_EQ(opt.default_gravity_sigma, 0.035);
  EXPECT_EQ(opt.gravity_loss_scale, 0.05);
  EXPECT_EQ(opt.gravity_loss_type, "cauchy");
}

// Suite 4: BundleAdjustmentOptions

TEST(BundleAdjustmentOptionsTest, ForkDefaults) {
  BundleAdjustmentOptions opt;
  EXPECT_TRUE(opt.optimize_rotations);
  EXPECT_TRUE(opt.optimize_translation);
  EXPECT_TRUE(opt.optimize_intrinsics);
  EXPECT_TRUE(opt.optimize_points);
  EXPECT_EQ(opt.min_num_view_per_track, 3);
  EXPECT_EQ(opt.loss_reprojection.name, "huber");
  EXPECT_EQ(opt.loss_reprojection.scale, 1.0);
  EXPECT_EQ(opt.loss_reprojection.weight, 1.0);
  EXPECT_EQ(opt.loss_lc_reprojection.name, "huber");
  EXPECT_EQ(opt.loss_lc_reprojection.scale, 1.0);
  // Regression: upstream field must not be accidentally deleted.
  EXPECT_TRUE(opt.refine_focal_length);
}

// Suite 5: Virtual destructor safety (delete via base-class pointer)

TEST(GlobalPositionerVirtualDtorTest, DeleteViaBasePointer) {
  // If GlobalPositioner lacks a virtual dtor, deleting via base ptr is UB.
  std::unique_ptr<GlobalPositioner> p =
      std::make_unique<GlobalPositioner>(GlobalPositionerOptions{});
  EXPECT_NE(p, nullptr);
}

TEST(RotationEstimatorVirtualDtorTest, DeleteViaBasePointer) {
  std::unique_ptr<RotationEstimator> p =
      std::make_unique<RotationEstimator>(RotationEstimatorOptions{});
  EXPECT_NE(p, nullptr);
}

}  // namespace
}  // namespace colmap
