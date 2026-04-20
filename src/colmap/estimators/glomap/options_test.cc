#include "colmap/estimators/glomap/bundle_adjustment_options.h"
#include "colmap/estimators/glomap/global_mapper_options.h"
#include "colmap/estimators/glomap/global_positioner_options.h"
#include "colmap/estimators/glomap/inlier_threshold_options.h"
#include "colmap/estimators/glomap/optimization_base_options.h"
#include "colmap/estimators/glomap/rotation_estimator_options.h"
#include "colmap/estimators/glomap/track_establishment_options.h"
#include "colmap/estimators/glomap/triangulator_options.h"
#include "colmap/estimators/glomap/view_graph_calibrator_options.h"

#include <gtest/gtest.h>

namespace colmap::glomap {
namespace {

TEST(GlobalPositionerOptions, DefaultConstruction) {
  GlobalPositionerOptions opts;
  EXPECT_TRUE(opts.generate_random_positions);
  EXPECT_DOUBLE_EQ(opts.random_init_scale, 100.0);
  EXPECT_FALSE(opts.optimize_rotations);
  EXPECT_DOUBLE_EQ(opts.rotation_prior_sigma, 0.01);
  EXPECT_EQ(opts.min_num_view_per_track, 3);
  EXPECT_EQ(opts.seed, 1u);
  EXPECT_EQ(opts.constraint_type, GlobalPositionerOptions::ConstraintType::ONLY_POINTS);
  EXPECT_EQ(opts.point_constraint_type,
            GlobalPositionerOptions::PointConstraintType::GEOMETRY_ONLY);
  EXPECT_TRUE(opts.use_log_scale_for_depth_map_scales);
}

TEST(GlobalPositionerOptions, LossFunctionConfigNested) {
  GlobalPositionerOptions opts;
  EXPECT_EQ(opts.loss_normal_geometry.name, "huber");
  EXPECT_DOUBLE_EQ(opts.loss_normal_geometry.scale, 0.1);
  EXPECT_DOUBLE_EQ(opts.loss_normal_geometry.weight, 1.0);
  EXPECT_EQ(opts.loss_normal_geometry_inlier.name, "trivial");
  EXPECT_EQ(opts.loss_rotation_prior.name, "trivial");
}

TEST(GlobalPositionerOptions, ConstraintTypeEnumValues) {
  EXPECT_EQ(static_cast<int>(GlobalPositionerOptions::ConstraintType::ONLY_POINTS), 0);
}

TEST(GlobalPositionerOptions, PointConstraintTypeEnumValues) {
  EXPECT_EQ(static_cast<int>(GlobalPositionerOptions::PointConstraintType::GEOMETRY_ONLY), 0);
  EXPECT_EQ(static_cast<int>(GlobalPositionerOptions::PointConstraintType::SPLIT_METRIC_DEPTH), 1);
}

TEST(RotationEstimatorOptions, DefaultConstruction) {
  RotationEstimatorOptions opts;
  EXPECT_EQ(opts.max_num_l1_iterations, 5);
  EXPECT_EQ(opts.max_num_irls_iterations, 100);
  EXPECT_EQ(opts.solver_type, RotationEstimatorOptions::SolverType::L1_IRLS);
  EXPECT_EQ(opts.weight_type, RotationEstimatorOptions::WeightType::GEMAN_MCCLURE);
  EXPECT_EQ(opts.max_num_ceres_iterations, 100);
}

TEST(RotationEstimatorOptions, SolverTypeEnum) {
  EXPECT_EQ(static_cast<int>(RotationEstimatorOptions::SolverType::L1_IRLS), 0);
  EXPECT_EQ(static_cast<int>(RotationEstimatorOptions::SolverType::CERES_GRAVITY), 1);
}

TEST(GlobalMapperOptions, DefaultConstruction) {
  GlobalMapperOptions opts;
  EXPECT_EQ(opts.num_iteration_bundle_adjustment, 3);
  EXPECT_FALSE(opts.skip_view_graph_calibration);
  EXPECT_TRUE(opts.skip_pruning);
}

TEST(ViewGraphCalibratorOptions, DefaultConstruction) {
  ViewGraphCalibratorOptions opts;
  EXPECT_DOUBLE_EQ(opts.thres_lower_ratio, 0.1);
  EXPECT_DOUBLE_EQ(opts.thres_higher_ratio, 10.0);
  EXPECT_DOUBLE_EQ(opts.solver_base.thres_loss_function, 1e-2);
}

TEST(TriangulatorOptions, DefaultConstruction) {
  TriangulatorOptions opts;
  EXPECT_DOUBLE_EQ(opts.tri_complete_max_reproj_error, 15.0);
  EXPECT_DOUBLE_EQ(opts.tri_min_angle, 1.0);
  EXPECT_EQ(opts.min_num_matches, 15);
}

TEST(TrackEstablishmentOptions, DefaultConstruction) {
  TrackEstablishmentOptions opts;
  EXPECT_DOUBLE_EQ(opts.thres_inconsistency, 10.0);
  EXPECT_EQ(opts.min_num_view_per_track, 3);
  EXPECT_EQ(opts.max_num_view_per_track, 100);
}

TEST(BundleAdjustmentOptions, DefaultConstruction) {
  BundleAdjustmentOptions opts;
  EXPECT_TRUE(opts.optimize_rotations);
  EXPECT_TRUE(opts.optimize_translation);
  EXPECT_EQ(opts.min_num_view_per_track, 3);
  EXPECT_EQ(opts.solver_base.solver_options.max_num_iterations, 200);
}

TEST(InlierThresholdOptions, DefaultConstruction) {
  InlierThresholdOptions opts;
  EXPECT_DOUBLE_EQ(opts.max_angle_error, 1.0);
  EXPECT_DOUBLE_EQ(opts.max_epipolar_error_E, 1.0);
  EXPECT_DOUBLE_EQ(opts.max_epipolar_error_F, 4.0);
  EXPECT_DOUBLE_EQ(opts.min_inlier_ratio, 0.25);
}

TEST(OptimizationBaseOptions, DefaultConstruction) {
  OptimizationBaseOptions opts;
  EXPECT_DOUBLE_EQ(opts.thres_loss_function, 0.1);
  EXPECT_EQ(opts.solver_options.max_num_iterations, 100);
  EXPECT_FALSE(opts.solver_options.minimizer_progress_to_stdout);
}

}  // namespace
}  // namespace colmap::glomap
