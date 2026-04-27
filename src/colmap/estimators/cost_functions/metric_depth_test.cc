// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
// (BSD-3-Clause license, see LICENSE)

#include "colmap/estimators/cost_functions/metric_depth.h"

#include <cmath>
#include <limits>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ceres/ceres.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

// Helper: build a non-trivial rotation so we exercise the rotation-bake-in.
Eigen::Quaterniond MakeRotation() {
  return Eigen::Quaterniond(
             Eigen::AngleAxisd(0.3, Eigen::Vector3d(1, 2, 3).normalized()))
      .normalized();
}

// Helper: pick X / c such that ``z_est = (R * (X - c))[2]`` equals a target.
Eigen::Vector3d MakePointWithDepth(const Eigen::Quaterniond& rotation,
                                   const Eigen::Vector3d& camera_center,
                                   double target_z) {
  // Choose ``point_vec_cam = (0, 0, target_z)``; then ``X - c = R^T * vec``.
  const Eigen::Vector3d vec_cam(0.0, 0.0, target_z);
  const Eigen::Vector3d vec_world = rotation.inverse() * vec_cam;
  return camera_center + vec_world;
}

TEST(MetricDepthError, ZeroResidualWhenPriorMatchesPredicted) {
  const Eigen::Quaterniond rotation = MakeRotation();
  const double sigma_depth = 0.5;
  const double depth_prior = 7.0;
  const double dmap_scale = 1.0;
  const Eigen::Vector3d camera_center(1.0, -2.0, 3.0);
  const Eigen::Vector3d point3D =
      MakePointWithDepth(rotation, camera_center, dmap_scale * depth_prior);

  std::unique_ptr<ceres::CostFunction> cost_function(
      MetricDepthError::Create(rotation, depth_prior, sigma_depth));
  ASSERT_NE(cost_function, nullptr);

  double residual = std::numeric_limits<double>::quiet_NaN();
  const double* parameters[3] = {
      camera_center.data(), point3D.data(), &dmap_scale};
  EXPECT_TRUE(cost_function->Evaluate(parameters, &residual, nullptr));
  EXPECT_NEAR(residual, 0.0, 1e-9);
}

TEST(MetricDepthError, LinearResidualMatchesFormula) {
  const Eigen::Quaterniond rotation = MakeRotation();
  const double sigma_depth = 0.4;
  const double depth_prior = 5.0;
  const double dmap_scale = 1.2;
  const double z_est = 8.0;
  const Eigen::Vector3d camera_center(0.5, 0.5, -0.5);
  const Eigen::Vector3d point3D =
      MakePointWithDepth(rotation, camera_center, z_est);

  std::unique_ptr<ceres::CostFunction> cost_function(MetricDepthError::Create(
      rotation,
      depth_prior,
      sigma_depth,
      /*use_log_scale=*/false,
      /*use_log_residual=*/false));

  // residual = (z - s*m) / (s*sigma)
  const double expected =
      (z_est - dmap_scale * depth_prior) / (dmap_scale * sigma_depth);

  double residual = std::numeric_limits<double>::quiet_NaN();
  const double* parameters[3] = {
      camera_center.data(), point3D.data(), &dmap_scale};
  EXPECT_TRUE(cost_function->Evaluate(parameters, &residual, nullptr));
  EXPECT_NEAR(residual, expected, 1e-9);
}

TEST(MetricDepthError, LogResidualMatchesFormula) {
  const Eigen::Quaterniond rotation = MakeRotation();
  const double sigma_depth = 0.4;
  const double depth_prior = 5.0;
  const double dmap_scale = 1.2;
  const double z_est = 8.0;
  const Eigen::Vector3d camera_center(0.5, 0.5, -0.5);
  const Eigen::Vector3d point3D =
      MakePointWithDepth(rotation, camera_center, z_est);

  std::unique_ptr<ceres::CostFunction> cost_function(MetricDepthError::Create(
      rotation,
      depth_prior,
      sigma_depth,
      /*use_log_scale=*/false,
      /*use_log_residual=*/true));

  // sigma_log = sigma_depth / depth_prior; residual = log(z / (s*m)) / sigma_log
  const double sigma_log = sigma_depth / depth_prior;
  const double expected = std::log(z_est / (dmap_scale * depth_prior)) /
                          sigma_log;

  double residual = std::numeric_limits<double>::quiet_NaN();
  const double* parameters[3] = {
      camera_center.data(), point3D.data(), &dmap_scale};
  EXPECT_TRUE(cost_function->Evaluate(parameters, &residual, nullptr));
  EXPECT_NEAR(residual, expected, 1e-9);
}

TEST(MetricDepthError, ZeroResidualBehindCameraGate) {
  const Eigen::Quaterniond rotation = MakeRotation();
  const double sigma_depth = 0.5;
  const double depth_prior = 4.0;
  const double dmap_scale = 1.0;
  const Eigen::Vector3d camera_center(0.0, 0.0, 0.0);
  // Negative target z → point behind camera.
  const Eigen::Vector3d point3D =
      MakePointWithDepth(rotation, camera_center, -2.0);

  std::unique_ptr<ceres::CostFunction> cost_function(MetricDepthError::Create(
      rotation,
      depth_prior,
      sigma_depth,
      /*use_log_scale=*/false,
      /*use_log_residual=*/false,
      /*zero_residual_behind=*/true));

  double residual = std::numeric_limits<double>::quiet_NaN();
  const double* parameters[3] = {
      camera_center.data(), point3D.data(), &dmap_scale};
  EXPECT_TRUE(cost_function->Evaluate(parameters, &residual, nullptr));
  EXPECT_NEAR(residual, 0.0, 1e-12);

  // Sanity: without the gate, the residual is non-zero for the same input.
  std::unique_ptr<ceres::CostFunction> cost_function_no_gate(
      MetricDepthError::Create(rotation,
                               depth_prior,
                               sigma_depth,
                               /*use_log_scale=*/false,
                               /*use_log_residual=*/false,
                               /*zero_residual_behind=*/false));
  double residual_no_gate = std::numeric_limits<double>::quiet_NaN();
  EXPECT_TRUE(
      cost_function_no_gate->Evaluate(parameters, &residual_no_gate, nullptr));
  EXPECT_GT(std::abs(residual_no_gate), 1e-3);
}

TEST(MetricDepthError, LogScaleParameterization) {
  const Eigen::Quaterniond rotation = MakeRotation();
  const double sigma_depth = 0.3;
  const double depth_prior = 6.0;
  const double linear_scale = 0.8;
  const double log_scale = std::log(linear_scale);
  const Eigen::Vector3d camera_center(1.0, 0.0, 0.0);
  // Make z = linear_scale * depth_prior so linear-residual variant returns 0.
  const Eigen::Vector3d point3D =
      MakePointWithDepth(rotation, camera_center, linear_scale * depth_prior);

  std::unique_ptr<ceres::CostFunction> cost_function_log(
      MetricDepthError::Create(rotation,
                               depth_prior,
                               sigma_depth,
                               /*use_log_scale=*/true,
                               /*use_log_residual=*/false));

  double residual = std::numeric_limits<double>::quiet_NaN();
  const double* parameters[3] = {
      camera_center.data(), point3D.data(), &log_scale};
  EXPECT_TRUE(cost_function_log->Evaluate(parameters, &residual, nullptr));
  EXPECT_NEAR(residual, 0.0, 1e-9);
}

TEST(MetricDepthError, AllTogglesSmokeFinite) {
  const Eigen::Quaterniond rotation = MakeRotation();
  const double sigma_depth = 0.5;
  const double depth_prior = 5.0;
  const double dmap_scale = 1.0;
  const Eigen::Vector3d camera_center(0.0, 0.0, 0.0);
  const Eigen::Vector3d point3D =
      MakePointWithDepth(rotation, camera_center, 4.0);

  // Walk through each toggle in turn. ``smooth_transition`` is only
  // reachable via ``use_log_residual=true`` (per metric_depth.h header).
  struct Toggle {
    bool use_log_scale;
    bool use_log_residual;
    bool zero_residual_behind;
    bool smooth_transition;
  };
  const Toggle toggles[] = {
      {true, false, false, false},
      {false, true, false, false},
      {false, false, true, false},
      {false, true, false, true},
      {true, true, true, true},
  };

  for (const auto& t : toggles) {
    std::unique_ptr<ceres::CostFunction> cost_function(
        MetricDepthError::Create(rotation,
                                 depth_prior,
                                 sigma_depth,
                                 t.use_log_scale,
                                 t.use_log_residual,
                                 t.zero_residual_behind,
                                 t.smooth_transition,
                                 /*threshold=*/0.1));
    ASSERT_NE(cost_function, nullptr);
    // For the log-scale variants, dmap_scale must be log(scale); use 0 → s=1.
    const double scale_param = t.use_log_scale ? 0.0 : dmap_scale;
    const double* params[3] = {
        camera_center.data(), point3D.data(), &scale_param};
    double residual = std::numeric_limits<double>::quiet_NaN();
    EXPECT_TRUE(cost_function->Evaluate(params, &residual, nullptr));
    EXPECT_TRUE(std::isfinite(residual));
  }
}

TEST(MetricDepthError, RejectsNonPositiveSigma) {
  const Eigen::Quaterniond rotation = MakeRotation();
  ceres::CostFunction* cost_function =
      MetricDepthError::Create(rotation, /*depth_prior=*/1.0, /*sigma=*/0.0);
  EXPECT_EQ(cost_function, nullptr);
}

}  // namespace
}  // namespace colmap
