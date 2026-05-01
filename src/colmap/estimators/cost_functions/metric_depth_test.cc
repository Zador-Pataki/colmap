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

MetricDepthOptions MakeMetricDepthOptions(
    MetricDepthResidualType residual_type = MetricDepthResidualType::kLinear,
    bool use_log_scale = false,
    bool zero_residual_behind = false,
    double log_linear_threshold = 0.1) {
  MetricDepthOptions options;
  options.use_log_scale = use_log_scale;
  options.residual_type = residual_type;
  options.zero_residual_behind = zero_residual_behind;
  options.log_linear_threshold = log_linear_threshold;
  return options;
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

  std::unique_ptr<ceres::CostFunction> cost_function(
      MetricDepthError::Create(rotation, depth_prior, sigma_depth));

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
      MakeMetricDepthOptions(MetricDepthResidualType::kLog)));

  // sigma_log = sigma_depth / depth_prior; residual = log(z / (s*m)) /
  // sigma_log
  const double sigma_log = sigma_depth / depth_prior;
  const double expected =
      std::log(z_est / (dmap_scale * depth_prior)) / sigma_log;

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
      MakeMetricDepthOptions(MetricDepthResidualType::kLinear,
                             /*use_log_scale=*/false,
                             /*zero_residual_behind=*/true)));

  double residual = std::numeric_limits<double>::quiet_NaN();
  const double* parameters[3] = {
      camera_center.data(), point3D.data(), &dmap_scale};
  EXPECT_TRUE(cost_function->Evaluate(parameters, &residual, nullptr));
  EXPECT_NEAR(residual, 0.0, 1e-12);

  // Sanity: without the gate, the residual is non-zero for the same input.
  std::unique_ptr<ceres::CostFunction> cost_function_no_gate(
      MetricDepthError::Create(
          rotation, depth_prior, sigma_depth, MakeMetricDepthOptions()));
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
      MetricDepthError::Create(
          rotation,
          depth_prior,
          sigma_depth,
          MakeMetricDepthOptions(MetricDepthResidualType::kLinear,
                                 /*use_log_scale=*/true)));

  double residual = std::numeric_limits<double>::quiet_NaN();
  const double* parameters[3] = {
      camera_center.data(), point3D.data(), &log_scale};
  EXPECT_TRUE(cost_function_log->Evaluate(parameters, &residual, nullptr));
  EXPECT_NEAR(residual, 0.0, 1e-9);
}

TEST(MetricDepthError, AllOptionsSmokeFinite) {
  const Eigen::Quaterniond rotation = MakeRotation();
  const double sigma_depth = 0.5;
  const double depth_prior = 5.0;
  const double dmap_scale = 1.0;
  const Eigen::Vector3d camera_center(0.0, 0.0, 0.0);
  const Eigen::Vector3d point3D =
      MakePointWithDepth(rotation, camera_center, 4.0);

  const MetricDepthOptions options[] = {
      MakeMetricDepthOptions(MetricDepthResidualType::kLinear,
                             /*use_log_scale=*/true),
      MakeMetricDepthOptions(MetricDepthResidualType::kLog),
      MakeMetricDepthOptions(MetricDepthResidualType::kLinear,
                             /*use_log_scale=*/false,
                             /*zero_residual_behind=*/true),
      MakeMetricDepthOptions(MetricDepthResidualType::kLogLinear),
      MakeMetricDepthOptions(MetricDepthResidualType::kLogLinear,
                             /*use_log_scale=*/true,
                             /*zero_residual_behind=*/true),
  };

  for (const MetricDepthOptions& option : options) {
    std::unique_ptr<ceres::CostFunction> cost_function(
        MetricDepthError::Create(rotation, depth_prior, sigma_depth, option));
    ASSERT_NE(cost_function, nullptr);
    // For the log-scale variants, dmap_scale must be log(scale); use 0 → s=1.
    const double scale_param = option.use_log_scale ? 0.0 : dmap_scale;
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

TEST(MetricDepthError, RejectsNonPositiveLogLinearThreshold) {
  const Eigen::Quaterniond rotation = MakeRotation();
  MetricDepthOptions options;
  options.residual_type = MetricDepthResidualType::kLogLinear;
  options.log_linear_threshold = 0.0;

  ceres::CostFunction* cost_function = MetricDepthError::Create(
      rotation, /*depth_prior=*/1.0, /*sigma_depth=*/0.5, options);
  EXPECT_EQ(cost_function, nullptr);
}

}  // namespace
}  // namespace colmap
