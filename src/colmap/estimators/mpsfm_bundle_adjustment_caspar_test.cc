// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.

#include <cmath>
#include <string>

#include <Eigen/Core>
#include <gtest/gtest.h>

#if __has_include("colmap/estimators/mpsfm_bundle_adjustment_caspar.h")
#define COLMAP_MPSFM_CASPAR_API_AVAILABLE 1
#include "colmap/estimators/mpsfm_bundle_adjustment_caspar.h"
#else
#define COLMAP_MPSFM_CASPAR_API_AVAILABLE 0
#endif

namespace colmap {
namespace {

constexpr const char* kExpectedApi =
    "colmap/estimators/mpsfm_bundle_adjustment_caspar.h with "
    "MpsfmCasparBundleAdjustmentProblem, "
    "MpsfmCasparBundleAdjustmentSummary, MpsfmCasparDepthObservation, "
    "MpsfmCasparIntrinsicsPrior, MpsfmCasparIntrinsicsRandomWalk, "
    "MpsfmCasparScalePrior, and SolveMpsfmCasparBundleAdjustment";

Eigen::Vector4d PinholePriorResidual(const Eigen::Vector4d& params,
                                     const Eigen::Vector4d& prior,
                                     const Eigen::Vector4d& std_devs) {
  return (params - prior).cwiseQuotient(std_devs);
}

Eigen::Vector4d IntrinsicsRandomWalkResidual(
    const Eigen::Vector4d& prev,
    const Eigen::Vector4d& next,
    const Eigen::Vector4d& variance_per_frame,
    const int frame_gap) {
  return (next - prev)
      .cwiseQuotient((variance_per_frame * frame_gap).cwiseSqrt());
}

double ScalePriorResidual(const Eigen::Vector2d& shift_scale,
                          const double scale_std) {
  return shift_scale.y() / scale_std;
}

double LogDepthResidual(const double point_cam_z,
                        const double depth_prior,
                        const double scale) {
  if (point_cam_z <= 0.0) {
    return 0.0;
  }
  return std::log(point_cam_z) - (std::log(depth_prior) + scale);
}

double RobustRho(const MpsfmCasparLossType type,
                 const double squared_norm,
                 const double scale) {
  switch (type) {
    case MpsfmCasparLossType::TRIVIAL:
      return squared_norm;
    case MpsfmCasparLossType::SOFT_L1:
      return 2.0 * scale * scale *
             (std::sqrt(1.0 + squared_norm / (scale * scale)) - 1.0);
    case MpsfmCasparLossType::CAUCHY:
      return scale * scale * std::log(1.0 + squared_norm / (scale * scale));
    case MpsfmCasparLossType::HUBER:
      return squared_norm <= scale * scale
                 ? squared_norm
                 : 2.0 * scale * std::sqrt(squared_norm) - scale * scale;
  }
  return squared_norm;
}

double RobustifiedSquaredNorm(const Eigen::Vector2d& residual,
                              const MpsfmCasparLossType type,
                              const double scale,
                              const double magnitude) {
  const double s = residual.squaredNorm();
  if (s <= 1e-15) {
    return 0.0;
  }
  const double factor = std::sqrt(magnitude * RobustRho(type, s, scale) / s);
  return (factor * residual).squaredNorm();
}

TEST(MPSFMCasparObjectiveReference, PinholePriorResidualAndJacobian) {
  const Eigen::Vector4d params(102.0, 195.0, 321.0, 242.0);
  const Eigen::Vector4d prior(100.0, 200.0, 320.0, 240.0);
  const Eigen::Vector4d std_devs(2.0, 5.0, 0.5, 4.0);

  EXPECT_TRUE(PinholePriorResidual(params, prior, std_devs)
                  .isApprox(Eigen::Vector4d(1.0, -1.0, 2.0, 0.5)));

  const double eps = 1e-6;
  for (int i = 0; i < 4; ++i) {
    Eigen::Vector4d plus = params;
    Eigen::Vector4d minus = params;
    plus[i] += eps;
    minus[i] -= eps;
    const Eigen::Vector4d numerical =
        (PinholePriorResidual(plus, prior, std_devs) -
         PinholePriorResidual(minus, prior, std_devs)) /
        (2.0 * eps);
    Eigen::Vector4d expected = Eigen::Vector4d::Zero();
    expected[i] = 1.0 / std_devs[i];
    EXPECT_TRUE(numerical.isApprox(expected, 1e-8));
  }
}

TEST(MPSFMCasparObjectiveReference, IntrinsicsRandomWalkResidualAndJacobian) {
  const Eigen::Vector4d prev(100.0, 101.0, 320.0, 240.0);
  const Eigen::Vector4d next(102.0, 98.0, 321.5, 239.0);
  const Eigen::Vector4d variance_per_frame(0.25, 1.0, 4.0, 9.0);
  constexpr int kFrameGap = 4;

  const Eigen::Vector4d inv_std =
      (variance_per_frame * kFrameGap).cwiseSqrt().cwiseInverse();
  EXPECT_TRUE(
      IntrinsicsRandomWalkResidual(prev, next, variance_per_frame, kFrameGap)
          .isApprox((next - prev).cwiseProduct(inv_std)));

  const double eps = 1e-6;
  for (int i = 0; i < 4; ++i) {
    Eigen::Vector4d plus = next;
    Eigen::Vector4d minus = next;
    plus[i] += eps;
    minus[i] -= eps;
    const Eigen::Vector4d numerical =
        (IntrinsicsRandomWalkResidual(
             prev, plus, variance_per_frame, kFrameGap) -
         IntrinsicsRandomWalkResidual(
             prev, minus, variance_per_frame, kFrameGap)) /
        (2.0 * eps);
    Eigen::Vector4d expected = Eigen::Vector4d::Zero();
    expected[i] = inv_std[i];
    EXPECT_TRUE(numerical.isApprox(expected, 1e-8));
  }
}

TEST(MPSFMCasparObjectiveReference, ScalePriorIgnoresShift) {
  const Eigen::Vector2d shift_scale(3.5, -0.25);
  constexpr double kScaleStd = 0.125;

  EXPECT_DOUBLE_EQ(ScalePriorResidual(shift_scale, kScaleStd), -2.0);

  const double eps = 1e-6;
  const Eigen::Vector2d shift_plus(shift_scale.x() + eps, shift_scale.y());
  const Eigen::Vector2d shift_minus(shift_scale.x() - eps, shift_scale.y());
  const double d_shift = (ScalePriorResidual(shift_plus, kScaleStd) -
                          ScalePriorResidual(shift_minus, kScaleStd)) /
                         (2.0 * eps);
  EXPECT_DOUBLE_EQ(d_shift, 0.0);
}

TEST(MPSFMCasparObjectiveReference, LogDepthMatchesCeresSemantics) {
  constexpr double kDepthPrior = 3.25;
  constexpr double kScale = -0.2;
  constexpr double kPositiveZ = 4.5;

  EXPECT_DOUBLE_EQ(LogDepthResidual(-0.1, kDepthPrior, kScale), 0.0);
  EXPECT_DOUBLE_EQ(LogDepthResidual(kPositiveZ, kDepthPrior, kScale),
                   std::log(kPositiveZ) - (std::log(kDepthPrior) + kScale));

  constexpr double kEps = 1e-6;
  const double numerical_dz =
      (LogDepthResidual(kPositiveZ + kEps, kDepthPrior, kScale) -
       LogDepthResidual(kPositiveZ - kEps, kDepthPrior, kScale)) /
      (2.0 * kEps);
  EXPECT_NEAR(numerical_dz, 1.0 / kPositiveZ, 1e-10);
}

TEST(MPSFMCasparObjectiveReference, RobustLossPayloadMatchesCeresCost) {
  const Eigen::Vector2d residual(1.2, -0.4);
  constexpr double kScale = 0.7;
  constexpr double kMagnitude = 3.5;

  for (const MpsfmCasparLossType type :
       {MpsfmCasparLossType::TRIVIAL,
        MpsfmCasparLossType::SOFT_L1,
        MpsfmCasparLossType::CAUCHY,
        MpsfmCasparLossType::HUBER}) {
    EXPECT_NEAR(RobustifiedSquaredNorm(residual, type, kScale, kMagnitude),
                kMagnitude * RobustRho(type, residual.squaredNorm(), kScale),
                1e-12);
  }
}

TEST(MPSFMCasparBundleAdjustmentApi, ExpectedHeaderIsAvailable) {
#if COLMAP_MPSFM_CASPAR_API_AVAILABLE
  SUCCEED();
#else
  FAIL() << "Missing expected MPSFM Caspar BA API: " << kExpectedApi;
#endif
}

#if COLMAP_MPSFM_CASPAR_API_AVAILABLE
TEST(MPSFMCasparBundleAdjustmentApi, ExpectedPublicTypesAreDeclared) {
  MpsfmCasparBundleAdjustmentProblem problem;
  MpsfmCasparBundleAdjustmentSummary summary;
  MpsfmCasparDepthObservation depth_observation;
  MpsfmCasparIntrinsicsPrior intrinsics_prior;
  MpsfmCasparIntrinsicsRandomWalk intrinsics_random_walk;
  MpsfmCasparScalePrior scale_prior;

  static_cast<void>(problem);
  static_cast<void>(summary);
  static_cast<void>(depth_observation);
  static_cast<void>(intrinsics_prior);
  static_cast<void>(intrinsics_random_walk);
  static_cast<void>(scale_prior);
}

TEST(MPSFMCasparBundleAdjustmentApi, SummaryBriefReportIncludesDiagnostics) {
  MpsfmCasparBundleAdjustmentSummary summary;
  summary.num_residuals = 42;
  summary.num_reprojection_factors = 7;
  summary.num_depth_factors = 8;
  summary.num_intrinsics_prior_factors = 2;
  summary.num_intrinsics_random_walk_factors = 1;
  summary.num_scale_prior_factors = 3;
  summary.construction_time = 0.25;
  summary.solve_time = 0.5;
  summary.initial_score = 10.0;
  summary.final_score = 1.0;
  summary.iteration_count = 4;
  summary.backend_message = "test backend message";

  const std::string report = summary.BriefReport();
  EXPECT_NE(report.find("Residuals: 42"), std::string::npos);
  EXPECT_NE(report.find("reprojection=7"), std::string::npos);
  EXPECT_NE(report.find("depth=8"), std::string::npos);
  EXPECT_NE(report.find("intrinsics_prior=2"), std::string::npos);
  EXPECT_NE(report.find("intrinsics_random_walk=1"), std::string::npos);
  EXPECT_NE(report.find("scale_prior=3"), std::string::npos);
  EXPECT_NE(report.find("Solver iterations: 4"), std::string::npos);
  EXPECT_NE(report.find("initial=10"), std::string::npos);
  EXPECT_NE(report.find("final=1"), std::string::npos);
  EXPECT_NE(report.find("test backend message"), std::string::npos);
}
#endif

}  // namespace
}  // namespace colmap
