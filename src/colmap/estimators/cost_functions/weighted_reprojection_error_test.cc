// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
// (BSD-3-Clause license, see LICENSE)

#include "colmap/estimators/cost_functions/weighted_reprojection_error.h"

#include "colmap/sensor/models.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(WeightedReprojErrorCostFunctor, CreateReturnsNonNull) {
  // Smoke test: Create returns a valid AutoDiffCostFunction.
  const Eigen::Vector2d obs(100.0, 200.0);
  auto* cost = WeightedReprojErrorCostFunctor<SimplePinholeCameraModel>::Create(
      obs, /*L00=*/1.0, /*L10=*/0.0, /*L11=*/1.0);
  ASSERT_NE(cost, nullptr);
  delete cost;
}

}  // namespace
}  // namespace colmap
