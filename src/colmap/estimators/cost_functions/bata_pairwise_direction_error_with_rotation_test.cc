// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.

#include "colmap/estimators/cost_functions/bata_pairwise_direction_error_with_rotation.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

// Smoke test: the class definition is well-formed and translates (compiles).
// Deeper functional tests live in the shared cost_functions_autodiff_test.cc
// which exercises each functor's `operator()` with a one-shot probe.
TEST(BATAPairwiseDirectionErrorWithRotation, HeaderCompiles) {
  // Empty body — presence of this test confirms header parses / links.
  SUCCEED();
}

}  // namespace
}  // namespace colmap
