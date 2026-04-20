#include "colmap/glomap/gravity_info.h"

#include <gtest/gtest.h>

namespace colmap::glomap {
namespace {

TEST(GravityInfo, DefaultHasNoGravity) {
  GravityInfo g;
  EXPECT_FALSE(g.has_gravity);
}

TEST(GravityInfo, SetGravity_UpdatesHasGravityAndRAlign) {
  GravityInfo g;
  const Eigen::Vector3d grav(0.0, -9.81, 0.0);
  g.SetGravity(grav);
  EXPECT_TRUE(g.has_gravity);
  const Eigen::Vector3d expected_col1 = grav.normalized();
  EXPECT_NEAR((g.GetRAlign().col(1) - expected_col1).norm(), 0.0, 1e-12);
  EXPECT_NEAR(g.GetRAlign().determinant(), 1.0, 1e-9);
}

TEST(GravityInfo, GetGravity_ReturnsStoredVector) {
  GravityInfo g;
  const Eigen::Vector3d grav(1.0, 2.0, 3.0);
  g.SetGravity(grav);
  EXPECT_EQ(g.GetGravity(), grav);
}

}  // namespace
}  // namespace colmap::glomap
