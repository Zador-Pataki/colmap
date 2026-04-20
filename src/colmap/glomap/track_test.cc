#include "colmap/glomap/track.h"

#include <gtest/gtest.h>

namespace colmap::glomap {
namespace {

TEST(Track, DefaultConstructor_ZeroXYZ) {
  Track t;
  EXPECT_EQ(t.xyz, Eigen::Vector3d::Zero());
  EXPECT_FALSE(t.is_initialized);
  EXPECT_TRUE(t.observations.empty());
  EXPECT_TRUE(t.lc_observations.empty());
}

TEST(Track, ObservationsWritable) {
  Track t;
  t.observations.emplace_back(image_t{1}, feature_t{2});
  EXPECT_EQ(t.observations.size(), 1u);
  EXPECT_EQ(t.observations.front().first, 1u);
  EXPECT_EQ(t.observations.front().second, 2u);
}

TEST(Track, LcObservationsWritable) {
  Track t;
  t.lc_observations.emplace_back(image_t{3}, feature_t{4});
  EXPECT_EQ(t.lc_observations.size(), 1u);
}

}  // namespace
}  // namespace colmap::glomap
