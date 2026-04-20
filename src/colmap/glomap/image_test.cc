#include "colmap/glomap/image.h"

#include <gtest/gtest.h>

namespace colmap::glomap {
namespace {

TEST(Image, DefaultConstructor_HasExpectedDefaults) {
  Image img;
  EXPECT_EQ(img.cluster_id, -1);
  EXPECT_FALSE(img.is_registered);
  EXPECT_FALSE(img.gravity_info.has_gravity);
  EXPECT_EQ(img.log_scale, 0.0);
  EXPECT_EQ(img.log_scale_stddev, 0.0);
  EXPECT_EQ(img.gravity_sigma, 0.0);
}

TEST(Image, NamedConstructor_SetsImmutableFields) {
  Image img(42, 7, "foo.jpg");
  EXPECT_EQ(img.image_id, 42u);
  EXPECT_EQ(img.file_name, "foo.jpg");
  EXPECT_EQ(img.camera_id, 7u);
}

TEST(Image, DepthPriorsWritable) {
  Image img;
  img.depth_priors = {1.0, 2.0};
  img.depth_prior_stddevs = {0.1, 0.2};
  img.depth_prior_validity = {true, false};
  EXPECT_EQ(img.depth_priors.size(), 2u);
  EXPECT_EQ(img.depth_prior_validity[0], true);
}

TEST(Image, IsInlierBoolVectorWritable) {
  Image img;
  img.is_inlier = {true, false, true};
  EXPECT_EQ(img.is_inlier.size(), 3u);
}

TEST(Image, IsDepthOutlier_IsTrackAnchor_IsExcluded_AllWritable) {
  Image img;
  img.is_depth_outlier = {true};
  img.is_track_anchor = {false};
  img.is_excluded = {true};
  EXPECT_TRUE(img.is_depth_outlier[0]);
  EXPECT_FALSE(img.is_track_anchor[0]);
  EXPECT_TRUE(img.is_excluded[0]);
}

TEST(Image, AngularStddevsVectorsWritable) {
  Image img;
  img.angular_stddevs = {Eigen::Vector2d(0.1, 0.2)};
  img.angular_cholesky_xy = {Eigen::Vector3d(1.0, 0.1, 0.9)};
  img.angular_stddevs_z = {0.05};
  EXPECT_EQ(img.angular_stddevs.size(), 1u);
  EXPECT_EQ(img.angular_stddevs_z[0], 0.05);
}

TEST(Image, LogScaleFields_SmokeAssignRead) {
  Image img;
  img.log_scale = 0.25;
  img.log_scale_stddev = 0.1;
  EXPECT_EQ(img.log_scale, 0.25);
  EXPECT_EQ(img.log_scale_stddev, 0.1);
}

TEST(Image, GravityInfo_DefaultHasNoGravity) {
  Image img;
  EXPECT_FALSE(img.gravity_info.has_gravity);
}

TEST(Image, Center_ComputesInverseTranslation) {
  Image img(1, 0, "x");
  img.cam_from_world = colmap::Rigid3d(Eigen::Quaterniond::Identity(),
                                       Eigen::Vector3d(1.0, 2.0, 3.0));
  // Center = -R^T t = -t (identity rotation).
  const Eigen::Vector3d c = img.Center();
  EXPECT_NEAR(c.x(), -1.0, 1e-12);
  EXPECT_NEAR(c.y(), -2.0, 1e-12);
  EXPECT_NEAR(c.z(), -3.0, 1e-12);
}

}  // namespace
}  // namespace colmap::glomap
