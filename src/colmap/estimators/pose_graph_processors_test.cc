// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
// (BSD-3-Clause license, see LICENSE)

#include "colmap/estimators/pose_graph_processors.h"

#include "colmap/scene/camera.h"
#include "colmap/scene/image.h"
#include "colmap/scene/pose_graph.h"
#include "colmap/scene/two_view_geometry.h"
#include "colmap/util/types.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

static Camera MakePinholeCamera(double focal, double cx, double cy) {
  Camera cam;
  cam.model_id = CameraModelId::kPinhole;
  cam.width = 640;
  cam.height = 480;
  cam.params = {focal, focal, cx, cy};
  return cam;
}

static Image MakeImageWithPoints(
    camera_t camera_id,
    const std::vector<Eigen::Vector2d>& pts) {
  Image img;
  img.SetCameraId(camera_id);
  std::vector<Point2D> p2ds;
  p2ds.reserve(pts.size());
  for (const auto& xy : pts) {
    Point2D p;
    p.xy = xy;
    p2ds.push_back(p);
  }
  img.SetPoints2D(std::move(p2ds));
  return img;
}

// Helper: add a valid edge between (id1, id2) with given num_matches.
static void AddEdge(PoseGraph& pg,
                    image_t id1,
                    image_t id2,
                    int num_matches,
                    int total_matches = 0) {
  PoseGraph::Edge e;
  e.num_matches = num_matches;
  e.total_matches = total_matches > 0 ? total_matches : num_matches;
  pg.AddEdge(id1, id2, e);
}

// ---- AssignMdrpResults ---------------------------------------------------

TEST(AssignMdrpResultsTest, EmptyResultsReturnsZeroZero) {
  PoseGraph pg;
  AddEdge(pg, 1, 2, 10);
  std::unordered_map<image_pair_t, PoseGraph::MDRPResult> results;
  const auto [valid, invalid] = pg.AssignMdrpResults(results, 0.1);
  EXPECT_EQ(valid, 0);
  EXPECT_EQ(invalid, 0);
  // Edge untouched.
  EXPECT_TRUE(pg.IsValid(ImagePairToPairId(1, 2)));
}

TEST(AssignMdrpResultsTest, ValidResultUpdatesEdgeFields) {
  PoseGraph pg;
  AddEdge(pg, 1, 2, 5);
  const image_pair_t pid = ImagePairToPairId(1, 2);

  PoseGraph::MDRPResult res;
  res.is_valid = true;
  res.rel_depth_scale = 2.5;
  res.weight = 0.8;
  Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
  Eigen::Vector3d t(0.1, 0.0, 0.0);
  res.cam2_from_cam1 = Rigid3d(q, t);
  res.cov_t = Eigen::Matrix3d::Identity() * 0.01;

  std::unordered_map<image_pair_t, PoseGraph::MDRPResult> results;
  results[pid] = res;

  const auto [valid, invalid] = pg.AssignMdrpResults(results, 0.3);
  EXPECT_EQ(valid, 1);
  EXPECT_EQ(invalid, 0);

  const auto& edge = pg.Edges().at(pid);
  EXPECT_NEAR(edge.rel_depth_scale, 2.5, 1e-9);
  EXPECT_NEAR(edge.weight, 0.8, 1e-9);
  EXPECT_TRUE(edge.valid);
}

TEST(AssignMdrpResultsTest, InvalidResultSetsEdgeInvalid) {
  PoseGraph pg;
  AddEdge(pg, 1, 2, 5);
  const image_pair_t pid = ImagePairToPairId(1, 2);

  PoseGraph::MDRPResult res;
  res.is_valid = false;

  std::unordered_map<image_pair_t, PoseGraph::MDRPResult> results;
  results[pid] = res;

  const auto [valid, invalid] = pg.AssignMdrpResults(results, 0.3);
  EXPECT_EQ(valid, 0);
  EXPECT_EQ(invalid, 1);
  EXPECT_FALSE(pg.IsValid(pid));
}

TEST(AssignMdrpResultsTest, UnknownPairIdIsSkipped) {
  PoseGraph pg;
  AddEdge(pg, 1, 2, 5);

  PoseGraph::MDRPResult res;
  res.is_valid = true;
  const image_pair_t unknown_pid = ImagePairToPairId(99, 100);

  std::unordered_map<image_pair_t, PoseGraph::MDRPResult> results;
  results[unknown_pid] = res;

  const auto [valid, invalid] = pg.AssignMdrpResults(results, 0.3);
  EXPECT_EQ(valid, 0);
  EXPECT_EQ(invalid, 0);
}

// ---- ExtractValidPairData -----------------------------------------------

TEST(ExtractValidPairDataTest, ReturnsOnlyValidPairs) {
  PoseGraph pg;
  AddEdge(pg, 1, 2, 10);  // valid by default
  AddEdge(pg, 3, 4, 5);   // will be invalidated

  pg.SetInvalidEdge(ImagePairToPairId(3, 4));

  const PoseGraph::ValidPairData data = pg.ExtractValidPairData();
  EXPECT_EQ(data.pair_ids.size(), 1u);
  EXPECT_EQ(data.image_ids1.size(), 1u);
  EXPECT_EQ(data.image_ids2.size(), 1u);
  EXPECT_EQ(data.inlier_counts.size(), 1u);
}

TEST(ExtractValidPairDataTest, ArraysAreAligned) {
  PoseGraph pg;
  AddEdge(pg, 1, 2, 7);
  AddEdge(pg, 3, 4, 4);

  const PoseGraph::ValidPairData data = pg.ExtractValidPairData();
  EXPECT_EQ(data.pair_ids.size(), data.inlier_counts.size());
  EXPECT_EQ(data.pair_ids.size(), data.image_ids1.size());
  EXPECT_EQ(data.pair_ids.size(), data.image_ids2.size());
}

TEST(ExtractValidPairDataTest, EmptyGraphReturnsEmptyStruct) {
  PoseGraph pg;
  const PoseGraph::ValidPairData data = pg.ExtractValidPairData();
  EXPECT_TRUE(data.pair_ids.empty());
}

// ---- KeepLargestConnectedComponents -------------------------------------

TEST(KeepLargestConnectedComponentsTest, TwoClustersKeepsLargest) {
  PoseGraph pg;
  // Cluster A: 1-2-3 (3 nodes)
  AddEdge(pg, 1, 2, 10);
  AddEdge(pg, 2, 3, 10);
  // Cluster B: 4-5 (2 nodes)
  AddEdge(pg, 4, 5, 10);

  pg.KeepLargestConnectedComponents(1, 1);

  // Edges in cluster B should be invalid.
  EXPECT_FALSE(pg.IsValid(ImagePairToPairId(4, 5)));
  // Edges in cluster A should remain valid.
  EXPECT_TRUE(pg.IsValid(ImagePairToPairId(1, 2)));
  EXPECT_TRUE(pg.IsValid(ImagePairToPairId(2, 3)));
}

TEST(KeepLargestConnectedComponentsTest, SingleClusterPreserved) {
  PoseGraph pg;
  AddEdge(pg, 1, 2, 10);
  AddEdge(pg, 2, 3, 10);

  pg.KeepLargestConnectedComponents(1, 1);

  EXPECT_TRUE(pg.IsValid(ImagePairToPairId(1, 2)));
  EXPECT_TRUE(pg.IsValid(ImagePairToPairId(2, 3)));
}

TEST(KeepLargestConnectedComponentsTest, MinComponentSizeDropsSmallClusters) {
  PoseGraph pg;
  // Cluster A: 1-2 (2 nodes), Cluster B: 3-4-5 (3 nodes)
  AddEdge(pg, 1, 2, 10);
  AddEdge(pg, 3, 4, 10);
  AddEdge(pg, 4, 5, 10);

  // Keep up to 2 largest but only if >= 3 nodes.
  pg.KeepLargestConnectedComponents(2, 3);

  // Cluster A (2 nodes) is below min_component_size=3, dropped.
  EXPECT_FALSE(pg.IsValid(ImagePairToPairId(1, 2)));
  // Cluster B (3 nodes) is kept.
  EXPECT_TRUE(pg.IsValid(ImagePairToPairId(3, 4)));
  EXPECT_TRUE(pg.IsValid(ImagePairToPairId(4, 5)));
}

// ---- FilterInlierNum / FilterInlierRatio --------------------------------

TEST(FilterInlierNumTest, DropsBelowThreshold) {
  PoseGraph pg;
  AddEdge(pg, 1, 2, 5);   // below threshold 10
  AddEdge(pg, 3, 4, 15);  // above

  pg.FilterInlierNum(10);

  EXPECT_FALSE(pg.IsValid(ImagePairToPairId(1, 2)));
  EXPECT_TRUE(pg.IsValid(ImagePairToPairId(3, 4)));
}

TEST(FilterInlierNumTest, PreservesAtThreshold) {
  PoseGraph pg;
  AddEdge(pg, 1, 2, 10);  // exactly at threshold — must survive (strict <)

  pg.FilterInlierNum(10);

  EXPECT_TRUE(pg.IsValid(ImagePairToPairId(1, 2)));
}

TEST(FilterInlierRatioTest, DropsBelowRatio) {
  PoseGraph pg;
  // 3 inliers out of 10 total = 0.3 ratio.
  {
    PoseGraph::Edge e;
    e.num_matches = 3;
    e.total_matches = 10;
    pg.AddEdge(1, 2, e);
  }
  // 8 inliers out of 10 total = 0.8 ratio.
  {
    PoseGraph::Edge e;
    e.num_matches = 8;
    e.total_matches = 10;
    pg.AddEdge(3, 4, e);
  }

  pg.FilterInlierRatio(0.5);

  EXPECT_FALSE(pg.IsValid(ImagePairToPairId(1, 2)));
  EXPECT_TRUE(pg.IsValid(ImagePairToPairId(3, 4)));
}

TEST(FilterInlierRatioTest, PreservesAtThreshold) {
  PoseGraph pg;
  PoseGraph::Edge e;
  e.num_matches = 5;
  e.total_matches = 10;  // ratio = 0.5 exactly
  pg.AddEdge(1, 2, e);

  pg.FilterInlierRatio(0.5);

  EXPECT_TRUE(pg.IsValid(ImagePairToPairId(1, 2)));
}

// ---- UndistortImages ---------------------------------------------------

TEST(UndistortImagesTest, PopulatesFeaturesUndist) {
  std::unordered_map<camera_t, Camera> cameras;
  cameras[0] = MakePinholeCamera(500.0, 320.0, 240.0);

  std::unordered_map<image_t, Image> images;
  images[1] = MakeImageWithPoints(0, {{100.0, 200.0}, {300.0, 400.0}});

  UndistortImages(cameras, images);

  EXPECT_EQ(images[1].features_undist.size(), 2u);
}

TEST(UndistortImagesTest, PreservesOriginalPoints2D) {
  std::unordered_map<camera_t, Camera> cameras;
  cameras[0] = MakePinholeCamera(500.0, 320.0, 240.0);

  std::unordered_map<image_t, Image> images;
  images[1] = MakeImageWithPoints(0, {{100.0, 200.0}});

  UndistortImages(cameras, images);

  EXPECT_EQ(images[1].NumPoints2D(), 1u);
  EXPECT_NEAR(images[1].Point2D(0).xy.x(), 100.0, 1e-9);
}

TEST(UndistortImagesTest, AlreadyUndistortedNotReprocessed) {
  std::unordered_map<camera_t, Camera> cameras;
  cameras[0] = MakePinholeCamera(500.0, 320.0, 240.0);

  std::unordered_map<image_t, Image> images;
  images[1] = MakeImageWithPoints(0, {{100.0, 200.0}});
  // Pre-populate features_undist — should not be overwritten.
  images[1].features_undist.push_back({9.0, 9.0, 9.0});

  // clean_points=false (default) → skip already-undistorted images.
  UndistortImages(cameras, images, /*clean_points=*/false);

  EXPECT_NEAR(images[1].features_undist[0].x(), 9.0, 1e-9);
}

// ---- DecomposeRelPose --------------------------------------------------

TEST(DecomposeRelPoseTest, PopulatesCam2FromCam1OnValidPairs) {
  PoseGraph pg;
  {
    PoseGraph::Edge e;
    e.config = TwoViewGeometry::ConfigurationType::CALIBRATED;
    // Set E = skew(t) * R for identity pose.
    e.E = Eigen::Matrix3d::Zero();
    e.F = Eigen::Matrix3d::Zero();
    e.valid = true;
    pg.AddEdge(1, 2, e);
  }

  std::unordered_map<camera_t, Camera> cameras;
  cameras[0] = MakePinholeCamera(500.0, 320.0, 240.0);
  cameras[1] = MakePinholeCamera(500.0, 320.0, 240.0);

  std::unordered_map<image_t, Image> images;
  images[1] = MakeImageWithPoints(0, {{100.0, 100.0}, {200.0, 200.0}});
  images[2] = MakeImageWithPoints(1, {{110.0, 110.0}, {210.0, 210.0}});
  images[1].SetCameraId(0);
  images[2].SetCameraId(1);

  // Cameras without prior focal length skip pair — set it.
  cameras[0].has_prior_focal_length = true;
  cameras[1].has_prior_focal_length = true;

  // Should not crash on valid pair.
  DecomposeRelPose(pg, cameras, images);
}

// ---- UpdateImagePairsConfig --------------------------------------------

TEST(UpdateImagePairsConfigTest, InvalidPairsStayInvalid) {
  PoseGraph pg;
  PoseGraph::Edge e;
  e.valid = false;
  e.config = TwoViewGeometry::ConfigurationType::UNCALIBRATED;
  pg.AddEdge(1, 2, e);

  std::unordered_map<camera_t, Camera> cameras;
  cameras[0] = MakePinholeCamera(500.0, 320.0, 240.0);
  cameras[1] = MakePinholeCamera(500.0, 320.0, 240.0);

  std::unordered_map<image_t, Image> images;
  images[1] = MakeImageWithPoints(0, {{100.0, 200.0}});
  images[2] = MakeImageWithPoints(1, {{110.0, 210.0}});

  UpdateImagePairsConfig(pg, cameras, images);

  // Invalid pair must remain invalid regardless.
  EXPECT_FALSE(pg.IsValid(ImagePairToPairId(1, 2)));
}

}  // namespace
}  // namespace colmap
