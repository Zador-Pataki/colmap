// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
// (BSD-3-Clause license, see LICENSE)

#include "colmap/sfm/track_establishment.h"

#include "colmap/scene/camera.h"
#include "colmap/scene/image.h"
#include "colmap/scene/pose_graph.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/scene/track.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

// Minimal reconstruction with N images, each having `num_points` Point2Ds.
// All images share a single trivial camera.
static Reconstruction MakeReconstruction(
    const std::vector<image_t>& image_ids, size_t num_points) {
  Reconstruction rec;
  Camera cam = Camera::CreateFromModelId(
      1, CameraModelId::kSimplePinhole, 1, 1, 1);
  rec.AddCameraWithTrivialRig(cam);

  for (image_t id : image_ids) {
    Image img;
    img.SetImageId(id);
    img.SetCameraId(cam.camera_id);
    img.SetPoints2D(std::vector<Eigen::Vector2d>(
        num_points, Eigen::Vector2d::Zero()));
    rec.AddImageWithTrivialFrame(std::move(img), Rigid3d{});
  }
  return rec;
}

// Add a Point3D covering one observation and link it.
static point3D_t AddPoint(Reconstruction& rec,
                          image_t image_id, point2D_t feat) {
  Track track;
  track.AddElement(image_id, feat);
  return rec.AddPoint3D(Eigen::Vector3d::Zero(), std::move(track));
}

// Helper: build a PoseGraph::Edge with one LC match.
static PoseGraph::Edge MakeLcEdge(point2D_t feat1, point2D_t feat2) {
  PoseGraph::Edge e;
  e.is_LC = true;
  e.valid = true;
  e.matches.push_back(Eigen::Vector2i(feat1, feat2));
  e.are_lc.push_back(true);
  return e;
}

TEST(TrackEngineTest, EmptyGraphIsNoop) {
  PoseGraph pose_graph;
  Reconstruction rec = MakeReconstruction({}, 0);
  TrackEngine engine;
  engine.ProcessLoopClosurePairs(pose_graph, rec);
  EXPECT_EQ(rec.NumPoints3D(), 0);
}

TEST(TrackEngineTest, NoLcEdgesIsNoop) {
  PoseGraph pose_graph;
  {
    PoseGraph::Edge e;
    e.is_LC = false;
    e.valid = true;
    e.matches.push_back(Eigen::Vector2i(0, 0));
    e.are_lc.push_back(true);
    pose_graph.AddEdge(1, 2, e);
  }
  Reconstruction rec = MakeReconstruction({1, 2}, 1);
  const point3D_t pid = AddPoint(rec, 1, 0);

  TrackEngine engine;
  engine.ProcessLoopClosurePairs(pose_graph, rec);

  EXPECT_EQ(rec.Point3D(pid).track.LcLength(), 0u);
}

TEST(TrackEngineTest, LcEdgeAddsLcElementsToBothTracks) {
  PoseGraph pose_graph;
  pose_graph.AddEdge(1, 2, MakeLcEdge(0, 0));

  Reconstruction rec = MakeReconstruction({1, 2}, 1);
  const point3D_t pid_a = AddPoint(rec, 1, 0);
  const point3D_t pid_b = AddPoint(rec, 2, 0);

  TrackEngine engine;
  engine.ProcessLoopClosurePairs(pose_graph, rec);

  EXPECT_EQ(rec.Point3D(pid_a).track.LcLength(), 1u);
  EXPECT_EQ(rec.Point3D(pid_b).track.LcLength(), 1u);
}

TEST(TrackEngineTest, PerMatchAreLC_OnlyFlaggedMatchAddsLC) {
  PoseGraph pose_graph;
  {
    PoseGraph::Edge e;
    e.is_LC = true;
    e.valid = true;
    // Two matches: only the second is LC.
    e.matches.push_back(Eigen::Vector2i(0, 0));
    e.are_lc.push_back(false);
    e.matches.push_back(Eigen::Vector2i(1, 1));
    e.are_lc.push_back(true);
    pose_graph.AddEdge(1, 2, e);
  }

  Reconstruction rec = MakeReconstruction({1, 2}, 2);
  const point3D_t pid_a = AddPoint(rec, 1, 0);
  const point3D_t pid_b = AddPoint(rec, 2, 0);
  const point3D_t pid_c = AddPoint(rec, 1, 1);
  const point3D_t pid_d = AddPoint(rec, 2, 1);

  TrackEngine engine;
  engine.ProcessLoopClosurePairs(pose_graph, rec);

  // Non-LC match (feat 0↔0) should NOT add LC links.
  EXPECT_EQ(rec.Point3D(pid_a).track.LcLength(), 0u);
  EXPECT_EQ(rec.Point3D(pid_b).track.LcLength(), 0u);
  // LC match (feat 1↔1) SHOULD add LC links.
  EXPECT_EQ(rec.Point3D(pid_c).track.LcLength(), 1u);
  EXPECT_EQ(rec.Point3D(pid_d).track.LcLength(), 1u);
}

TEST(TrackEngineTest, FilterBothInSetProcesses) {
  PoseGraph pose_graph;
  pose_graph.AddEdge(1, 2, MakeLcEdge(0, 0));

  Reconstruction rec = MakeReconstruction({1, 2}, 1);
  const point3D_t pid_a = AddPoint(rec, 1, 0);
  const point3D_t pid_b = AddPoint(rec, 2, 0);

  TrackEngine engine;
  engine.SetImageIdsToProcess({1, 2});
  engine.ProcessLoopClosurePairs(pose_graph, rec);

  EXPECT_EQ(rec.Point3D(pid_a).track.LcLength(), 1u);
  EXPECT_EQ(rec.Point3D(pid_b).track.LcLength(), 1u);
}

TEST(TrackEngineTest, FilterNeitherInSetSkips) {
  PoseGraph pose_graph;
  pose_graph.AddEdge(3, 4, MakeLcEdge(0, 0));

  Reconstruction rec = MakeReconstruction({3, 4}, 1);
  const point3D_t pid_a = AddPoint(rec, 3, 0);
  const point3D_t pid_b = AddPoint(rec, 4, 0);

  TrackEngine engine;
  engine.SetImageIdsToProcess({1, 2});  // neither 3 nor 4 in set
  engine.ProcessLoopClosurePairs(pose_graph, rec);

  EXPECT_EQ(rec.Point3D(pid_a).track.LcLength(), 0u);
  EXPECT_EQ(rec.Point3D(pid_b).track.LcLength(), 0u);
}

TEST(TrackEngineTest, FilterOnlyOneInSetSkips) {
  PoseGraph pose_graph;
  pose_graph.AddEdge(1, 4, MakeLcEdge(0, 0));

  Reconstruction rec = MakeReconstruction({1, 4}, 1);
  const point3D_t pid_a = AddPoint(rec, 1, 0);
  const point3D_t pid_b = AddPoint(rec, 4, 0);

  TrackEngine engine;
  engine.SetImageIdsToProcess({1, 2});  // image 1 in set, image 4 not
  engine.ProcessLoopClosurePairs(pose_graph, rec);

  // Both should be skipped because image 4 is not in the set.
  EXPECT_EQ(rec.Point3D(pid_a).track.LcLength(), 0u);
  EXPECT_EQ(rec.Point3D(pid_b).track.LcLength(), 0u);
}

TEST(TrackEngineTest, SameTrackBothSidesNoLcLink) {
  PoseGraph pose_graph;
  pose_graph.AddEdge(1, 2, MakeLcEdge(0, 0));

  // Single Point3D spanning both images.
  Reconstruction rec = MakeReconstruction({1, 2}, 1);
  Track t;
  t.AddElement(1, 0);
  t.AddElement(2, 0);
  const point3D_t pid = rec.AddPoint3D(Eigen::Vector3d::Zero(), std::move(t));

  TrackEngine engine;
  engine.ProcessLoopClosurePairs(pose_graph, rec);

  // Same Point3D on both sides → no LC link added.
  EXPECT_EQ(rec.Point3D(pid).track.LcLength(), 0u);
}

TEST(TrackEngineTest, OrphanObsCreatesStubPoint3D) {
  // Neither feat exists in any Point3D → one stub Point3D created with
  // two lc_elements.
  PoseGraph pose_graph;
  pose_graph.AddEdge(1, 2, MakeLcEdge(5, 7));

  Reconstruction rec = MakeReconstruction({1, 2}, 10);

  TrackEngine engine;
  engine.ProcessLoopClosurePairs(pose_graph, rec);

  // One stub Point3D created.
  EXPECT_EQ(rec.NumPoints3D(), 1u);
  const auto& [pid, p3d] = *rec.Points3D().begin();
  EXPECT_EQ(p3d.track.Length(), 0u);   // no regular elements
  EXPECT_EQ(p3d.track.LcLength(), 2u); // two LC observations
}

TEST(TrackEngineTest, OnlyOneSideHasTrack) {
  // feat1 in image1 has a Point3D; feat2 in image2 does not.
  PoseGraph pose_graph;
  pose_graph.AddEdge(1, 2, MakeLcEdge(0, 3));

  Reconstruction rec = MakeReconstruction({1, 2}, 5);
  const point3D_t pid_a = AddPoint(rec, 1, 0);
  // image2, feat3 intentionally left unlinked.

  TrackEngine engine;
  engine.ProcessLoopClosurePairs(pose_graph, rec);

  // One-sided LC: feat2 from image2 added as LC to pid_a.
  EXPECT_EQ(rec.Point3D(pid_a).track.LcLength(), 1u);
  EXPECT_EQ(rec.Point3D(pid_a).track.LcElements()[0].image_id, 2u);
  EXPECT_EQ(rec.Point3D(pid_a).track.LcElements()[0].point2D_idx, 3u);
}

}  // namespace
}  // namespace colmap
