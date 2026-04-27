// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Tests for ImagePairsInlierCount: depth-aware MDRP re-scorer with
// 4-gate degeneracy filtering (epipolar / cheirality / triangulation
// angle / angle-from-epipole, where the epipole gate is depth-aware).

#include "colmap/sfm/image_pair_inliers.h"

#include "colmap/geometry/essential_matrix.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/scene/camera.h"
#include "colmap/scene/correspondence_graph.h"
#include "colmap/scene/image.h"
#include "colmap/scene/two_view_geometry.h"
#include "colmap/sensor/models.h"
#include "colmap/util/types.h"

#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtest/gtest.h>

namespace colmap {
namespace {

// Build a calibrated pinhole camera with focal=1 and principal point at 0.
// MeanFocalLength = 1 keeps the essential-matrix epipolar threshold equal to
// `options.max_epipolar_error_E`, simplifying expectations.
Camera MakeUnitCamera(camera_t camera_id) {
  return Camera::CreateFromModelId(
      camera_id, CameraModelId::kSimplePinhole, /*focal_length=*/1.0,
      /*width=*/100, /*height=*/100);
}

// Helper: build a canonical CALIBRATED pair where image 1 is at the origin
// and image 2 is translated along +x. World 3D points project cleanly into
// both images. ``world_points`` are 3D points with positive z in cam1 frame.
//
// Sets:
//   - cameras (camera 1 used by both images)
//   - images[id1].features_undist[k] = normalized ray to point k in cam1
//   - images[id2].features_undist[k] = normalized ray to point k in cam2
//   - images[id*].features[k] = pixel (== ray.head<2>() since f=1, c=0)
//   - depth_prior_validity[k] = true for all k by default
//   - corr_graph: ImagePair with config CALIBRATED and matches[k] = (k,k)
struct TwoViewSetup {
  std::unordered_map<camera_t, Camera> cameras;
  std::unordered_map<image_t, Image> images;
  CorrespondenceGraph corr_graph;
  image_t image_id1 = 1;
  image_t image_id2 = 2;
  camera_t camera_id = 1;
  Rigid3d cam2_from_cam1;
};

TwoViewSetup MakeTwoView(const std::vector<Eigen::Vector3d>& cam1_points,
                         const Rigid3d& cam2_from_cam1) {
  TwoViewSetup setup;
  setup.cam2_from_cam1 = cam2_from_cam1;
  setup.cameras.emplace(setup.camera_id, MakeUnitCamera(setup.camera_id));

  Image img1;
  img1.SetImageId(setup.image_id1);
  img1.SetCameraId(setup.camera_id);
  Image img2;
  img2.SetImageId(setup.image_id2);
  img2.SetCameraId(setup.camera_id);

  for (const auto& p1 : cam1_points) {
    Eigen::Vector3d p2 = cam2_from_cam1 * p1;
    Eigen::Vector3d r1 = p1.normalized();
    Eigen::Vector3d r2 = p2.normalized();
    img1.features_undist.push_back(r1);
    img2.features_undist.push_back(r2);
    // For pinhole f=1 c=0, pixel = (x/z, y/z) of the camera-frame point.
    img1.features.emplace_back(p1.x() / p1.z(), p1.y() / p1.z());
    img2.features.emplace_back(p2.x() / p2.z(), p2.y() / p2.z());
    img1.depth_prior_validity.push_back(true);
    img2.depth_prior_validity.push_back(true);
    img1.depth_priors.push_back(p1.z());
    img2.depth_priors.push_back(p2.z());
  }

  setup.images.emplace(setup.image_id1, std::move(img1));
  setup.images.emplace(setup.image_id2, std::move(img2));

  // Create the image pair.
  CorrespondenceGraph::ImagePair image_pair(setup.image_id1, setup.image_id2);
  image_pair.is_valid = true;
  image_pair.two_view_geometry.config = TwoViewGeometry::CALIBRATED;
  image_pair.two_view_geometry.cam2_from_cam1 = cam2_from_cam1;
  image_pair.two_view_geometry.E = EssentialMatrixFromPose(cam2_from_cam1);

  const int n = static_cast<int>(cam1_points.size());
  Eigen::MatrixXi matches(n, 2);
  for (int k = 0; k < n; ++k) {
    matches(k, 0) = k;
    matches(k, 1) = k;
  }
  image_pair.matches = std::move(matches);

  const image_pair_t pair_id =
      ImagePairToPairId(setup.image_id1, setup.image_id2);
  setup.corr_graph.MutableImagePairs().emplace(pair_id, std::move(image_pair));
  return setup;
}

// Helper: pull the ImagePair back out of the corr_graph after running
// ImagePairsInlierCount so we can inspect ``inliers``.
const CorrespondenceGraph::ImagePair& GetPair(const TwoViewSetup& setup) {
  const image_pair_t pair_id =
      ImagePairToPairId(setup.image_id1, setup.image_id2);
  return setup.corr_graph.ImagePairsMap().at(pair_id);
}

// Default identity rotation + +x baseline; depth-1 plane yields wide
// triangulation angle and points well off the epipole (which lies at +x).
Rigid3d DefaultPose() {
  return Rigid3d(Eigen::Quaterniond::Identity(),
                 Eigen::Vector3d(1.0, 0.0, 0.0));
}

InlierThresholdOptions DefaultOptions() {
  InlierThresholdOptions options;
  // Generous so we accept clean inliers but reject perturbed ones in the
  // EpipolarGateDrops test.
  options.max_epipolar_error_E = 1e-2;
  options.min_angle_from_epipole = 3.0;     // depth-prior valid path
  options.thres_epipole = 3.0;
  options.thres_epipole_nodepth = 3.0;
  return options;
}

// ---------------------------------------------------------------------------
// Test 1: All clean matches pass all four gates. counter == matches.size().
// ---------------------------------------------------------------------------
TEST(ImagePairInlierCount, AllInliersPassFourGates) {
  // Five 3D points spread across the cam1 image plane at depth ~5; +x
  // baseline puts the epipole on the +x axis so y-spread points are well
  // off-axis.
  std::vector<Eigen::Vector3d> pts = {
      {0.0, 0.0, 5.0},
      {0.0, 0.5, 5.0},
      {0.0, -0.5, 5.0},
      {-0.3, 0.4, 5.0},
      {-0.4, -0.3, 5.0},
  };
  auto setup = MakeTwoView(pts, DefaultPose());
  ImagePairsInlierCount(setup.corr_graph, setup.cameras, setup.images,
                        DefaultOptions(), /*clean_inliers=*/true);
  const auto& pair = GetPair(setup);
  EXPECT_EQ(pair.inliers.size(), pts.size());
}

// ---------------------------------------------------------------------------
// Test 2: Inject Sampson noise into half the matches by perturbing
// features_undist on image 2 along the y-axis. Default
// max_epipolar_error_E = 1e-2 means perturbations >> 1e-2 fail Gate 1.
// ---------------------------------------------------------------------------
TEST(ImagePairInlierCount, EpipolarGateDrops) {
  std::vector<Eigen::Vector3d> pts = {
      {0.0, 0.0, 5.0},
      {0.0, 0.5, 5.0},
      {0.0, -0.5, 5.0},
      {-0.3, 0.4, 5.0},
      {-0.4, -0.3, 5.0},
      {0.2, 0.2, 5.0},
  };
  auto setup = MakeTwoView(pts, DefaultPose());

  // Perturb features_undist on image 2 by a large y-offset for indices 0,2,4
  // (3 of 6). Re-normalize to keep them as unit rays.
  auto& img2 = setup.images.at(setup.image_id2);
  for (int k : {0, 2, 4}) {
    img2.features_undist[k] += Eigen::Vector3d(0.0, 0.5, 0.0);
    img2.features_undist[k].normalize();
  }
  ImagePairsInlierCount(setup.corr_graph, setup.cameras, setup.images,
                        DefaultOptions(), /*clean_inliers=*/true);
  const auto& pair = GetPair(setup);
  EXPECT_EQ(pair.inliers.size(), 3u);  // 6 - 3 dropped
  // Only the unperturbed indices {1, 3, 5} should survive.
  for (int k : pair.inliers) {
    EXPECT_TRUE(k == 1 || k == 3 || k == 5) << "k=" << k;
  }
}

// ---------------------------------------------------------------------------
// Test 3: A 3D point behind the cam1 OR cam2 should fail cheirality even if
// epipolar geometry holds. We can't trivially place a behind-camera ray AND
// keep zero Sampson error, but ``CheckCheirality`` runs as Gate 2 only after
// Gate 1 passes — so we set up matches where the ray pair satisfies the
// epipolar constraint but corresponds to a virtual point behind cam2.
//
// Construction: pick a 3D point in cam1 frame with negative z by mirroring
// — both cam1 and cam2 see "rays" along (small_x, small_y, -1). Sampson
// error with E (which is bilinear in rays) still hits ~0 because flipping
// signs of both rays preserves x1^T E x2.
// ---------------------------------------------------------------------------
TEST(ImagePairInlierCount, CheiralityGateDrops) {
  // Build a clean configuration first.
  std::vector<Eigen::Vector3d> pts = {
      {0.0, 0.0, 5.0},
      {0.0, 0.5, 5.0},
  };
  auto setup = MakeTwoView(pts, DefaultPose());

  // Now flip BOTH features_undist for index 0 to negate the z component.
  // x1^T E x2 = (-x1)^T E (-x2) so Sampson error is preserved, but the
  // triangulated depth becomes negative — Gate 2 fails.
  auto& img1 = setup.images.at(setup.image_id1);
  auto& img2 = setup.images.at(setup.image_id2);
  img1.features_undist[0] = -img1.features_undist[0];
  img2.features_undist[0] = -img2.features_undist[0];

  ImagePairsInlierCount(setup.corr_graph, setup.cameras, setup.images,
                        DefaultOptions(), /*clean_inliers=*/true);
  const auto& pair = GetPair(setup);
  // Index 0 should fail cheirality; index 1 should still pass.
  EXPECT_EQ(pair.inliers.size(), 1u);
  EXPECT_EQ(pair.inliers.front(), 1);
}

// ---------------------------------------------------------------------------
// Test 4: A match whose ray direction in image 2 is very close to the
// epipole-from-cam1 direction (here the +x axis since translation is +x)
// should fail Gate 4. We keep epipolar error ~0 by placing both rays on the
// epipolar plane.
//
// Concretely: with cam2_from_cam1 = (I, [1,0,0]), the epipole on image 2 is
// the projection of the cam1 center, i.e. -(cam2_from_cam1 R^T t1) = ... the
// ray direction along +x (in cam2 frame). A point in cam1 with very small
// y/z and large +x will produce a cam2 ray very near (+x).
// ---------------------------------------------------------------------------
TEST(ImagePairInlierCount, EpipoleGateDrops) {
  std::vector<Eigen::Vector3d> pts = {
      // Index 0: large +x in cam1 frame -> nearly along epipole on cam2.
      {100.0, 0.0, 1.0},
      // Index 1: clean off-axis baseline, well off epipole.
      {0.0, 0.5, 5.0},
  };
  auto setup = MakeTwoView(pts, DefaultPose());

  InlierThresholdOptions options = DefaultOptions();
  // Use a strict epipole gate (degrees) so the near-epipole match fails.
  options.min_angle_from_epipole = 30.0;
  options.thres_epipole_nodepth = 30.0;

  ImagePairsInlierCount(setup.corr_graph, setup.cameras, setup.images,
                        options, /*clean_inliers=*/true);
  const auto& pair = GetPair(setup);
  // Index 0 should fail Gate 4; index 1 should pass.
  EXPECT_EQ(pair.inliers.size(), 1u);
  EXPECT_EQ(pair.inliers.front(), 1);
}

// ---------------------------------------------------------------------------
// Test 5: With depth_prior_validity all-true vs all-false, the epipole gate
// uses ``min_angle_from_epipole`` vs ``3.0`` (hard-coded "nodepth" path).
// Set min_angle_from_epipole = 30°; place a match at ~10° from epipole — it
// should fail under the depth-prior path (30°) but pass under nodepth (3°).
// ---------------------------------------------------------------------------
TEST(ImagePairInlierCount, DepthPriorOnVsOff) {
  // Place a point so the cam2 ray sits ~10° away from the epipole. With +x
  // baseline (epipole21 ≈ +x in cam2), choose cam1 point (10, 0, 1):
  //   cam2 point = (11, 0, 1), normalized ray ≈ (0.9959, 0, 0.0905)
  //   angle from +x ≈ atan2(0.0905, 0.9959) ≈ 5.2°.
  // To get exactly between 3° and 30°, use (5, 0, 1) -> cam2 (6, 0, 1) ->
  // angle atan2(1/sqrt(37), 6/sqrt(37)) = atan2(1, 6) ≈ 9.46°.
  std::vector<Eigen::Vector3d> pts = {
      {5.0, 0.0, 1.0},
  };

  // Run 1: depth_prior_validity = all true (depth path uses 30°).
  {
    auto setup = MakeTwoView(pts, DefaultPose());
    InlierThresholdOptions options = DefaultOptions();
    options.min_angle_from_epipole = 30.0;       // depth-prior threshold
    options.thres_epipole_nodepth = 3.0;         // nodepth fallback (unused)
    ImagePairsInlierCount(setup.corr_graph, setup.cameras, setup.images,
                          options, /*clean_inliers=*/true);
    const auto& pair = GetPair(setup);
    // ~9.46° < 30° -> fails depth-prior epipole gate.
    EXPECT_EQ(pair.inliers.size(), 0u);
  }

  // Run 2: depth_prior_validity = all false (nodepth path uses 3°).
  {
    auto setup = MakeTwoView(pts, DefaultPose());
    auto& img1 = setup.images.at(setup.image_id1);
    auto& img2 = setup.images.at(setup.image_id2);
    std::fill(img1.depth_prior_validity.begin(),
              img1.depth_prior_validity.end(), false);
    std::fill(img2.depth_prior_validity.begin(),
              img2.depth_prior_validity.end(), false);

    InlierThresholdOptions options = DefaultOptions();
    options.min_angle_from_epipole = 30.0;
    options.thres_epipole_nodepth = 3.0;
    ImagePairsInlierCount(setup.corr_graph, setup.cameras, setup.images,
                          options, /*clean_inliers=*/true);
    const auto& pair = GetPair(setup);
    // ~9.46° > 3° -> passes nodepth epipole gate.
    EXPECT_EQ(pair.inliers.size(), 1u);
  }
}

}  // namespace
}  // namespace colmap
