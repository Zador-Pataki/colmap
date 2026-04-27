// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "colmap/sfm/track_establishment.h"

#include "colmap/scene/correspondence_graph.h"
#include "colmap/util/types.h"

#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

namespace colmap {
namespace {

// Build an ImagePair entry directly into the corr_graph's image_pairs map
// with the glomap-fork-extension fields ``matches`` and ``inliers`` populated.
// ``EstablishTracksFromCorrGraph`` only reads those two fields plus the pair
// keys, so we bypass ``AddTwoViewGeometry`` and the colmap flat_corrs path.
void AddImagePair(CorrespondenceGraph& corr_graph,
                  image_t image_id1,
                  image_t image_id2,
                  const std::vector<std::pair<int, int>>& matches,
                  const std::vector<int>& inliers) {
  const image_pair_t pair_id = ImagePairToPairId(image_id1, image_id2);
  CorrespondenceGraph::ImagePair image_pair(image_id1, image_id2);
  Eigen::MatrixXi matches_mat(static_cast<int>(matches.size()), 2);
  for (size_t i = 0; i < matches.size(); ++i) {
    matches_mat(static_cast<int>(i), 0) = matches[i].first;
    matches_mat(static_cast<int>(i), 1) = matches[i].second;
  }
  image_pair.matches = std::move(matches_mat);
  image_pair.inliers = inliers;
  image_pair.num_matches = static_cast<point2D_t>(inliers.size());
  corr_graph.MutableImagePairs().emplace(pair_id, std::move(image_pair));
}

// Build a map ``image_id -> [Vector2d(0,0), Vector2d(1,1), ...]`` so all
// keypoints on a given image are well-separated (intra-image consistency
// trivially holds when all features are at distinct integer pixels).
std::unordered_map<image_t, std::vector<Eigen::Vector2d>>
MakeWellSeparatedKeypoints(const std::vector<image_t>& image_ids,
                           int num_points_per_image) {
  std::unordered_map<image_t, std::vector<Eigen::Vector2d>> result;
  for (const image_t image_id : image_ids) {
    auto& kps = result[image_id];
    kps.reserve(num_points_per_image);
    for (int i = 0; i < num_points_per_image; ++i) {
      // Spread keypoints 100px apart on both axes so every intra-image
      // pair is far above the default 10px threshold.
      kps.emplace_back(100.0 * i, 100.0 * i);
    }
  }
  return result;
}

// Helper: collect keys of corr_graph.image_pairs into a vector.
std::vector<image_pair_t> CollectPairIds(const CorrespondenceGraph& g) {
  std::vector<image_pair_t> ids;
  ids.reserve(g.ImagePairsMap().size());
  for (const auto& [pair_id, _] : g.ImagePairsMap()) ids.push_back(pair_id);
  return ids;
}

// 3 images, 3 valid pairs (1-2, 1-3, 2-3), 5 inlier matches per pair, all
// pointing to the same 5 underlying 3D points (feature index i corresponds
// to point i on every image). Expect 5 tracks of length 3.
TEST(TrackEstablishment, Basic3PairTriangle) {
  CorrespondenceGraph corr_graph;
  corr_graph.AddImage(1, 5);
  corr_graph.AddImage(2, 5);
  corr_graph.AddImage(3, 5);

  std::vector<std::pair<int, int>> matches;
  std::vector<int> inliers;
  for (int i = 0; i < 5; ++i) {
    matches.emplace_back(i, i);
    inliers.push_back(i);
  }
  AddImagePair(corr_graph, 1, 2, matches, inliers);
  AddImagePair(corr_graph, 1, 3, matches, inliers);
  AddImagePair(corr_graph, 2, 3, matches, inliers);

  const auto keypoints = MakeWellSeparatedKeypoints({1, 2, 3}, 5);
  const TrackEstablishmentOptions options;  // defaults: min_views=3
  const auto tracks = EstablishTracksFromCorrGraph(
      CollectPairIds(corr_graph), corr_graph, keypoints, options);

  EXPECT_EQ(tracks.size(), 5);
  for (const auto& [pid, point3D] : tracks) {
    EXPECT_EQ(point3D.track.Length(), 3);
  }
}

// Same geometry but ``min_num_views_per_track = 4`` rejects every length-3
// track.
TEST(TrackEstablishment, LengthFilterDropsShortTracks) {
  CorrespondenceGraph corr_graph;
  corr_graph.AddImage(1, 5);
  corr_graph.AddImage(2, 5);
  corr_graph.AddImage(3, 5);

  std::vector<std::pair<int, int>> matches;
  std::vector<int> inliers;
  for (int i = 0; i < 5; ++i) {
    matches.emplace_back(i, i);
    inliers.push_back(i);
  }
  AddImagePair(corr_graph, 1, 2, matches, inliers);
  AddImagePair(corr_graph, 1, 3, matches, inliers);
  AddImagePair(corr_graph, 2, 3, matches, inliers);

  const auto keypoints = MakeWellSeparatedKeypoints({1, 2, 3}, 5);
  TrackEstablishmentOptions options;
  options.min_num_views_per_track = 4;
  const auto tracks = EstablishTracksFromCorrGraph(
      CollectPairIds(corr_graph), corr_graph, keypoints, options);

  EXPECT_TRUE(tracks.empty());
}

// Drive two distinct features on image 1 (idx 0 and idx 1) into the same
// union-find root via image 2's feature 0:
//   pair (1,2): match (img1:0 <-> img2:0)
//   pair (2,3): match (img2:0 <-> img3:0) -- chain via image 3
//   pair (1,3): match (img1:1 <-> img3:0) -- this fuses img1:0 ~ img1:1
// keypoints for img1:0 = (0,0) and img1:1 = (1000,1000) -> intra-image
// consistency violated (>10px), track dropped.
// 4 images so the surviving tracks (built from idx 1..4) still have length 3.
TEST(TrackEstablishment, IntraImageConsistencyDropsInconsistentTrack) {
  CorrespondenceGraph corr_graph;
  corr_graph.AddImage(1, 5);
  corr_graph.AddImage(2, 5);
  corr_graph.AddImage(3, 5);

  // Inconsistent fusion path for feature 0 chain.
  AddImagePair(corr_graph, 1, 2, {{0, 0}}, {0});
  AddImagePair(corr_graph, 2, 3, {{0, 0}}, {0});
  AddImagePair(corr_graph, 1, 3, {{1, 0}}, {0});

  // Keypoints: image 1 has feature 0 and feature 1 placed FAR apart so the
  // intra-image consistency check rejects the merged track.
  std::unordered_map<image_t, std::vector<Eigen::Vector2d>> keypoints;
  keypoints[1] = {Eigen::Vector2d(0, 0), Eigen::Vector2d(1000, 1000),
                  Eigen::Vector2d(200, 200), Eigen::Vector2d(300, 300),
                  Eigen::Vector2d(400, 400)};
  keypoints[2] = {Eigen::Vector2d(0, 0), Eigen::Vector2d(100, 100),
                  Eigen::Vector2d(200, 200), Eigen::Vector2d(300, 300),
                  Eigen::Vector2d(400, 400)};
  keypoints[3] = {Eigen::Vector2d(0, 0), Eigen::Vector2d(100, 100),
                  Eigen::Vector2d(200, 200), Eigen::Vector2d(300, 300),
                  Eigen::Vector2d(400, 400)};

  TrackEstablishmentOptions options;
  options.min_num_views_per_track = 2;  // accept length-2 if any survived
  options.intra_image_consistency_threshold = 10.0;
  const auto tracks = EstablishTracksFromCorrGraph(
      CollectPairIds(corr_graph), corr_graph, keypoints, options);

  // The single track we constructed is inconsistent; nothing else exists.
  EXPECT_TRUE(tracks.empty());
}

// ``ignore_match`` returning true for any match touching image 1 strips
// image 1's contribution from union-find. The remaining 5 tracks then have
// length 2 (only images 2 and 3) so default min_views=3 drops them all;
// loosening to min_views=2 surfaces them and asserts none of the kept tracks
// reference image 1.
TEST(TrackEstablishment, IgnoreMatchPredicateDropsImage) {
  CorrespondenceGraph corr_graph;
  corr_graph.AddImage(1, 5);
  corr_graph.AddImage(2, 5);
  corr_graph.AddImage(3, 5);

  std::vector<std::pair<int, int>> matches;
  std::vector<int> inliers;
  for (int i = 0; i < 5; ++i) {
    matches.emplace_back(i, i);
    inliers.push_back(i);
  }
  AddImagePair(corr_graph, 1, 2, matches, inliers);
  AddImagePair(corr_graph, 1, 3, matches, inliers);
  AddImagePair(corr_graph, 2, 3, matches, inliers);

  const auto keypoints = MakeWellSeparatedKeypoints({1, 2, 3}, 5);
  const auto pair_ids = CollectPairIds(corr_graph);

  const MatchPredicate ignore_image1 =
      [](image_t i1, point2D_t /*p1*/, image_t i2, point2D_t /*p2*/) {
        return i1 == 1 || i2 == 1;
      };

  // With min_views=3, dropping image 1 leaves only length-2 tracks across
  // images 2-3, all filtered out.
  {
    TrackEstablishmentOptions options;  // default min_views=3
    const auto tracks = EstablishTracksFromCorrGraph(
        pair_ids, corr_graph, keypoints, options, ignore_image1);
    EXPECT_TRUE(tracks.empty());
  }

  // With min_views=2 the surviving tracks become visible; verify image 1 is
  // entirely absent.
  {
    TrackEstablishmentOptions options;
    options.min_num_views_per_track = 2;
    const auto tracks = EstablishTracksFromCorrGraph(
        pair_ids, corr_graph, keypoints, options, ignore_image1);
    EXPECT_EQ(tracks.size(), 5);
    for (const auto& [pid, point3D] : tracks) {
      EXPECT_EQ(point3D.track.Length(), 2);
      for (const auto& el : point3D.track.Elements()) {
        EXPECT_NE(el.image_id, 1u);
      }
    }
  }
}

// Empty ``valid_pair_ids`` yields no tracks and no crash.
TEST(TrackEstablishment, EmptyInputReturnsEmpty) {
  CorrespondenceGraph corr_graph;
  std::unordered_map<image_t, std::vector<Eigen::Vector2d>> keypoints;
  TrackEstablishmentOptions options;
  const auto tracks =
      EstablishTracksFromCorrGraph({}, corr_graph, keypoints, options);
  EXPECT_TRUE(tracks.empty());
}

}  // namespace
}  // namespace colmap
