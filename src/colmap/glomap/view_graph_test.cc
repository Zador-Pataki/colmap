#include "colmap/glomap/view_graph.h"

#include <gtest/gtest.h>

namespace colmap::glomap {
namespace {

ImagePair MakePair(image_t a, image_t b) { return ImagePair(a, b); }

TEST(ViewGraph, DefaultConstructor_ProducesEmpty) {
  ViewGraph g;
  EXPECT_TRUE(g.image_pairs.empty());
  EXPECT_EQ(g.num_images, 0u);
  EXPECT_EQ(g.num_pairs, 0u);
  EXPECT_TRUE(g.GetAdjacencyList().empty());
}

TEST(ViewGraph, KeepLargestConnectedComponents_ReturnsLargest) {
  ViewGraph g;
  // Component A: {1,2,3}. Component B: {10,11}.
  for (auto [a, b] : std::vector<std::pair<image_t, image_t>>{
           {1, 2}, {2, 3}, {10, 11}}) {
    auto p = MakePair(a, b);
    g.image_pairs.insert({p.pair_id, p});
  }
  std::unordered_map<image_t, Image> imgs;
  for (image_t id : {1u, 2u, 3u, 10u, 11u}) {
    imgs.insert({id, Image(id, 0, std::to_string(id))});
  }
  EXPECT_EQ(g.KeepLargestConnectedComponents(imgs), 3);
  EXPECT_TRUE(imgs.at(1).is_registered);
  EXPECT_TRUE(imgs.at(2).is_registered);
  EXPECT_TRUE(imgs.at(3).is_registered);
  EXPECT_FALSE(imgs.at(10).is_registered);
  EXPECT_FALSE(imgs.at(11).is_registered);
}

TEST(ViewGraph, MarkConnectedComponents_AssignsClusterId) {
  ViewGraph g;
  for (auto [a, b] : std::vector<std::pair<image_t, image_t>>{
           {1, 2}, {2, 3}, {10, 11}}) {
    auto p = MakePair(a, b);
    g.image_pairs.insert({p.pair_id, p});
  }
  std::unordered_map<image_t, Image> imgs;
  for (image_t id : {1u, 2u, 3u, 10u, 11u}) {
    imgs.insert({id, Image(id, 0, std::to_string(id))});
  }
  EXPECT_EQ(g.MarkConnectedComponents(imgs, /*min_num_img=*/2), 2);
  // First (largest) cluster = 0 for imgs 1,2,3. Second cluster = 1 for 10,11.
  EXPECT_EQ(imgs.at(1).cluster_id, 0);
  EXPECT_EQ(imgs.at(10).cluster_id, 1);
}

TEST(ViewGraph, EstablishAdjacencyList_MatchesExpected) {
  ViewGraph g;
  auto p = MakePair(1, 2);
  g.image_pairs.insert({p.pair_id, p});
  g.EstablishAdjacencyList();
  const auto& adj = g.GetAdjacencyList();
  EXPECT_EQ(adj.at(1).count(2), 1u);
  EXPECT_EQ(adj.at(2).count(1), 1u);
}

TEST(ViewGraph, RemoveInvalidPair_FlipsIsValid) {
  ViewGraph g;
  auto p = MakePair(5, 6);
  g.image_pairs.insert({p.pair_id, p});
  g.RemoveInvalidPair(p.pair_id);
  EXPECT_FALSE(g.image_pairs.at(p.pair_id).is_valid);
}

TEST(ViewGraph, LoopClosurePairFlag_RoundTrip) {
  ImagePair p = MakePair(2, 7);
  p.type = PairType::LOOP_CLOSURE;
  p.is_LC = true;
  EXPECT_EQ(p.type, PairType::LOOP_CLOSURE);
  EXPECT_TRUE(p.is_LC);
}

}  // namespace
}  // namespace colmap::glomap
