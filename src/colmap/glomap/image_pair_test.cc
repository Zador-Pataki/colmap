#include "colmap/glomap/image_pair.h"

#include <sstream>

#include <gtest/gtest.h>

namespace colmap::glomap {
namespace {

TEST(ImagePair, DefaultConstructor_UninitializedIds) {
  ImagePair p;
  EXPECT_EQ(p.pair_id, static_cast<image_pair_t>(-1));
  EXPECT_EQ(p.image_id1, static_cast<image_t>(-1));
  EXPECT_EQ(p.image_id2, static_cast<image_t>(-1));
}

TEST(ImagePair, NamedConstructor_SetsConstIdFields) {
  ImagePair p(3, 5);
  EXPECT_EQ(p.image_id1, 3u);
  EXPECT_EQ(p.image_id2, 5u);
  EXPECT_EQ(p.pair_id, ImagePair::ImagePairToPairId(3, 5));
}

TEST(ImagePair, NonConstFieldsWritable) {
  ImagePair p(1, 2);
  p.is_valid = false;
  p.weight = 0.5;
  EXPECT_FALSE(p.is_valid);
  EXPECT_EQ(p.weight, 0.5);
}

TEST(ImagePair, PairTypeEnum_DefaultsToAdjacent) {
  ImagePair p(1, 2);
  EXPECT_EQ(p.type, PairType::ADJACENT);
}

TEST(ImagePair, PairTypeEnum_HasThreeValues) {
  EXPECT_EQ(static_cast<int>(PairType::ADJACENT), 0);
  EXPECT_EQ(static_cast<int>(PairType::NONADJACENT), 1);
  EXPECT_EQ(static_cast<int>(PairType::LOOP_CLOSURE), 2);
}

TEST(ImagePair, IsLCField_DefaultsFalse) {
  ImagePair p(1, 2);
  EXPECT_FALSE(p.is_LC);
}

TEST(ImagePair, ImagePairToPairId_Symmetric) {
  EXPECT_EQ(ImagePair::ImagePairToPairId(2, 5),
            ImagePair::ImagePairToPairId(5, 2));
}

TEST(ImagePair, PairIdToImagePair_InvertsImagePairToPairId) {
  const image_pair_t id = ImagePair::ImagePairToPairId(7, 13);
  image_t a = 0;
  image_t b = 0;
  ImagePair::PairIdToImagePair(id, a, b);
  EXPECT_EQ(a, 7u);
  EXPECT_EQ(b, 13u);
}

TEST(ImagePair, OstreamOperator_EmitsExpectedPrefix) {
  ImagePair p(1, 2);
  std::ostringstream os;
  os << p;
  EXPECT_NE(os.str().find("ImagePair(image_id1=1, image_id2=2"), std::string::npos);
}

}  // namespace
}  // namespace colmap::glomap
