#include "colmap/glomap/image_pair.h"

#include <ostream>

namespace colmap::glomap {

image_pair_t ImagePair::ImagePairToPairId(const image_t image_id1,
                                          const image_t image_id2) {
  if (image_id1 > image_id2) {
    return static_cast<image_pair_t>(kMaxNumImages) * image_id2 + image_id1;
  }
  return static_cast<image_pair_t>(kMaxNumImages) * image_id1 + image_id2;
}

void ImagePair::PairIdToImagePair(const image_pair_t pair_id,
                                  image_t& image_id1,
                                  image_t& image_id2) {
  image_id1 = static_cast<image_t>(pair_id % kMaxNumImages);
  image_id2 = static_cast<image_t>((pair_id - image_id1) / kMaxNumImages);
}

std::ostream& operator<<(std::ostream& stream, const ImagePair& tform) {
  stream << "ImagePair(image_id1=" << tform.image_id1
         << ", image_id2=" << tform.image_id2 << ", pair_id=" << tform.pair_id
         << ", is_valid=" << tform.is_valid << ", weight=" << tform.weight
         << ", rel_depth_scale=" << tform.rel_depth_scale
         << ", config=" << tform.config
         << ", cam2_from_cam1=" << tform.cam2_from_cam1
         << ", num_matches=" << tform.matches.rows()
         << ", num_inliers=" << tform.inliers.size() << ")";
  return stream;
}

}  // namespace colmap::glomap
