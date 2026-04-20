#pragma once

#include "colmap/estimators/glomap/inlier_threshold_options.h"
#include "colmap/glomap/camera.h"
#include "colmap/glomap/image.h"
#include "colmap/glomap/image_pair.h"
#include "colmap/glomap/math/rigid3d.h"
#include "colmap/glomap/view_graph.h"

namespace colmap::glomap {

enum class GeometryConfig {
  UNKNOWN,
  ESSENTIAL,
  FUNDAMENTAL,
  HOMOGRAPHY,
};

struct PerPairStats {
  GeometryConfig type = GeometryConfig::UNKNOWN;
  size_t total_matches = 0;
  size_t num_inliers = 0;
  size_t failed_epipolar_E = 0;
  size_t failed_cheirality_E = 0;
  size_t failed_angle_check_E = 0;
  size_t failed_epipole_check_E = 0;
  size_t failed_epipolar_F = 0;
  size_t failed_cheirality_F = 0;
  size_t failed_epipolar_H = 0;
};

class ImagePairInliers {
 public:
  ImagePairInliers(
      ImagePair& image_pair,
      const std::unordered_map<image_t, Image>& images,
      const InlierThresholdOptions& options,
      const std::unordered_map<camera_t, Camera>* cameras = nullptr)
      : image_pair(image_pair),
        images(images),
        cameras(cameras),
        options(options) {}

  double ScoreError();
  PerPairStats stats;

 protected:
  double ScoreErrorEssential();
  double ScoreErrorFundamental();
  double ScoreErrorHomography();

  ImagePair& image_pair;
  const std::unordered_map<image_t, Image>& images;
  const std::unordered_map<camera_t, Camera>* cameras;
  const InlierThresholdOptions& options;
};

void ImagePairsInlierCount(ViewGraph& view_graph,
                           const std::unordered_map<camera_t, Camera>& cameras,
                           const std::unordered_map<image_t, Image>& images,
                           const InlierThresholdOptions& options,
                           bool clean_inliers);

}  // namespace colmap::glomap
