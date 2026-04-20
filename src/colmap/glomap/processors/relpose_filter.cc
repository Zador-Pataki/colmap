#include "colmap/glomap/processors/relpose_filter.h"

#include "colmap/glomap/math/rigid3d.h"
#include "colmap/geometry/rigid3.h"

namespace colmap::glomap {

std::vector<image_pair_t> RelPoseFilter::FilterRotations(
    ViewGraph& view_graph,
    const std::unordered_map<image_t, Image>& images,
    double max_angle) {
  int num_invalid = 0;
  std::vector<image_pair_t> invalid_pair_ids;
  for (auto& [pair_id, image_pair] : view_graph.image_pairs) {
    if (!image_pair.is_valid) continue;

    const Image& image1 = images.at(image_pair.image_id1);
    const Image& image2 = images.at(image_pair.image_id2);

    if (!image1.is_registered || !image2.is_registered) {
      continue;
    }

    colmap::Rigid3d pose_calc =
        image2.cam_from_world * colmap::Inverse(image1.cam_from_world);

    double angle = CalcAngle(pose_calc, image_pair.cam2_from_cam1);
    if (angle > max_angle) {
      image_pair.is_valid = false;
      num_invalid++;
      invalid_pair_ids.push_back(pair_id);
    }
  }

  LOG(INFO) << "Filtered " << num_invalid << " relative rotation with angle > "
            << max_angle << " degrees";

  return invalid_pair_ids;
}

void RelPoseFilter::FilterInlierNum(ViewGraph& view_graph, int min_inlier_num) {
  int num_invalid = 0;
  for (auto& [pair_id, image_pair] : view_graph.image_pairs) {
    if (!image_pair.is_valid) continue;
    if (static_cast<int>(image_pair.inliers.size()) < min_inlier_num) {
      image_pair.is_valid = false;
      num_invalid++;
    }
  }

  LOG(INFO) << "Filtered " << num_invalid
            << " relative poses with inlier number < " << min_inlier_num;
}

void RelPoseFilter::FilterInlierRatio(ViewGraph& view_graph,
                                      double min_inlier_ratio) {
  int num_invalid = 0;
  for (auto& [pair_id, image_pair] : view_graph.image_pairs) {
    if (!image_pair.is_valid) continue;

    if (image_pair.inliers.size() / double(image_pair.matches.rows()) <
        min_inlier_ratio) {
      image_pair.is_valid = false;
      num_invalid++;
    }
  }

  LOG(INFO) << "Filtered " << num_invalid
            << " relative poses with inlier ratio < " << min_inlier_ratio;
}

}  // namespace colmap::glomap
