#pragma once

#include "colmap/glomap/image.h"
#include "colmap/glomap/view_graph.h"

#include <vector>

namespace colmap::glomap {

struct RelPoseFilter {
  static std::vector<image_pair_t> FilterRotations(
      ViewGraph& view_graph,
      const std::unordered_map<image_t, Image>& images,
      double max_angle = 5.0);

  static void FilterInlierNum(ViewGraph& view_graph, int min_inlier_num = 30);

  static void FilterInlierRatio(ViewGraph& view_graph,
                                double min_inlier_ratio = 0.25);
};

}  // namespace colmap::glomap
