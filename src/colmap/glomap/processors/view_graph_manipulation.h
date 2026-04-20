#pragma once

#include "colmap/glomap/camera.h"
#include "colmap/glomap/image.h"
#include "colmap/glomap/view_graph.h"

#include <unordered_map>

namespace colmap::glomap {

struct ViewGraphManipulater {
  enum StrongClusterCriteria {
    INLIER_NUM,
    WEIGHT,
  };

  static image_pair_t SparsifyGraph(ViewGraph& view_graph,
                                    std::unordered_map<image_t, Image>& images,
                                    int expected_degree = 50);

  static image_t EstablishStrongClusters(
      ViewGraph& view_graph,
      std::unordered_map<image_t, Image>& images,
      StrongClusterCriteria criteria = INLIER_NUM,
      double min_thres = 100,
      int min_num_images = 2);

  static void UpdateImagePairsConfig(
      ViewGraph& view_graph,
      const std::unordered_map<camera_t, Camera>& cameras,
      const std::unordered_map<image_t, Image>& images);

  static void DecomposeRelPose(ViewGraph& view_graph,
                               std::unordered_map<camera_t, Camera>& cameras,
                               std::unordered_map<image_t, Image>& images);
};

}  // namespace colmap::glomap
