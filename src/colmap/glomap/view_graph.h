#pragma once

#include "colmap/glomap/image.h"
#include "colmap/glomap/image_pair.h"
#include "colmap/glomap/types.h"

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace colmap::glomap {

class ViewGraph {
 public:
  void RemoveInvalidPair(image_pair_t pair_id);

  // Mark images not connected to any other image as not registered. Returns
  // the number of images in the largest connected component.
  int KeepLargestConnectedComponents(
      std::unordered_map<image_t, Image>& images);

  // Mark each image's cluster_id (sorted by component size). min_num_img < 0
  // means "no cutoff".
  int MarkConnectedComponents(std::unordered_map<image_t, Image>& images,
                              int min_num_img = -1);

  void EstablishAdjacencyList();

  const std::unordered_map<image_t, std::unordered_set<image_t>>&
  GetAdjacencyList() const {
    return adjacency_list;
  }

  // Data.
  std::unordered_map<image_pair_t, ImagePair> image_pairs;

  image_t num_images = 0;
  image_pair_t num_pairs = 0;

 private:
  int FindConnectedComponent();

  void BFS(image_t root,
           std::unordered_map<image_t, bool>& visited,
           std::unordered_set<image_t>& component);

  std::unordered_map<image_t, std::unordered_set<image_t>> adjacency_list;
  std::vector<std::unordered_set<image_t>> connected_components;
};

}  // namespace colmap::glomap
