#include "colmap/glomap/view_graph.h"

#include <algorithm>
#include <queue>

namespace colmap::glomap {

void ViewGraph::RemoveInvalidPair(image_pair_t pair_id) {
  ImagePair& pair = image_pairs.at(pair_id);
  pair.is_valid = false;
}

int ViewGraph::KeepLargestConnectedComponents(
    std::unordered_map<image_t, Image>& images) {
  EstablishAdjacencyList();

  const int num_comp = FindConnectedComponent();

  int max_idx = -1;
  int max_img = 0;
  for (int comp = 0; comp < num_comp; ++comp) {
    if (static_cast<int>(connected_components[comp].size()) > max_img) {
      max_img = static_cast<int>(connected_components[comp].size());
      max_idx = comp;
    }
  }

  if (max_img == 0) return 0;

  const std::unordered_set<image_t> largest_component =
      connected_components[max_idx];

  for (auto& [image_id, image] : images) image.is_registered = false;
  for (auto image_id : largest_component) images[image_id].is_registered = true;

  num_pairs = 0;
  for (auto& [pair_id, image_pair] : image_pairs) {
    if (!images[image_pair.image_id1].is_registered ||
        !images[image_pair.image_id2].is_registered) {
      image_pair.is_valid = false;
    }
    if (image_pair.is_valid) ++num_pairs;
  }

  num_images = largest_component.size();
  return max_img;
}

int ViewGraph::FindConnectedComponent() {
  connected_components.clear();
  std::unordered_map<image_t, bool> visited;
  for (auto& [image_id, neighbors] : adjacency_list) {
    visited[image_id] = false;
  }

  for (auto& [image_id, neighbors] : adjacency_list) {
    if (!visited[image_id]) {
      std::unordered_set<image_t> component;
      BFS(image_id, visited, component);
      connected_components.push_back(component);
    }
  }

  return static_cast<int>(connected_components.size());
}

int ViewGraph::MarkConnectedComponents(
    std::unordered_map<image_t, Image>& images, int min_num_img) {
  EstablishAdjacencyList();

  const int num_comp = FindConnectedComponent();

  std::vector<std::pair<int, int>> cluster_num_img(num_comp);
  for (int comp = 0; comp < num_comp; ++comp) {
    cluster_num_img[comp] =
        std::make_pair(static_cast<int>(connected_components[comp].size()),
                       comp);
  }
  std::sort(cluster_num_img.begin(), cluster_num_img.end(),
            std::greater<>());

  for (auto& [image_id, image] : images) image.cluster_id = -1;

  int comp = 0;
  for (; comp < num_comp; ++comp) {
    if (cluster_num_img[comp].first < min_num_img) break;
    for (auto image_id :
         connected_components[cluster_num_img[comp].second]) {
      images[image_id].cluster_id = comp;
    }
  }

  return comp;
}

void ViewGraph::BFS(image_t root,
                    std::unordered_map<image_t, bool>& visited,
                    std::unordered_set<image_t>& component) {
  std::queue<image_t> q;
  q.push(root);
  visited[root] = true;
  component.insert(root);

  while (!q.empty()) {
    const image_t curr = q.front();
    q.pop();

    for (image_t neighbor : adjacency_list[curr]) {
      if (!visited[neighbor]) {
        q.push(neighbor);
        visited[neighbor] = true;
        component.insert(neighbor);
      }
    }
  }
}

void ViewGraph::EstablishAdjacencyList() {
  adjacency_list.clear();
  for (auto& [pair_id, image_pair] : image_pairs) {
    if (image_pair.is_valid) {
      adjacency_list[image_pair.image_id1].insert(image_pair.image_id2);
      adjacency_list[image_pair.image_id2].insert(image_pair.image_id1);
    }
  }
}

}  // namespace colmap::glomap
