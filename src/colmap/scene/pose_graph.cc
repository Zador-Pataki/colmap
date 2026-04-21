#include "colmap/scene/pose_graph.h"

#include "colmap/math/connected_components.h"

#include <algorithm>
#include <queue>

namespace colmap {

void PoseGraph::Load(const CorrespondenceGraph& corr_graph) {
  for (const auto& [pair_id, num_matches] :
       corr_graph.NumMatchesBetweenAllImages()) {
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    const TwoViewGeometry two_view_geometry = corr_graph.ExtractTwoViewGeometry(
        image_id1, image_id2, /*extract_inlier_matches=*/false);
    if (two_view_geometry.cam2_from_cam1.has_value()) {
      Edge edge;
      edge.cam2_from_cam1 = *two_view_geometry.cam2_from_cam1;
      edge.num_matches = num_matches;
      AddEdge(image_id1, image_id2, std::move(edge));
    }
  }

  LOG(INFO) << "Loaded " << edges_.size() << " edges into pose graph";
}

std::unordered_set<frame_t> PoseGraph::ComputeLargestConnectedFrameComponent(
    const Reconstruction& reconstruction, bool filter_unregistered) const {
  std::unordered_set<frame_t> nodes;
  std::vector<std::pair<frame_t, frame_t>> graph_edges;

  for (const auto& [pair_id, edge] : ValidEdges()) {
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    const frame_t frame_id1 = reconstruction.Image(image_id1).FrameId();
    const frame_t frame_id2 = reconstruction.Image(image_id2).FrameId();

    // If filter_unregistered, skip pairs where either frame has no pose.
    if (filter_unregistered) {
      if (!reconstruction.Frame(frame_id1).HasPose() ||
          !reconstruction.Frame(frame_id2).HasPose()) {
        continue;
      }
    }

    nodes.insert(frame_id1);
    nodes.insert(frame_id2);
    graph_edges.emplace_back(frame_id1, frame_id2);
  }

  if (nodes.empty()) {
    return {};
  }

  const std::vector<frame_t> largest_component_vec =
      FindLargestConnectedComponent(nodes, graph_edges);
  return std::unordered_set<frame_t>(largest_component_vec.begin(),
                                     largest_component_vec.end());
}

void PoseGraph::InvalidatePairsOutsideActiveImageIds(
    const std::unordered_set<image_t>& active_image_ids) {
  for (const auto& [pair_id, edge] : edges_) {
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    if (!active_image_ids.count(image_id1) ||
        !active_image_ids.count(image_id2)) {
      SetInvalidEdge(pair_id);
    }
  }
}

int PoseGraph::MarkConnectedComponents(
    const Reconstruction& reconstruction,
    std::unordered_map<frame_t, int>& cluster_ids,
    int min_num_images) const {
  std::unordered_set<frame_t> nodes;
  std::vector<std::pair<frame_t, frame_t>> graph_edges;
  for (const auto& [pair_id, edge] : ValidEdges()) {
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    const frame_t frame_id1 = reconstruction.Image(image_id1).FrameId();
    const frame_t frame_id2 = reconstruction.Image(image_id2).FrameId();
    nodes.insert(frame_id1);
    nodes.insert(frame_id2);
    graph_edges.emplace_back(frame_id1, frame_id2);
  }

  const std::vector<std::vector<frame_t>> connected_components =
      FindConnectedComponents(nodes, graph_edges);
  const int num_comp = static_cast<int>(connected_components.size());

  std::vector<std::pair<int, int>> comp_num_images(num_comp);
  for (int comp = 0; comp < num_comp; comp++) {
    comp_num_images[comp] =
        std::make_pair(connected_components[comp].size(), comp);
  }
  std::sort(comp_num_images.begin(), comp_num_images.end(), std::greater<>());

  // Clear and populate cluster_ids output parameter
  cluster_ids.clear();
  for (const auto& [frame_id, frame] : reconstruction.Frames()) {
    cluster_ids[frame_id] = -1;
  }

  int comp = 0;
  for (; comp < num_comp; comp++) {
    if (comp_num_images[comp].first < min_num_images) break;
    for (auto frame_id : connected_components[comp_num_images[comp].second]) {
      cluster_ids[frame_id] = comp;
    }
  }

  return comp;
}

std::pair<int, int> PoseGraph::AssignMdrpResults(
    const std::unordered_map<image_pair_t, MDRPResult>& results,
    double metric_scale_stddev) {
  int num_valid = 0;
  int num_invalid = 0;
  const double scale_sq = metric_scale_stddev * metric_scale_stddev;

  for (const auto& [pair_id, res] : results) {
    auto it = edges_.find(pair_id);
    if (it == edges_.end()) continue;
    Edge& edge = it->second;

    if (!res.is_valid) {
      edge.valid = false;
      ++num_invalid;
      continue;
    }

    edge.valid = true;
    edge.cam2_from_cam1 = res.cam2_from_cam1;
    edge.weight = res.weight;
    edge.rel_depth_scale = res.rel_depth_scale;
    edge.num_matches = static_cast<int>(res.inliers.size());

    // Augment cov_t with scale uncertainty: C += t*t^T * sigma_s^2.
    const Eigen::Vector3d t = edge.cam2_from_cam1.translation();
    edge.cov_t = res.cov_t + t * t.transpose() * scale_sq;
    ++num_valid;
  }

  return {num_valid, num_invalid};
}

PoseGraph::ValidPairData PoseGraph::ExtractValidPairData() const {
  ValidPairData data;
  for (const auto& [pair_id, edge] : edges_) {
    if (!edge.valid) continue;
    const auto [id1, id2] = PairIdToImagePair(pair_id);
    data.pair_ids.push_back(pair_id);
    data.image_ids1.push_back(id1);
    data.image_ids2.push_back(id2);
    data.inlier_counts.push_back(edge.num_matches);
    data.are_lc.push_back(edge.are_lc);
  }
  return data;
}

void PoseGraph::FilterInlierNum(int min_inliers) {
  for (auto& [pair_id, edge] : edges_) {
    if (!edge.valid) continue;
    if (edge.num_matches < min_inliers) {
      edge.valid = false;
    }
  }
}

void PoseGraph::FilterInlierRatio(double min_ratio) {
  for (auto& [pair_id, edge] : edges_) {
    if (!edge.valid) continue;
    if (edge.total_matches <= 0) continue;
    const double ratio =
        static_cast<double>(edge.num_matches) / edge.total_matches;
    if (ratio < min_ratio) {
      edge.valid = false;
    }
  }
}

void PoseGraph::KeepLargestConnectedComponents(size_t k,
                                               size_t min_component_size) {
  // Build adjacency list over valid edges (image-level).
  std::unordered_map<image_t, std::unordered_set<image_t>> adj;
  for (const auto& [pair_id, edge] : edges_) {
    if (!edge.valid) continue;
    const auto [id1, id2] = PairIdToImagePair(pair_id);
    adj[id1].insert(id2);
    adj[id2].insert(id1);
  }

  if (adj.empty()) return;

  // BFS to find all connected components.
  std::unordered_map<image_t, bool> visited;
  for (const auto& [node, _] : adj) visited[node] = false;

  std::vector<std::unordered_set<image_t>> components;
  for (const auto& [start, _] : adj) {
    if (visited[start]) continue;
    std::unordered_set<image_t> comp;
    std::queue<image_t> q;
    q.push(start);
    visited[start] = true;
    comp.insert(start);
    while (!q.empty()) {
      const image_t cur = q.front();
      q.pop();
      for (const image_t nb : adj.at(cur)) {
        if (!visited[nb]) {
          visited[nb] = true;
          comp.insert(nb);
          q.push(nb);
        }
      }
    }
    components.push_back(std::move(comp));
  }

  // Sort components descending by size.
  std::sort(components.begin(), components.end(),
            [](const auto& a, const auto& b) { return a.size() > b.size(); });

  // Build the set of images to keep.
  std::unordered_set<image_t> keep;
  for (size_t i = 0; i < std::min(k, components.size()); ++i) {
    if (components[i].size() < min_component_size) break;
    keep.insert(components[i].begin(), components[i].end());
  }

  // Invalidate edges whose either endpoint was dropped.
  for (auto& [pair_id, edge] : edges_) {
    if (!edge.valid) continue;
    const auto [id1, id2] = PairIdToImagePair(pair_id);
    if (!keep.count(id1) || !keep.count(id2)) {
      edge.valid = false;
    }
  }
}

}  // namespace colmap
