#pragma once

#include "colmap/geometry/rigid3.h"
#include "colmap/scene/correspondence_graph.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/scene/two_view_geometry.h"
#include "colmap/util/types.h"

#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Eigen/Core>

namespace colmap {

class PoseGraph {
 public:
  // Minimal relative pose data for pose graph edge.
  struct Edge {
    Edge() = default;

    explicit Edge(const Rigid3d& cam2_from_cam1)
        : cam2_from_cam1(cam2_from_cam1) {}

    // Relative pose from image 1 to image 2.
    Rigid3d cam2_from_cam1;

    // Number of two-view matches used to compute the relative pose.
    int num_matches = 0;

    // Whether this edge is valid for reconstruction.
    bool valid = true;

    // MDRP-derived relative depth scale.
    double rel_depth_scale = 1.0;

    // Translation covariance matrix.
    Eigen::Matrix3d cov_t = Eigen::Matrix3d::Zero();

    // Feature correspondences: matches[i] = {feat_idx_in_image1, feat_idx_in_image2}.
    // Populated for LC edges by the caller before ProcessLoopClosurePairs.
    std::vector<Eigen::Vector2i> matches;

    // Per-match loop-closure flag (parallel to matches).
    std::vector<bool> are_lc;

    // Whole-edge loop-closure flag.
    bool is_LC = false;

    // Edge weight for rotation averaging.
    double weight = 1.0;

    // Two-view geometry matrices (populated by DecomposeRelPose).
    std::optional<Eigen::Matrix3d> E;
    std::optional<Eigen::Matrix3d> F;
    std::optional<Eigen::Matrix3d> H;

    // TwoViewGeometry::ConfigurationType value; 0 = UNDEFINED.
    int config = 0;

    // Total feature correspondences (used for FilterInlierRatio).
    // num_matches holds the inlier count; total_matches holds the full count.
    int total_matches = 0;

    // Image IDs for the two endpoints (set by AddEdge; derive from pair_id key
    // on read). Stored here so Python callers can access them without the key.
    image_t image_id1 = kInvalidImageId;
    image_t image_id2 = kInvalidImageId;

    // Inlier indices into matches (parallel to matches[i]).
    std::vector<int> inliers;

    // Invert the geometry to match swapped image order.
    void Invert() { cam2_from_cam1 = Inverse(cam2_from_cam1); }
  };

  // Input record for AssignMdrpResults.
  struct MDRPResult {
    bool is_valid = false;
    Rigid3d cam2_from_cam1;
    double weight = 1.0;
    double rel_depth_scale = 1.0;
    std::vector<int> inliers;
    Eigen::Matrix3d cov_t = Eigen::Matrix3d::Zero();
  };

  // Output record for ExtractValidPairData.
  struct ValidPairData {
    std::vector<image_pair_t> pair_ids;
    std::vector<image_t> image_ids1;
    std::vector<image_t> image_ids2;
    std::vector<int> inlier_counts;
    std::vector<std::vector<bool>> are_lc;
  };

  PoseGraph() = default;
  ~PoseGraph() = default;

  // Edge accessors.
  inline std::unordered_map<image_pair_t, Edge>& Edges();
  inline const std::unordered_map<image_pair_t, Edge>& Edges() const;
  inline size_t NumEdges() const;
  inline bool Empty() const;
  inline void Clear();

  // Load edges from the correspondence graph.
  void Load(const CorrespondenceGraph& corr_graph);

  // Edge operations.
  inline Edge& AddEdge(image_t image_id1, image_t image_id2, Edge edge);
  inline bool HasEdge(image_t image_id1, image_t image_id2) const;
  // Returns a reference to the edge and whether the IDs were swapped.
  inline std::pair<Edge&, bool> EdgeRef(image_t image_id1, image_t image_id2);
  inline std::pair<const Edge&, bool> EdgeRef(image_t image_id1,
                                              image_t image_id2) const;
  inline Edge GetEdge(image_t image_id1, image_t image_id2) const;
  inline bool DeleteEdge(image_t image_id1, image_t image_id2);
  inline void UpdateEdge(image_t image_id1, image_t image_id2, Edge edge);

  // Validity operations.
  inline bool IsValid(image_pair_t pair_id) const;
  inline void SetValidEdge(image_pair_t pair_id);
  inline void SetInvalidEdge(image_pair_t pair_id);

  // Returns a filter view over valid edges only.
  auto ValidEdges() const {
    return filter_view(
        [](const std::pair<const image_pair_t, Edge>& kv) {
          return kv.second.valid;
        },
        edges_.begin(),
        edges_.end());
  }

  // Compute the largest connected component of frames.
  // If filter_unregistered is true, only considers frames with HasPose().
  // Returns the set of frame_ids in the largest connected component.
  std::unordered_set<frame_t> ComputeLargestConnectedFrameComponent(
      const Reconstruction& reconstruction,
      bool filter_unregistered = true) const;

  // Mark image pairs as invalid if either image is not in the active set.
  void InvalidatePairsOutsideActiveImageIds(
      const std::unordered_set<image_t>& active_image_ids);

  // Mark connected clusters of images, where the cluster_id is sorted by the
  // the number of images. Populates `cluster_ids` output parameter.
  int MarkConnectedComponents(const Reconstruction& reconstruction,
                              std::unordered_map<frame_t, int>& cluster_ids,
                              int min_num_images = -1) const;

  // Batch-assign MDRP results to matching edges.
  // Returns (num_valid_assigned, num_invalid_assigned).
  std::pair<int, int> AssignMdrpResults(
      const std::unordered_map<image_pair_t, MDRPResult>& results,
      double metric_scale_stddev);

  // Return per-edge data arrays for all currently valid edges.
  // All output arrays are aligned by index.
  ValidPairData ExtractValidPairData() const;

  // Invalidate edges with inlier count (num_matches) strictly below threshold.
  void FilterInlierNum(int min_inliers);

  // Invalidate edges with inlier_ratio = num_matches / total_matches strictly
  // below min_ratio. total_matches must be pre-populated on each edge.
  void FilterInlierRatio(double min_ratio);

  // Keep the top-k connected components (by image count) of valid edges.
  // Components smaller than min_component_size are dropped even if in top-k.
  // Invalidates edges belonging to dropped components.
  void KeepLargestConnectedComponents(size_t k, size_t min_component_size);

 private:
  // Map from pair ID to edge data. The pair ID is computed from the
  // two image IDs using ImagePairToPairId, with the smaller ID first.
  std::unordered_map<image_pair_t, Edge> edges_;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

std::unordered_map<image_pair_t, PoseGraph::Edge>& PoseGraph::Edges() {
  return edges_;
}

const std::unordered_map<image_pair_t, PoseGraph::Edge>& PoseGraph::Edges()
    const {
  return edges_;
}

size_t PoseGraph::NumEdges() const { return edges_.size(); }

bool PoseGraph::Empty() const { return edges_.empty(); }

void PoseGraph::Clear() { edges_.clear(); }

PoseGraph::Edge& PoseGraph::AddEdge(image_t image_id1,
                                    image_t image_id2,
                                    PoseGraph::Edge edge) {
  const bool swapped = ShouldSwapImagePair(image_id1, image_id2);
  if (swapped) {
    edge.Invert();
  }
  edge.image_id1 = swapped ? image_id2 : image_id1;
  edge.image_id2 = swapped ? image_id1 : image_id2;
  const image_pair_t pair_id = ImagePairToPairId(image_id1, image_id2);
  auto [it, inserted] = edges_.emplace(pair_id, std::move(edge));
  if (!inserted) {
    throw std::runtime_error(
        "Image pair already exists: " + std::to_string(image_id1) + ", " +
        std::to_string(image_id2));
  }
  return it->second;
}

bool PoseGraph::HasEdge(image_t image_id1, image_t image_id2) const {
  const image_pair_t pair_id = ImagePairToPairId(image_id1, image_id2);
  return edges_.find(pair_id) != edges_.end();
}

std::pair<PoseGraph::Edge&, bool> PoseGraph::EdgeRef(image_t image_id1,
                                                     image_t image_id2) {
  const bool swapped = ShouldSwapImagePair(image_id1, image_id2);
  const image_pair_t pair_id = ImagePairToPairId(image_id1, image_id2);
  return {edges_.at(pair_id), swapped};
}

std::pair<const PoseGraph::Edge&, bool> PoseGraph::EdgeRef(
    image_t image_id1, image_t image_id2) const {
  const bool swapped = ShouldSwapImagePair(image_id1, image_id2);
  const image_pair_t pair_id = ImagePairToPairId(image_id1, image_id2);
  return {edges_.at(pair_id), swapped};
}

bool PoseGraph::DeleteEdge(image_t image_id1, image_t image_id2) {
  const image_pair_t pair_id = ImagePairToPairId(image_id1, image_id2);
  return edges_.erase(pair_id) > 0;
}

PoseGraph::Edge PoseGraph::GetEdge(image_t image_id1, image_t image_id2) const {
  const bool swapped = ShouldSwapImagePair(image_id1, image_id2);
  const image_pair_t pair_id = ImagePairToPairId(image_id1, image_id2);
  PoseGraph::Edge result = edges_.at(pair_id);
  if (swapped) {
    result.Invert();
  }
  return result;
}

void PoseGraph::UpdateEdge(image_t image_id1,
                           image_t image_id2,
                           PoseGraph::Edge edge) {
  if (ShouldSwapImagePair(image_id1, image_id2)) {
    edge.Invert();
  }
  const image_pair_t pair_id = ImagePairToPairId(image_id1, image_id2);
  auto it = edges_.find(pair_id);
  if (it == edges_.end()) {
    throw std::runtime_error(
        "Image pair does not exist: " + std::to_string(image_id1) + ", " +
        std::to_string(image_id2));
  }
  it->second = std::move(edge);
}

bool PoseGraph::IsValid(image_pair_t pair_id) const {
  auto it = edges_.find(pair_id);
  return it != edges_.end() && it->second.valid;
}

void PoseGraph::SetValidEdge(image_pair_t pair_id) {
  auto it = edges_.find(pair_id);
  THROW_CHECK(it != edges_.end()) << "Edge does not exist";
  it->second.valid = true;
}

void PoseGraph::SetInvalidEdge(image_pair_t pair_id) {
  auto it = edges_.find(pair_id);
  THROW_CHECK(it != edges_.end()) << "Edge does not exist";
  it->second.valid = false;
}

}  // namespace colmap
