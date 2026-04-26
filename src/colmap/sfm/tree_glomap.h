
#include <unordered_set>

#include "colmap/scene/camera.h"
#include "colmap/scene/correspondence_graph.h"
#include "colmap/scene/image.h"
#include "colmap/util/types.h"
#include <unordered_map>

namespace colmap {
namespace glomap_ra {
using ViewGraph = colmap::CorrespondenceGraph;
using ImagePair = colmap::CorrespondenceGraph::ImagePair;
enum WeightType { INLIER_NUM, INLIER_RATIO };

// Return the number of nodes in the tree
int BFS(const std::vector<std::vector<int>>& graph,
        int root,
        std::vector<int>& parents,
        std::vector<std::pair<int, int>> banned_edges = {});

image_t MaximumSpanningTree(ViewGraph& view_graph,
                            const std::unordered_map<image_t, Image>& images,
                            std::unordered_map<image_t, image_t>& parents,
                            WeightType type,
                            bool prioritize_tracking = false);
}  // namespace glomap_ra
}  // namespace colmap