
#pragma once

#include "colmap/sfm/union_find_glomap.h"
#include "colmap/scene/camera.h"
#include "colmap/scene/correspondence_graph.h"
#include "colmap/scene/image.h"
#include "colmap/util/types.h"
#include <unordered_map>

#include <unordered_set>

namespace colmap {
namespace glomap_ra {
using ViewGraph = colmap::CorrespondenceGraph;
using ImagePair = colmap::CorrespondenceGraph::ImagePair;
// (glomap::Observation, glomap::Track) are vendored below
// because colmap's Point3D / Track use a different shape; the
// pycolmap binding converts between them.
using feature_t = colmap::point2D_t;
using track_t = uint64_t;
using Observation = std::pair<colmap::image_t, feature_t>;
struct Track {
  track_t track_id = 0;
  Eigen::Vector3d xyz = Eigen::Vector3d::Zero();
  Eigen::Matrix<uint8_t, 3, 1> color = Eigen::Matrix<uint8_t, 3, 1>::Zero();
  bool is_initialized = false;
  std::vector<Observation> observations;
  std::vector<Observation> lc_observations;
};


struct TrackEstablishmentOptions {
  // the max allowed distance for features in the same track in the same image
  double thres_inconsistency = 10.;

  // The minimal number of tracks for each view,
  int min_num_tracks_per_view = -1;

  // The minimal number of tracks for each view pair
  int min_num_view_per_track = 3;

  // The maximal number of tracks for each view pair
  int max_num_view_per_track = 100;

  // The maximal number of tracks
  int max_num_tracks = 10000000;
};

class TrackEngine {
 public:
  TrackEngine(ViewGraph& view_graph,
              const std::unordered_map<image_t, Image>& images,
              const TrackEstablishmentOptions& options)
      : options_(options), view_graph_(view_graph), images_(images) {}

  // Establish tracks from the view graph. Exclude the tracks that are not
  // consistent Return the number of tracks
  size_t EstablishFullTracks(std::unordered_map<track_t, Track>& tracks);

  // Subsample the tracks, and exclude too short / long tracks
  // Return the number of tracks
  size_t FindTracksForProblem(
      const std::unordered_map<track_t, Track>& tracks_full,
      std::unordered_map<track_t, Track>& tracks_selected);

 private:
  // Blindly concatenate tracks if any matches occur
  void BlindConcatenation();

  // Iterate through the collected tracks and record the items for each track
  void TrackCollection(std::unordered_map<track_t, Track>& tracks);

  // Iterate over loop-closure pairs and add cross-track LC observations
  void ProcessLoopClosurePairs(std::unordered_map<track_t, Track>& tracks);

  const TrackEstablishmentOptions& options_;

  ViewGraph& view_graph_;
  const std::unordered_map<image_t, Image>& images_;

  // Internal structure used for concatenating tracks
  UnionFind<image_pair_t> uf_;
};

}  // namespace glomap_ra
}  // namespace colmap
