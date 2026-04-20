
#pragma once

#include "colmap/estimators/glomap/track_establishment_options.h"

#include "colmap/glomap/math/union_find.h"
#include "colmap/glomap/camera.h"
#include "colmap/glomap/image.h"
#include "colmap/glomap/image_pair.h"
#include "colmap/glomap/track.h"
#include "colmap/glomap/view_graph.h"

#include <unordered_set>

namespace colmap::glomap {

// TrackEstablishmentOptions is defined in colmap/estimators/glomap/track_establishment_options.h (§07).

class TrackEngine {
 public:
  TrackEngine(const ViewGraph& view_graph,
              const std::unordered_map<image_t, Image>& images,
              const TrackEstablishmentOptions& options)
      : options_(options), view_graph_(view_graph), images_(images) {}

  // Set which image IDs to process (empty means process all)
  void SetImageIdsToProcess(const std::unordered_set<image_t>& image_ids) {
    image_ids_to_process_ = image_ids;
  }

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

  const ViewGraph& view_graph_;
  const std::unordered_map<image_t, Image>& images_;

  // Set of image IDs to process (empty means process all)
  std::unordered_set<image_t> image_ids_to_process_;

  // Internal structure used for concatenating tracks
  UnionFind<image_pair_t> uf_;
};

}  // namespace colmap::glomap
