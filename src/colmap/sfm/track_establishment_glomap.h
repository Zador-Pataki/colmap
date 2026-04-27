
#pragma once

#include "colmap/math/union_find.h"
#include "colmap/scene/camera.h"
#include "colmap/scene/correspondence_graph.h"
#include "colmap/scene/image.h"
#include "colmap/scene/point3d.h"
#include "colmap/util/types.h"
#include <unordered_map>

#include <unordered_set>

// The ``colmap::glomap_ra`` namespace is kept post-collapse for binding-path
// stability — pycolmap exposes these algorithms under a ``glomap_ra``
// submodule (e.g. ``pycolmap.glomap_ra.filter_tracks_by_angle``) and the
// videosfm callers depend on those import paths. The C++ types
// underneath now reuse native colmap (``Point3D``/``Track``/``UnionFind``).
namespace colmap {
namespace glomap_ra {
using ViewGraph = colmap::CorrespondenceGraph;
using ImagePair = colmap::CorrespondenceGraph::ImagePair;
// Track shape collapsed onto colmap::Point3D — its ``track`` member
// (colmap::Track) holds the regular observations as TrackElements and
// the ``lc_elements`` parallel vector for loop-closure observations.
// xyz / color / is_initialized live directly on Point3D. The
// ``track_t`` typedef is just colmap::point3D_t (the natural map key).
using feature_t = colmap::point2D_t;
using track_t = colmap::point3D_t;


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
  size_t EstablishFullTracks(std::unordered_map<track_t, Point3D>& tracks);

  // Subsample the tracks, and exclude too short / long tracks
  // Return the number of tracks
  size_t FindTracksForProblem(
      const std::unordered_map<track_t, Point3D>& tracks_full,
      std::unordered_map<track_t, Point3D>& tracks_selected);

 private:
  // Blindly concatenate tracks if any matches occur
  void BlindConcatenation();

  // Iterate through the collected tracks and record the items for each track
  void TrackCollection(std::unordered_map<track_t, Point3D>& tracks);

  // Iterate over loop-closure pairs and add cross-track LC observations
  void ProcessLoopClosurePairs(std::unordered_map<track_t, Point3D>& tracks);

  const TrackEstablishmentOptions& options_;

  ViewGraph& view_graph_;
  const std::unordered_map<image_t, Image>& images_;

  // Internal structure used for concatenating tracks
  UnionFind<image_pair_t> uf_;
};

}  // namespace glomap_ra
}  // namespace colmap
