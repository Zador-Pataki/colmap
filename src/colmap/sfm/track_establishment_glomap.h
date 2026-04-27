
#pragma once

#include "colmap/scene/camera.h"
#include "colmap/scene/correspondence_graph.h"
#include "colmap/scene/image.h"
#include "colmap/scene/point3d.h"
#include "colmap/util/types.h"
#include <unordered_map>

#include <unordered_set>

// The ``colmap::sfm_ext`` namespace marks SFM-extension symbols added on
// top of upstream colmap4 (track establishment + track filtering +
// pair-inlier scoring). It lets a colmap reviewer see at a glance what is
// fork-extension vs upstream-native code. pycolmap exposes these
// algorithms under a ``sfm_ext`` submodule
// (e.g. ``pycolmap.sfm_ext.filter_tracks_by_angle``); a Python-side alias
// ``pycolmap.glomap_ra = pycolmap.sfm_ext`` keeps legacy import paths
// working. The C++ types underneath reuse native colmap
// (``Point3D``/``Track``/``UnionFind``). Filenames keep the ``_glomap``
// suffix for now — file rename is a separate, larger churn.
namespace colmap {
namespace sfm_ext {
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
  // Iterate over loop-closure pairs and add cross-track LC observations
  // to the dict produced by the native UF + grouping helper.
  void ProcessLoopClosurePairs(std::unordered_map<track_t, Point3D>& tracks);

  const TrackEstablishmentOptions& options_;

  ViewGraph& view_graph_;
  const std::unordered_map<image_t, Image>& images_;
};

}  // namespace sfm_ext
}  // namespace colmap
