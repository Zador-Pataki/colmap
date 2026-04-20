#pragma once

namespace colmap::glomap {

struct TrackEstablishmentOptions {
  // Max allowed distance for features in the same track in the same image.
  double thres_inconsistency = 10.0;

  int min_num_tracks_per_view = -1;
  int min_num_view_per_track = 3;
  int max_num_view_per_track = 100;
  int max_num_tracks = 10000000;
};

}  // namespace colmap::glomap
