#include "colmap/sfm/track_establishment_glomap.h"

#include "colmap/sfm/track_establishment.h"

#include <limits>
#include <set>
#include <utility>

namespace colmap {
namespace glomap_ra {
using ViewGraph = colmap::CorrespondenceGraph;
using ImagePair = colmap::CorrespondenceGraph::ImagePair;

size_t TrackEngine::EstablishFullTracks(
    std::unordered_map<track_t, Point3D>& tracks) {
  tracks.clear();

  // Build per-image keypoint coordinates from the fork's ``Image::features``
  // (the native helper does the same from ``image.Points2D()`` but our
  // images_ dict pre-dates the rec-resident points2D).
  std::unordered_map<image_t, std::vector<Eigen::Vector2d>>
      image_id_to_keypoints;
  image_id_to_keypoints.reserve(images_.size());
  for (const auto& [image_id, image] : images_) {
    image_id_to_keypoints.emplace(image_id, image.features);
  }

  // Pre-compute the LC observation set so the predicate is O(log n) per
  // match. The set stays small (only LC inliers across all valid pairs).
  using LCKey = std::pair<uint64_t, uint64_t>;
  auto encode = [](image_t i, point2D_t p) {
    return (static_cast<uint64_t>(i) << 32) | static_cast<uint64_t>(p);
  };
  std::set<LCKey> lc_matches;
  std::vector<image_pair_t> valid_pair_ids;
  valid_pair_ids.reserve(view_graph_.MutableImagePairs().size());
  for (const auto& [pair_id, image_pair] : view_graph_.MutableImagePairs()) {
    if (!image_pair.is_valid) continue;
    valid_pair_ids.push_back(pair_id);
    const auto& matches = image_pair.matches;
    const auto& inliers = image_pair.inliers;
    const auto& are_lc = image_pair.are_lc;
    for (size_t i = 0; i < inliers.size(); ++i) {
      const size_t idx = inliers[i];
      if (idx < are_lc.size() && are_lc[idx]) {
        lc_matches.insert(
            {encode(image_pair.image_id1, matches(idx, 0)),
             encode(image_pair.image_id2, matches(idx, 1))});
      }
    }
  }

  // Native helper handles UF + grouping + intra-image consistency + length
  // filter + greedy subsample. The ignore_match predicate excludes LC
  // observations from the regular pass; ProcessLoopClosurePairs adds them
  // back as parallel ``lc_elements`` below.
  ::colmap::TrackEstablishmentOptions to;
  to.intra_image_consistency_threshold = options_.thres_inconsistency;
  to.min_num_views_per_track = options_.min_num_view_per_track;
  // Fork preserves all consistent + length-filtered tracks at this stage;
  // FindTracksForProblem owns the subsequent greedy subsample.
  to.required_tracks_per_view = std::numeric_limits<int>::max();

  auto ignore_match = [&](image_t i1, point2D_t p1, image_t i2, point2D_t p2) {
    return lc_matches.count({encode(i1, p1), encode(i2, p2)}) > 0;
  };

  tracks = EstablishTracksFromCorrGraph(valid_pair_ids,
                                        view_graph_,
                                        image_id_to_keypoints,
                                        to,
                                        ignore_match);

  // Second pass: add LC observations to existing tracks (or mint new ones
  // for LC-only observation pairs).
  ProcessLoopClosurePairs(tracks);

  return tracks.size();
}

void TrackEngine::ProcessLoopClosurePairs(
    std::unordered_map<track_t, Point3D>& tracks) {
  // Build a lookup from observation (image_id << 32 | feature_id) to track_id
  std::unordered_map<uint64_t, track_t> obs_to_track;
  for (const auto& [track_id, point3D] : tracks) {
    for (const auto& el : point3D.track.Elements()) {
      const uint64_t key = (static_cast<uint64_t>(el.image_id) << 32) |
                           static_cast<uint64_t>(el.point2D_idx);
      obs_to_track.emplace(key, track_id);
    }
  }

  for (const auto& [pair_id, image_pair] : view_graph_.MutableImagePairs()) {
    if (!image_pair.is_valid) continue;

    const Eigen::MatrixXi& matches = image_pair.matches;
    const std::vector<int>& inliers = image_pair.inliers;
    const std::vector<bool>& are_lc = image_pair.are_lc;

    // Only process matches that are marked as LC matches
    bool has_lc_matches = false;
    for (size_t i = 0; i < inliers.size(); i++) {
      const size_t idx = inliers[i];
      if (idx < are_lc.size() && are_lc[idx]) {
        has_lc_matches = true;
        break;
      }
    }
    if (!has_lc_matches) continue;

    for (size_t i = 0; i < inliers.size(); i++) {
      const size_t idx = inliers[i];

      // Only process LC matches
      if (idx >= are_lc.size() || !are_lc[idx]) {
        continue;
      }

      const uint32_t point1_idx = matches(idx, 0);
      const uint32_t point2_idx = matches(idx, 1);

      const image_t image_id1 = image_pair.image_id1;
      const image_t image_id2 = image_pair.image_id2;

      const uint64_t obs1_key = (static_cast<uint64_t>(image_id1) << 32) |
                                static_cast<uint64_t>(point1_idx);
      const uint64_t obs2_key = (static_cast<uint64_t>(image_id2) << 32) |
                                static_cast<uint64_t>(point2_idx);

      const bool has_track1 =
          (obs_to_track.find(obs1_key) != obs_to_track.end());
      const bool has_track2 =
          (obs_to_track.find(obs2_key) != obs_to_track.end());

      if (!has_track1 && !has_track2) {
        // Neither observation exists in any built track: create two new tracks
        // Track A: obsA as regular observation, obsB as LC observation
        // Track B: obsB as regular observation, obsA as LC observation
        const track_t tid_a = static_cast<track_t>(obs1_key);
        const track_t tid_b = static_cast<track_t>(obs2_key);

        Point3D track_a;
        track_a.track.AddElement(image_id1, point1_idx);
        track_a.track.lc_elements.emplace_back(image_id2, point2_idx);

        Point3D track_b;
        track_b.track.AddElement(image_id2, point2_idx);
        track_b.track.lc_elements.emplace_back(image_id1, point1_idx);

        // Insert the new tracks. Track ids derive from observation
        // (image_id, point2D_idx) pairs which must be unique post-MDRP;
        // assert no collision so a violation surfaces fast rather than
        // silently dropping a track.
        const auto inserted_a = tracks.emplace(tid_a, std::move(track_a));
        THROW_CHECK(inserted_a.second)
            << "Track id collision on " << tid_a
            << " — observation key reused unexpectedly";
        const auto inserted_b = tracks.emplace(tid_b, std::move(track_b));
        THROW_CHECK(inserted_b.second)
            << "Track id collision on " << tid_b
            << " — observation key reused unexpectedly";

        // Register in lookup so subsequent LC pairs can find them
        obs_to_track[obs1_key] = tid_a;
        obs_to_track[obs2_key] = tid_b;

        continue;
      }

      if (has_track1 && has_track2) {
        const track_t t1 = obs_to_track.at(obs1_key);
        const track_t t2 = obs_to_track.at(obs2_key);
        if (t1 != t2) {
          // Add reciprocal LC observations without merging tracks
          tracks[t1].track.lc_elements.emplace_back(image_id2, point2_idx);
          tracks[t2].track.lc_elements.emplace_back(image_id1, point1_idx);
        }
        continue;
      }

      // Only one side exists in a track: add the other observation as LC to it
      if (has_track1) {
        const track_t t1 = obs_to_track.at(obs1_key);
        tracks[t1].track.lc_elements.emplace_back(image_id2, point2_idx);
      } else if (has_track2) {
        const track_t t2 = obs_to_track.at(obs2_key);
        tracks[t2].track.lc_elements.emplace_back(image_id1, point1_idx);
      }
    }
  }
}

size_t TrackEngine::FindTracksForProblem(
    const std::unordered_map<track_t, Point3D>& tracks_full,
    std::unordered_map<track_t, Point3D>& tracks_selected) {
  // Sort the tracks by length
  std::vector<std::pair<size_t, track_t>> track_lengths;

  // std::unordered_map<ViewId, std::vector<TrackId>> map_track;
  for (const auto& [track_id, point3D] : tracks_full) {
    const size_t obs_plus_lc =
        point3D.track.Length() + point3D.track.lc_elements.size();
    if (obs_plus_lc < options_.min_num_view_per_track) {
      continue;
    }
    // FUTURE: have a more elegant way of filtering tracks
    if (point3D.track.Length() > options_.max_num_view_per_track) {
      continue;
    }
    track_lengths.emplace_back(
        std::make_pair(point3D.track.Length(), track_id));
  }
  std::sort(std::rbegin(track_lengths), std::rend(track_lengths));

  // Initialize the track per camera number to zero
  std::unordered_map<image_t, track_t> tracks_per_camera;

  // If we only want to select a subset of images, then only add the tracks
  // corresponding to those images
  std::unordered_map<track_t, Point3D> tracks;
  for (const auto& [image_id, image] : images_) {
    if (!image.is_registered) continue;

    tracks_per_camera[image_id] = 0;
  }

  int cameras_left = tracks_per_camera.size();
  for (const auto& [track_length, track_id] : track_lengths) {
    const auto& point3D = tracks_full.at(track_id);

    // Collect the image ids. For each image, only increment the counter by 1
    std::unordered_set<image_t> image_ids;
    Point3D track_temp;
    for (const auto& el : point3D.track.Elements()) {
      if (tracks_per_camera.count(el.image_id) == 0) continue;

      track_temp.track.AddElement(el);
      image_ids.insert(el.image_id);
    }

    // Also carry over LC observations. We copy those whose image is present
    // in the selection domain (registered and tracked by tracks_per_camera).
    for (const auto& lc_el : point3D.track.lc_elements) {
      if (tracks_per_camera.count(lc_el.image_id) == 0) continue;
      track_temp.track.lc_elements.emplace_back(lc_el);
    }

    if (track_temp.track.Length() < options_.min_num_view_per_track) {
      continue;
    }
    if (image_ids.size() == 2) {
      // For tracks with exactly 2 observations, check that both have valid
      // depth priors
      bool all_have_valid_depth = true;
      for (const auto& el : track_temp.track.Elements()) {
        if (images_.find(el.image_id) == images_.end()) {
          all_have_valid_depth = false;
          break;
        }
        const Image& image = images_.at(el.image_id);
        if (!image.is_registered ||
            !image.depth_prior_validity[el.point2D_idx] ||
            image.depth_priors[el.point2D_idx] <= 1e-6) {
          all_have_valid_depth = false;
          break;
        }
      }

      if (!all_have_valid_depth) {
        continue;
      }
    }

    // A flag to see if the track has already been added or not to avoid
    // multiple insertion into the set to be efficient
    bool added = false;
    for (const auto& el : track_temp.track.Elements()) {
      // Getting the current number of tracks
      auto& track_per_camera = tracks_per_camera[el.image_id];
      if (track_per_camera > options_.min_num_tracks_per_view) {
        continue;
      }

      // Otherwise, increase the track number per camera
      ++track_per_camera;
      if (track_per_camera > options_.min_num_tracks_per_view) --cameras_left;

      if (!added) {
        tracks.insert(std::make_pair(track_id, track_temp));
        added = true;
      }
    }
    // Stop iterating if all cameras have enough tracks assigned
    if (cameras_left == 0) break;
    if (tracks.size() > options_.max_num_tracks) break;
  }

  // To avoid flushing the track_full, we copy the selected tracks to the
  // selected tracks
  tracks_selected = tracks;

  return tracks.size();
}

}  // namespace glomap_ra
}  // namespace colmap
