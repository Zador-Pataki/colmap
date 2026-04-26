#include "colmap/sfm/track_establishment_glomap.h"

namespace colmap {
namespace glomap_ra {
using ViewGraph = colmap::CorrespondenceGraph;
using ImagePair = colmap::CorrespondenceGraph::ImagePair;


size_t TrackEngine::EstablishFullTracks(
    std::unordered_map<track_t, Track>& tracks) {
  tracks.clear();
  uf_.Clear();

  // Blindly concatenate tracks if any matches occur
  BlindConcatenation();

  // Iterate through the collected tracks and record the items for each track
  TrackCollection(tracks);

  // Second pass: process loop-closure pairs to add lc_observations without
  // merging
  ProcessLoopClosurePairs(tracks);



  return tracks.size();
}

void TrackEngine::BlindConcatenation() {
  // Initialize the union find data structure by connecting all the
  // correspondences
  size_t counter = 0;
  for (auto pair : view_graph_.MutableImagePairs()) {
    if ((counter + 1) % 1000 == 0 ||
        counter == view_graph_.MutableImagePairs().size() - 1) {
      std::cout << "\r Initializing pairs " << counter + 1 << " / "
                << view_graph_.MutableImagePairs().size() << std::flush;
    }
    counter++;

    const image_pair_t pair_id = pair.first;

    const ImagePair& image_pair = pair.second;
    if (!image_pair.is_valid) {
      continue;
    }

    // Get the matches
    const Eigen::MatrixXi& matches = image_pair.matches;

    // Get the inlier mask
    const std::vector<int>& inliers = image_pair.inliers;

    // Get the LC match flags
    const std::vector<bool>& are_lc = image_pair.are_lc;

    for (size_t i = 0; i < inliers.size(); i++) {
      size_t idx = inliers[i];

      // Skip LC matches - they will be handled in ProcessLoopClosurePairs
      if (idx < are_lc.size() && are_lc[idx]) {
        continue;
      }

      // Get point indices
      const uint32_t& point1_idx = matches(idx, 0);
      const uint32_t& point2_idx = matches(idx, 1);

      image_pair_t point_global_id1 =
          static_cast<image_pair_t>(image_pair.image_id1) << 32 |
          static_cast<image_pair_t>(point1_idx);
      image_pair_t point_global_id2 =
          static_cast<image_pair_t>(image_pair.image_id2) << 32 |
          static_cast<image_pair_t>(point2_idx);

      // Link the first point to the second point. Take the smallest one as the
      // root
      if (point_global_id2 < point_global_id1) {
        uf_.Union(point_global_id1, point_global_id2);
      } else
        uf_.Union(point_global_id2, point_global_id1);
    }
  }
  std::cout << std::endl;
}

void TrackEngine::TrackCollection(std::unordered_map<track_t, Track>& tracks) {
  std::unordered_map<uint64_t, std::unordered_set<uint64_t>> track_map;

  size_t counter = 0;
  for (auto pair : view_graph_.MutableImagePairs()) {
    if ((counter + 1) % 1000 == 0 ||
        counter == view_graph_.MutableImagePairs().size() - 1) {
      std::cout << "\r Establishing pairs " << counter + 1 << " / "
                << view_graph_.MutableImagePairs().size() << std::flush;
    }
    counter++;

    const image_pair_t pair_id = pair.first;

    const ImagePair& image_pair = pair.second;
    if (!image_pair.is_valid) {
      continue;
    }

    // Get the matches
    const Eigen::MatrixXi& matches = image_pair.matches;

    // Get the inlier mask
    const std::vector<int>& inliers = image_pair.inliers;

    // Get the LC match flags
    const std::vector<bool>& are_lc = image_pair.are_lc;

    for (size_t i = 0; i < inliers.size(); i++) {
      size_t idx = inliers[i];

      // Skip LC matches - they will be handled in ProcessLoopClosurePairs
      if (idx < are_lc.size() && are_lc[idx]) {
        continue;
      }

      // Get point indices
      const uint32_t& point1_idx = matches(idx, 0);
      const uint32_t& point2_idx = matches(idx, 1);

      image_pair_t point_global_id1 =
          static_cast<image_pair_t>(image_pair.image_id1) << 32 |
          static_cast<image_pair_t>(point1_idx);
      image_pair_t point_global_id2 =
          static_cast<image_pair_t>(image_pair.image_id2) << 32 |
          static_cast<image_pair_t>(point2_idx);

      // Only add to track_map if this correspondence was actually processed in
      // BlindConcatenation (i.e., if it exists in the union-find structure) We
      // check by seeing if Find returns a value that was actually connected
      image_pair_t track_id = uf_.Find(point_global_id1);

      // Check if point_global_id1 was actually in a union (not just a singleton
      // created by Find) If it's a singleton that wasn't unioned, Find will
      // return itself, but we only want correspondences that were actually
      // processed. We can verify by checking if the track_id matches
      // point_global_id1 and if point_global_id2 is also in the same track
      image_pair_t track_id2 = uf_.Find(point_global_id2);

      // Only add if both points are in the union-find and connected (or if
      // they're the same track) This ensures we only include correspondences
      // that were processed in BlindConcatenation
      if (track_id == track_id2) {
        track_map[track_id].insert(point_global_id1);
        track_map[track_id].insert(point_global_id2);
      }
    }
  }
  std::cout << std::endl;

  counter = 0;
  for (auto& [track_id, correspondence_set] : track_map) {
    if ((counter + 1) % 1000 == 0 || counter == track_map.size() - 1) {
      std::cout << "\r Establishing tracks " << counter + 1 << " / "
                << track_map.size() << std::flush;
    }
    counter++;

    std::unordered_map<image_t, std::vector<Eigen::Vector2d>> image_id_set;
    for (auto point_global_id : correspondence_set) {
      // image_id is the higher 32 bits and feature_id is the lower 32 bits
      image_t image_id = point_global_id >> 32;
      feature_t feature_id = point_global_id & 0xFFFFFFFF;
      if (image_id_set.find(image_id) != image_id_set.end()) {
        for (const auto& feature : image_id_set.at(image_id)) {
          if ((feature - images_.at(image_id).features[feature_id]).norm() >
              options_.thres_inconsistency) {
            tracks[track_id].observations.clear();
            break;
          }
        }
        if (tracks[track_id].observations.size() == 0) {
          break;
        }
      } else
        image_id_set.insert(
            std::make_pair(image_id, std::vector<Eigen::Vector2d>()));

      image_id_set[image_id].push_back(
          images_.at(image_id).features[feature_id]);

      tracks[track_id].observations.emplace_back(image_id, feature_id);
    }
  }

  std::cout << std::endl;

}

void TrackEngine::ProcessLoopClosurePairs(
    std::unordered_map<track_t, Track>& tracks) {
  // Build a lookup from observation (image_id << 32 | feature_id) to track_id
  std::unordered_map<uint64_t, track_t> obs_to_track;
  for (const auto& [track_id, track] : tracks) {
    for (const auto& obs : track.observations) {
      const uint64_t key = (static_cast<uint64_t>(obs.first) << 32) |
                           static_cast<uint64_t>(obs.second);
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
        Track track_a;
        track_a.track_id = static_cast<track_t>(obs1_key);
        track_a.observations.emplace_back(image_id1, point1_idx);
        track_a.lc_observations.emplace_back(image_id2, point2_idx);

        Track track_b;
        track_b.track_id = static_cast<track_t>(obs2_key);
        track_b.observations.emplace_back(image_id2, point2_idx);
        track_b.lc_observations.emplace_back(image_id1, point1_idx);

        // Insert the new tracks
        tracks.emplace(track_a.track_id, std::move(track_a));
        tracks.emplace(track_b.track_id, std::move(track_b));

        // Register in lookup so subsequent LC pairs can find them
        obs_to_track[obs1_key] = static_cast<track_t>(obs1_key);
        obs_to_track[obs2_key] = static_cast<track_t>(obs2_key);

        continue;
      }

      if (has_track1 && has_track2) {
        const track_t t1 = obs_to_track.at(obs1_key);
        const track_t t2 = obs_to_track.at(obs2_key);
        if (t1 != t2) {
          // Add reciprocal LC observations without merging tracks
          tracks[t1].lc_observations.emplace_back(image_id2, point2_idx);
          tracks[t2].lc_observations.emplace_back(image_id1, point1_idx);
        }
        continue;
      }

      // Only one side exists in a track: add the other observation as LC to it
      if (has_track1) {
        const track_t t1 = obs_to_track.at(obs1_key);
        tracks[t1].lc_observations.emplace_back(image_id2, point2_idx);
      } else if (has_track2) {
        const track_t t2 = obs_to_track.at(obs2_key);
        tracks[t2].lc_observations.emplace_back(image_id1, point1_idx);
      }
    }
  }
}

size_t TrackEngine::FindTracksForProblem(
    const std::unordered_map<track_t, Track>& tracks_full,
    std::unordered_map<track_t, Track>& tracks_selected) {
  // Sort the tracks by length
  std::vector<std::pair<size_t, track_t>> track_lengths;

  // std::unordered_map<ViewId, std::vector<TrackId>> map_track;
  for (const auto& [track_id, track] : tracks_full) {
    const size_t obs_plus_lc =
        track.observations.size() + track.lc_observations.size();
    if (obs_plus_lc < options_.min_num_view_per_track) {
      continue;
    }
    // FUTURE: have a more elegant way of filtering tracks
    if (track.observations.size() > options_.max_num_view_per_track) {
      continue;
    }
    track_lengths.emplace_back(
        std::make_pair(track.observations.size(), track_id));
  }
  std::sort(std::rbegin(track_lengths), std::rend(track_lengths));

  // Initialize the track per camera number to zero
  std::unordered_map<image_t, track_t> tracks_per_camera;

  // If we only want to select a subset of images, then only add the tracks
  // corresponding to those images
  std::unordered_map<track_t, Track> tracks;
  for (const auto& [image_id, image] : images_) {
    if (!image.is_registered) continue;

    tracks_per_camera[image_id] = 0;
  }

  int cameras_left = tracks_per_camera.size();
  for (const auto& [track_length, track_id] : track_lengths) {
    const auto& track = tracks_full.at(track_id);

    // Collect the image ids. For each image, only increment the counter by 1
    std::unordered_set<image_t> image_ids;
    Track track_temp;
    for (const auto& [image_id, feature_id] : track.observations) {
      if (tracks_per_camera.count(image_id) == 0) continue;

      track_temp.track_id = track_id;
      track_temp.observations.emplace_back(
          std::make_pair(image_id, feature_id));
      image_ids.insert(image_id);
    }

    // Also carry over LC observations. We copy those whose image is present
    // in the selection domain (registered and tracked by tracks_per_camera).
    for (const auto& [lc_image_id, lc_feature_id] : track.lc_observations) {
      if (tracks_per_camera.count(lc_image_id) == 0) continue;
      track_temp.lc_observations.emplace_back(
          std::make_pair(lc_image_id, lc_feature_id));
    }

    // const size_t obs_plus_lc_selected =
    //     track_temp.observations.size() + track_temp.lc_observations.size();
    if (track_temp.observations.size() < options_.min_num_view_per_track) {
      continue;
    }
    if (image_ids.size() == 2) {
      // For tracks with exactly 2 observations, check that both have valid
      // depth priors
      bool all_have_valid_depth = true;
      for (const auto& [image_id, feature_id] : track_temp.observations) {
        if (images_.find(image_id) == images_.end()) {
          all_have_valid_depth = false;
          break;
        }
        const Image& image = images_.at(image_id);
        if (!image.is_registered || !image.depth_prior_validity[feature_id] ||
            image.depth_priors[feature_id] <= 1e-6) {
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
    // for (auto &image_id : image_ids) {
    for (const auto& [image_id, feature_id] : track_temp.observations) {
      // Getting the current number of tracks
      auto& track_per_camera = tracks_per_camera[image_id];
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
