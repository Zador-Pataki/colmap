#include "track_establishment.h"

#include <glog/logging.h>

namespace colmap::glomap {

size_t TrackEngine::EstablishFullTracks(
    std::unordered_map<track_t, Track>& tracks) {
  tracks.clear();
  uf_.Clear();

  LOG(INFO) << "Establishing full tracks from view graph with "
            << view_graph_.image_pairs.size() << " image pairs";

  // Blindly concatenate tracks if any matches occur
  BlindConcatenation();

  // Iterate through the collected tracks and record the items for each track
  TrackCollection(tracks);

  // Second pass: process loop-closure pairs to add lc_observations without
  // merging
  ProcessLoopClosurePairs(tracks);

  // Compute statistics: mean/median lengths for tracks and lc tracks
  std::vector<size_t> lengths;
  std::vector<size_t> lc_lengths;
  lengths.reserve(tracks.size());
  lc_lengths.reserve(tracks.size());
  size_t sum_len = 0, sum_lc_len = 0;
  for (const auto& [tid, track] : tracks) {
    const size_t len = track.observations.size();
    const size_t lc_len = track.lc_observations.size();
    lengths.push_back(len);
    lc_lengths.push_back(lc_len);
    sum_len += len;
    sum_lc_len += lc_len;
  }
  auto median_of = [](std::vector<size_t>& v) -> double {
    if (v.empty()) return 0.0;
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    if (v.size() % 2 == 1) {
      return static_cast<double>(v[v.size() / 2]);
    } else {
      const auto it1 = std::max_element(v.begin(), v.begin() + v.size() / 2);
      const auto mid2 = v[v.size() / 2];
      return 0.5 * (static_cast<double>(*it1) + static_cast<double>(mid2));
    }
  };
  const double mean_len =
      tracks.empty()
          ? 0.0
          : static_cast<double>(sum_len) / static_cast<double>(tracks.size());
  const double mean_lc_len = tracks.empty()
                                 ? 0.0
                                 : static_cast<double>(sum_lc_len) /
                                       static_cast<double>(tracks.size());
  double median_len = median_of(lengths);
  double median_lc_len = median_of(lc_lengths);
  double max_len = *std::max_element(lengths.begin(), lengths.end());
  double max_lc_len = *std::max_element(lc_lengths.begin(), lc_lengths.end());

  LOG(INFO) << "Track length stats: mean=" << mean_len
            << ", median=" << median_len << ", max=" << max_len;
  LOG(INFO) << "LC track length stats: mean=" << mean_lc_len
            << ", median=" << median_lc_len << ", max=" << max_lc_len;

  LOG(INFO) << "Total tracks established: " << tracks.size();
  return tracks.size();
}

void TrackEngine::BlindConcatenation() {
  // Initialize the union find data structure by connecting all the
  // correspondences
  size_t pairs_processed = 0;
  size_t pairs_skipped_invalid = 0;
  size_t pairs_with_inliers = 0;
  size_t total_inliers_processed = 0;
  size_t lc_matches_skipped = 0;

  size_t counter = 0;
  for (auto pair : view_graph_.image_pairs) {
    if ((counter + 1) % 1000 == 0 ||
        counter == view_graph_.image_pairs.size() - 1) {
      std::cout << "\r Initializing pairs " << counter + 1 << " / "
                << view_graph_.image_pairs.size() << std::flush;
    }
    counter++;

    const image_pair_t pair_id = pair.first;

    const ImagePair& image_pair = pair.second;
    if (!image_pair.is_valid) {
      pairs_skipped_invalid++;
      continue;
    }

    // If filtering by image IDs, only process pairs where both images are in
    // the set
    if (!image_ids_to_process_.empty()) {
      bool id1_in_set = image_ids_to_process_.find(image_pair.image_id1) !=
                        image_ids_to_process_.end();
      bool id2_in_set = image_ids_to_process_.find(image_pair.image_id2) !=
                        image_ids_to_process_.end();
      if (!id1_in_set || !id2_in_set) {
        continue;  // Skip pairs where not both images are in the set
      }
    }

    pairs_processed++;

    // Get the matches
    const Eigen::MatrixXi& matches = image_pair.matches;

    // Get the inlier mask
    const std::vector<int>& inliers = image_pair.inliers;

    // Get the LC match flags
    const std::vector<bool>& are_lc = image_pair.are_lc;

    if (!inliers.empty()) {
      pairs_with_inliers++;
    }

    for (size_t i = 0; i < inliers.size(); i++) {
      size_t idx = inliers[i];

      // Skip LC matches - they will be handled in ProcessLoopClosurePairs
      if (idx < are_lc.size() && are_lc[idx]) {
        lc_matches_skipped++;
        continue;
      }

      total_inliers_processed++;

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

  LOG(INFO) << "BlindConcatenation statistics:";
  LOG(INFO) << "  Total image pairs: " << view_graph_.image_pairs.size();
  LOG(INFO) << "  Pairs skipped (invalid): " << pairs_skipped_invalid;
  LOG(INFO) << "  Pairs processed: " << pairs_processed;
  LOG(INFO) << "  Pairs with inliers: " << pairs_with_inliers;
  LOG(INFO) << "  Total inliers processed: " << total_inliers_processed;
  LOG(INFO) << "  LC matches skipped: " << lc_matches_skipped;
}

void TrackEngine::TrackCollection(std::unordered_map<track_t, Track>& tracks) {
  std::unordered_map<uint64_t, std::unordered_set<uint64_t>> track_map;
  std::unordered_map<uint64_t, int> track_counter;

  // Create tracks from the connected components of the point correspondences
  size_t pairs_processed = 0;
  size_t pairs_skipped_invalid = 0;
  size_t lc_matches_skipped = 0;

  size_t counter = 0;
  for (auto pair : view_graph_.image_pairs) {
    if ((counter + 1) % 1000 == 0 ||
        counter == view_graph_.image_pairs.size() - 1) {
      std::cout << "\r Establishing pairs " << counter + 1 << " / "
                << view_graph_.image_pairs.size() << std::flush;
    }
    counter++;

    const image_pair_t pair_id = pair.first;

    const ImagePair& image_pair = pair.second;
    if (!image_pair.is_valid) {
      pairs_skipped_invalid++;
      continue;
    }

    // If filtering by image IDs, only process pairs where both images are in
    // the set
    if (!image_ids_to_process_.empty()) {
      bool id1_in_set = image_ids_to_process_.find(image_pair.image_id1) !=
                        image_ids_to_process_.end();
      bool id2_in_set = image_ids_to_process_.find(image_pair.image_id2) !=
                        image_ids_to_process_.end();
      if (!id1_in_set || !id2_in_set) {
        continue;  // Skip pairs where not both images are in the set
      }
    }

    pairs_processed++;

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
        lc_matches_skipped++;
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
        track_counter[track_id]++;
      }
    }
  }
  std::cout << std::endl;

  counter = 0;
  size_t discarded_counter = 0;
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
          discarded_counter++;
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

  size_t tracks_kept = tracks.size();
  size_t tracks_in_map = track_map.size();

  LOG(INFO) << "TrackCollection statistics:";
  LOG(INFO) << "  Tracks in map (before consistency check): " << tracks_in_map;
  LOG(INFO) << "  Tracks discarded (inconsistency): " << discarded_counter;
  LOG(INFO) << "  Tracks kept (after consistency check): " << tracks_kept;
  LOG(INFO) << "  LC matches skipped: " << lc_matches_skipped;
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

  size_t pairs_processed = 0;
  size_t lc_obs_added = 0;

  for (const auto& [pair_id, image_pair] : view_graph_.image_pairs) {
    if (!image_pair.is_valid) continue;

    // If filtering by image IDs, only process pairs where both images are in
    // the set
    if (!image_ids_to_process_.empty()) {
      bool id1_in_set = image_ids_to_process_.find(image_pair.image_id1) !=
                        image_ids_to_process_.end();
      bool id2_in_set = image_ids_to_process_.find(image_pair.image_id2) !=
                        image_ids_to_process_.end();
      if (!id1_in_set || !id2_in_set) {
        continue;  // Skip pairs where not both images are in the set
      }
    }

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

    pairs_processed++;

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

        lc_obs_added += 2;
        continue;
      }

      if (has_track1 && has_track2) {
        const track_t t1 = obs_to_track.at(obs1_key);
        const track_t t2 = obs_to_track.at(obs2_key);
        if (t1 != t2) {
          // Add reciprocal LC observations without merging tracks
          tracks[t1].lc_observations.emplace_back(image_id2, point2_idx);
          tracks[t2].lc_observations.emplace_back(image_id1, point1_idx);
          lc_obs_added += 2;
        }
        continue;
      }

      // Only one side exists in a track: add the other observation as LC to it
      if (has_track1) {
        const track_t t1 = obs_to_track.at(obs1_key);
        tracks[t1].lc_observations.emplace_back(image_id2, point2_idx);
        lc_obs_added++;
      } else if (has_track2) {
        const track_t t2 = obs_to_track.at(obs2_key);
        tracks[t2].lc_observations.emplace_back(image_id1, point1_idx);
        lc_obs_added++;
      }
    }
  }

  LOG(INFO) << "ProcessLoopClosurePairs: processed pairs with LC matches="
            << pairs_processed << ", lc_observations added=" << lc_obs_added;
}

size_t TrackEngine::FindTracksForProblem(
    const std::unordered_map<track_t, Track>& tracks_full,
    std::unordered_map<track_t, Track>& tracks_selected) {
  // Sort the tracks by length
  std::vector<std::pair<size_t, track_t>> track_lengths;

  size_t tracks_too_short = 0;
  size_t tracks_too_long = 0;
  size_t tracks_invalid_depth_2views = 0;
  size_t tracks_too_few_images = 0;
  size_t tracks_per_view_limit = 0;

  // std::unordered_map<ViewId, std::vector<TrackId>> map_track;
  for (const auto& [track_id, track] : tracks_full) {
    const size_t obs_plus_lc =
        track.observations.size() + track.lc_observations.size();
    if (obs_plus_lc < options_.min_num_view_per_track) {
      tracks_too_short++;
      continue;
    }
    // FUTURE: have a more elegant way of filtering tracks
    if (track.observations.size() > options_.max_num_view_per_track) {
      tracks_too_long++;
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
      tracks_too_few_images++;
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
        tracks_invalid_depth_2views++;
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
        if (!added) {
          tracks_per_view_limit++;
        }
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

  LOG(INFO) << "FindTracksForProblem statistics:";
  LOG(INFO) << "  Total input tracks: " << tracks_full.size();
  LOG(INFO) << "  Tracks skipped (too short, < "
            << options_.min_num_view_per_track
            << " views): " << tracks_too_short;
  LOG(INFO) << "  Tracks skipped (too long, > "
            << options_.max_num_view_per_track
            << " views): " << tracks_too_long;
  LOG(INFO) << "  Tracks skipped (too few images after filtering): "
            << tracks_too_few_images;
  LOG(INFO) << "  Tracks skipped (2-view tracks without valid depth): "
            << tracks_invalid_depth_2views;
  LOG(INFO) << "  Tracks skipped (per-view limit reached): "
            << tracks_per_view_limit;
  LOG(INFO) << "  Tracks selected: " << tracks.size() << " (out of "
            << track_lengths.size() << " eligible tracks)";

  return tracks.size();
}

}  // namespace colmap::glomap
