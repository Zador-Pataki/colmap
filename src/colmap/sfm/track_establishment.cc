// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
// (BSD-3-Clause license, see LICENSE)

#include "colmap/sfm/track_establishment.h"

#include "colmap/util/logging.h"

namespace colmap {

void TrackEngine::ProcessLoopClosurePairs(const PoseGraph& pose_graph,
                                          Reconstruction& reconstruction) {
  size_t pairs_processed = 0;
  size_t lc_obs_added = 0;

  for (const auto& [pair_id, edge] : pose_graph.Edges()) {
    if (!edge.valid) continue;
    if (!edge.is_LC) continue;

    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);

    if (!image_ids_to_process_.empty()) {
      if (!image_ids_to_process_.count(image_id1) ||
          !image_ids_to_process_.count(image_id2))
        continue;
    }

    if (edge.matches.empty()) continue;

    THROW_CHECK_EQ(edge.are_lc.size(), edge.matches.size());

    pairs_processed++;

    const Image& img1 = reconstruction.Image(image_id1);
    const Image& img2 = reconstruction.Image(image_id2);

    for (size_t i = 0; i < edge.matches.size(); ++i) {
      if (!edge.are_lc[i]) continue;

      const point2D_t feat1 = static_cast<point2D_t>(edge.matches[i][0]);
      const point2D_t feat2 = static_cast<point2D_t>(edge.matches[i][1]);

      const bool has_p3d1 = img1.Point2D(feat1).HasPoint3D();
      const bool has_p3d2 = img2.Point2D(feat2).HasPoint3D();

      if (has_p3d1 && has_p3d2) {
        const point3D_t pid1 = img1.Point2D(feat1).point3D_id;
        const point3D_t pid2 = img2.Point2D(feat2).point3D_id;
        if (pid1 != pid2) {
          LOG(WARNING) << "ProcessLoopClosurePairs: LC pair (" << image_id1
                       << "," << feat1 << ") and (" << image_id2 << "," << feat2
                       << ") belong to different Point3Ds (" << pid1 << " vs "
                       << pid2 << "); appending to first only.";
          reconstruction.Point3D(pid1).track.AddLcElement(image_id2, feat2);
          lc_obs_added++;
        }
      } else if (has_p3d1) {
        const point3D_t pid1 = img1.Point2D(feat1).point3D_id;
        reconstruction.Point3D(pid1).track.AddLcElement(image_id2, feat2);
        lc_obs_added++;
      } else if (has_p3d2) {
        const point3D_t pid2 = img2.Point2D(feat2).point3D_id;
        reconstruction.Point3D(pid2).track.AddLcElement(image_id1, feat1);
        lc_obs_added++;
      } else {
        // Orphan: neither observation belongs to an existing Point3D.
        // Create a stub Point3D with zero xyz; GP will initialize it.
        const point3D_t pid =
            reconstruction.AddPoint3D(Eigen::Vector3d::Zero(), Track{});
        reconstruction.Point3D(pid).track.AddLcElement(image_id1, feat1);
        reconstruction.Point3D(pid).track.AddLcElement(image_id2, feat2);
        lc_obs_added += 2;
      }
    }
  }

  LOG(INFO) << "ProcessLoopClosurePairs: pairs=" << pairs_processed
            << ", lc_observations_added=" << lc_obs_added;
}

}  // namespace colmap
