// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "colmap/sfm/relative_pose_estimation.h"

#include "colmap/estimators/solvers/poselib_utils.h"
#include "colmap/util/logging.h"
#include "colmap/util/threading.h"

#include <iostream>
#include <vector>

#include <PoseLib/robust.h>

namespace colmap {

void EstimateRelativePoses(CorrespondenceGraph& view_graph,
                           const std::unordered_map<camera_t, Camera>& cameras,
                           const std::unordered_map<image_t, Image>& images,
                           const RelativePoseEstimationOptions& options) {
  std::vector<image_pair_t> valid_pair_ids;
  for (const auto& [pair_id, image_pair] : view_graph.MutableImagePairs()) {
    if (!image_pair.is_valid) continue;
    valid_pair_ids.push_back(pair_id);
  }

  const int64_t num_image_pairs = valid_pair_ids.size();
  constexpr int64_t kNumChunks = 10;
  const int64_t interval = static_cast<int64_t>(
      std::ceil(static_cast<double>(num_image_pairs) / kNumChunks));

  ThreadPool thread_pool(ThreadPool::kMaxNumThreads);
  LOG(INFO) << "Estimating relative pose for " << num_image_pairs << " pairs";

  for (int64_t chunk = 0; chunk < kNumChunks; ++chunk) {
    std::cout << "\r Estimating relative pose: " << chunk * kNumChunks << "%"
              << std::flush;
    const int64_t start = chunk * interval;
    const int64_t end =
        std::min<int64_t>((chunk + 1) * interval, num_image_pairs);

    for (int64_t i = start; i < end; ++i) {
      thread_pool.AddTask([&, i]() {
        // Reuse heap allocations across tasks on the same thread.
        thread_local std::vector<Eigen::Vector2d> points2D_1;
        thread_local std::vector<Eigen::Vector2d> points2D_2;
        thread_local std::vector<char> inliers;

        auto& image_pair = view_graph.MutableImagePairs().at(valid_pair_ids[i]);
        const Image& image1 = images.at(image_pair.image_id1);
        const Image& image2 = images.at(image_pair.image_id2);
        const Eigen::MatrixXi& matches = image_pair.matches;

        points2D_1.clear();
        points2D_2.clear();
        for (Eigen::Index idx = 0; idx < matches.rows(); ++idx) {
          points2D_1.push_back(image1.features[matches(idx, 0)]);
          points2D_2.push_back(image2.features[matches(idx, 1)]);
        }

        inliers.clear();
        poselib::CameraPose pose_rel;
        try {
          poselib::estimate_relative_pose(
              points2D_1,
              points2D_2,
              ConvertCameraToPoseLibCamera(cameras.at(image1.CameraId())),
              ConvertCameraToPoseLibCamera(cameras.at(image2.CameraId())),
              options.ransac_options,
              options.bundle_options,
              &pose_rel,
              &inliers);
        } catch (const std::exception& e) {
          LOG(ERROR) << "Error in relative pose estimation: " << e.what();
          image_pair.is_valid = false;
          return;
        }

        // PoseLib stores its quaternion as [w, x, y, z]; Eigen as [x, y, z, w].
        // Reorder via (k + 1) % 4 mapping: Eigen idx 0..2 -> poselib 1..3 (x,y,z),
        // Eigen idx 3 -> poselib 0 (w). Same mapping glomap uses.
        Rigid3d cam2_from_cam1;
        for (int k = 0; k < 4; ++k) {
          cam2_from_cam1.rotation().coeffs()[k] = pose_rel.q[(k + 1) % 4];
        }
        cam2_from_cam1.translation() = pose_rel.t;
        image_pair.two_view_geometry.cam2_from_cam1 = cam2_from_cam1;
      });
    }
    thread_pool.Wait();
  }

  std::cout << "\r Estimating relative pose: 100%" << std::endl;
  LOG(INFO) << "Estimating relative pose done";
}

}  // namespace colmap
