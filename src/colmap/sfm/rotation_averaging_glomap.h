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

#pragma once

#include "colmap/estimators/rotation_averaging.h"  // shared options struct
#include "colmap/scene/camera.h"
#include "colmap/scene/correspondence_graph.h"
#include "colmap/scene/image.h"
#include "colmap/util/types.h"

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <unordered_map>
#include <utility>
#include <vector>

// Glomap-fork rotation averaging port. Parallel implementation to colmap's
// upstream `EstimateRotations(PoseGraph, …)` — operates on
// CorrespondenceGraph + std::unordered_map<image_t, Image> and supports the
// fork's video-aware solver, precomputed weights, and risky-LC filtering
// (see RotationEstimatorOptions in colmap/estimators/rotation_averaging.h
// for the option surface). The L1 + IRLS core mirrors Theia's
// RobustRotationEstimator (http://www.theia-sfm.org/) and the
// "Gravity Aligned Rotation Averaging" extensions.

namespace colmap {
namespace glomap_ra {

// Glomap's ViewGraph maps to colmap's CorrespondenceGraph (its image_pairs
// hold the same per-pair geometry once the fork-additions on
// CorrespondenceGraph::ImagePair are populated).
using ViewGraph = colmap::CorrespondenceGraph;
using ImagePair = colmap::CorrespondenceGraph::ImagePair;

// Find the largest connected component in the view graph (over valid
// pairs), set ``image.is_registered = true`` for every image in it
// (others get false), and invalidate every pair that crosses the
// component boundary. Returns the number of images in the largest CC.
//
// Mirrors glomap::ViewGraph::KeepLargestConnectedComponents and is
// expected to be called by the pycolmap binding immediately before
// RotationEstimator::EstimateRotations.
int KeepLargestConnectedComponents(ViewGraph& view_graph,
                                   std::unordered_map<image_t, Image>& images);

// Per-pair temporary state held during the iterative solve.
struct ImagePairTempInfo {
  image_pair_t index = -1;
  bool has_gravity = false;
  double xz_error = 0;
  Eigen::Matrix3d R_rel = Eigen::Matrix3d::Identity();
  double angle_rel = 0;
};

// Estimates the global orientations of all views. Returns per-pair weights
// (final IRLS weights, or precomputed weights when use_precomputed_weights).
class RotationEstimator {
 public:
  explicit RotationEstimator(const RotationEstimatorOptions& options)
      : options_(options) {}

  std::unordered_map<image_pair_t, double> EstimateRotations(
      ViewGraph& view_graph, std::unordered_map<image_t, Image>& images);

 protected:
  void InitializeFromMaximumSpanningTree(
      ViewGraph& view_graph, std::unordered_map<image_t, Image>& images);

  void SetupLinearSystem(ViewGraph& view_graph,
                         std::unordered_map<image_t, Image>& images);

  bool SolveL1Regression(ViewGraph& view_graph,
                         std::unordered_map<image_t, Image>& images);

  std::pair<bool, std::unordered_map<image_pair_t, double>> SolveIRLS(
      ViewGraph& view_graph, std::unordered_map<image_t, Image>& images);

  std::pair<bool, std::unordered_map<image_pair_t, double>> SolveWeightedLS(
      ViewGraph& view_graph, std::unordered_map<image_t, Image>& images);

  bool SolveCeres(ViewGraph& view_graph,
                  std::unordered_map<image_t, Image>& images);

  // True when normal inliers >= LC inliers, used by SolveCeres to pick the
  // tracking-vs-LC loss function.
  static bool IsTrackingPair(const ImagePair& image_pair);

  void UpdateGlobalRotations(ViewGraph& view_graph,
                             std::unordered_map<image_t, Image>& images);

  void ComputeResiduals(ViewGraph& view_graph,
                        std::unordered_map<image_t, Image>& images);

  double ComputeAverageStepSize(
      const std::unordered_map<image_t, Image>& images);

  const RotationEstimatorOptions& options_;

  // Sparse linear system A x = b in tangent space.
  Eigen::SparseMatrix<double> sparse_matrix_;
  Eigen::VectorXd tangent_space_step_;
  Eigen::VectorXd tangent_space_residual_;
  Eigen::VectorXd rotation_estimated_;

  std::unordered_map<image_t, image_t> image_id_to_idx_;
  std::unordered_map<image_pair_t, ImagePairTempInfo> rel_temp_info_;

  image_t fixed_camera_id_ = kInvalidImageId;
  Eigen::Vector3d fixed_camera_rotation_;
};

}  // namespace glomap_ra
}  // namespace colmap
