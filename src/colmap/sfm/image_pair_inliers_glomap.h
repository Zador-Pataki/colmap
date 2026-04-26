#pragma once

#include "colmap/sfm/rigid3d_glomap.h"
#include "colmap/scene/camera.h"
#include "colmap/scene/correspondence_graph.h"
#include "colmap/scene/image.h"
#include "colmap/util/types.h"
#include <unordered_map>
#include "colmap/util/logging.h"

namespace colmap {
namespace glomap_ra {
using ViewGraph = colmap::CorrespondenceGraph;
using ImagePair = colmap::CorrespondenceGraph::ImagePair;
// Vendored from glomap/types.h: thresholds used by inlier scoring +
// downstream relpose / triangulation gating. Same shape as
// pyglomap.InlierThresholdOptions for opt_track-style call-site
// projection.
struct InlierThresholdOptions {
  double max_angle_error = 1.;
  double max_reprojection_error = 1e-2;
  double min_triangulation_angle = 1.;
  double max_epipolar_error_E = 1.;
  double max_epipolar_error_F = 4.;
  double max_epipolar_error_H = 4.;
  double min_angle_from_epipole = 3.;
  double min_inlier_num = 30;
  double min_inlier_ratio = 0.25;
  double max_rotation_error = 10.;  // degrees
  // Fork additions for MDRP/depth-aware inlier scoring (the depth-flag-aware
  // branches in image_pair_inliers.cc consult these):
  double thres_epipole = 3.;            // degrees, with depth prior
  double thres_epipole_nodepth = 3.;    // degrees, without depth prior
};


void ImagePairsInlierCount(ViewGraph& view_graph,
                           const std::unordered_map<camera_t, Camera>& cameras,
                           const std::unordered_map<image_t, Image>& images,
                           const InlierThresholdOptions& options,
                           bool clean_inliers);

}  // namespace glomap_ra
}  // namespace colmap
