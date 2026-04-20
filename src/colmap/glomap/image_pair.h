#pragma once

#include "colmap/glomap/types.h"

#include <colmap/estimators/two_view_geometry.h>

#include <iosfwd>
#include <vector>

#include <Eigen/Core>

namespace colmap::glomap {

// Pair-classification tag consumed by §09 (RA) and §10 (GP) for loss / weight
// selection. Fork's legacy `is_LC` bool is retained alongside for call sites
// that have not migrated.
enum class PairType : int {
  ADJACENT = 0,
  NONADJACENT = 1,
  LOOP_CLOSURE = 2,
};

struct ImagePair {
  ImagePair() : pair_id(-1), image_id1(-1), image_id2(-1) {}
  ImagePair(image_t img_id1,
            image_t img_id2,
            colmap::Rigid3d pose_rel = colmap::Rigid3d())
      : pair_id(ImagePairToPairId(img_id1, img_id2)),
        image_id1(img_id1),
        image_id2(img_id2),
        cam2_from_cam1(pose_rel) {}

  // Ids are kept constant.
  const image_pair_t pair_id;
  const image_t image_id1;
  const image_t image_id2;

  // Pair classification; see enum PairType above.
  PairType type = PairType::ADJACENT;

  // Whether the image pair is valid.
  bool is_valid = true;

  // Whether this is a loop closure (legacy; retained alongside type).
  bool is_LC = false;

  // Initial inlier rate.
  double weight = 0;

  // Relative depth scale between the two depth maps; depth_2 = scale * depth_1.
  // -1 means "not yet computed".
  double rel_depth_scale = -1.0;

  // One of `colmap::TwoViewGeometry::ConfigurationType`.
  int config = colmap::TwoViewGeometry::UNDEFINED;

  // Epipolar / homography matrices.
  Eigen::Matrix3d E = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d F = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d H = Eigen::Matrix3d::Zero();

  // Relative pose.
  colmap::Rigid3d cam2_from_cam1;

  // 3x3 covariance for relative translation (zero = not computed).
  Eigen::Matrix3d cov_t = Eigen::Matrix3d::Zero();

  // Matches as Nx2 indices: col 0 feature idx in image1, col 1 in image2.
  Eigen::MatrixXi matches;

  // Row indices of inlier matches.
  std::vector<int> inliers;

  // Whether each match is a loop-closure match (same size as matches.rows()).
  std::vector<bool> are_lc;

  static image_pair_t ImagePairToPairId(image_t image_id1, image_t image_id2);
  static void PairIdToImagePair(image_pair_t pair_id,
                                image_t& image_id1,
                                image_t& image_id2);
};

std::ostream& operator<<(std::ostream& stream, const ImagePair& tform);

}  // namespace colmap::glomap
