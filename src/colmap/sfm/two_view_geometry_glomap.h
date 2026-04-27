#pragma once

#include "colmap/scene/camera.h"
#include "colmap/scene/correspondence_graph.h"
#include "colmap/scene/image.h"
#include "colmap/util/logging.h"
#include "colmap/util/types.h"

// Fork additions kept here because no native colmap4 equivalent exists:
// - CheckCheirality (depth-gated single-pair variant — native batches and
//   has no min/max depth)
// - GetOrientationSignum (vendored from GC-RANSAC for F-cheirality)
// - SampsonError(E, Vec3, Vec3) (depth-aware variant — divides by z+EPS
//   before applying Sampson formula)
//
// Helpers that duplicated native were dropped:
// - EssentialFromMotion -> use colmap::EssentialMatrixFromPose
//   (colmap/geometry/essential_matrix.h:83). Migrated callers in
//   image_pair_inliers_glomap.cc.
// - SampsonError(E, Vec2, Vec2) -> use colmap::ComputeSquaredSampsonError
//   (colmap/geometry/essential_matrix.h:144) with .homogeneous()
//   conversion at the call site. Migrated callers in
//   image_pair_inliers_glomap.cc.
// - HomographyError -> use colmap::ComputeSquaredHomographyError
//   (colmap/geometry/homography_matrix.h:110, note arg order is
//   (point1, point2, H)). Migrated callers in image_pair_inliers_glomap.cc.
// - FundamentalFromMotionAndCameras was a 1-line wrapper with zero
//   call sites — pure dead-code delete (no migration needed). If a
//   future caller appears, build it inline as
//   ``FundamentalFromEssentialMatrix(K2, EssentialMatrixFromPose(pose), K1)``
//   from colmap/geometry/essential_matrix.h.

namespace colmap {
namespace sfm_ext {

// Cheirality check for essential matrix
bool CheckCheirality(const Rigid3d& pose,
                     const Eigen::Vector3d& x1,
                     const Eigen::Vector3d& x2,
                     double min_depth = 0.,
                     double max_depth = 100.);

// Get the orientation signum for fundamental matrix
// For chierality check of fundamental matrix
double GetOrientationSignum(const Eigen::Matrix3d& F,
                            const Eigen::Vector3d& epipole,
                            const Eigen::Vector2d& pt1,
                            const Eigen::Vector2d& pt2);

// Sampson error for the essential matrix
// Input the normalized image ray (3d), divides by z+EPS — fork's
// depth-aware variant. The Vec2 overload that duplicated native
// ComputeSquaredSampsonError was removed; callers use that directly.
double SampsonError(const Eigen::Matrix3d& E,
                    const Eigen::Vector3d& x1,
                    const Eigen::Vector3d& x2);

}  // namespace sfm_ext
}  // namespace colmap
