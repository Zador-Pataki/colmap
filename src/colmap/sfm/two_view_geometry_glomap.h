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
// Helpers that DID duplicate native (EssentialFromMotion, the Vec2 overload
// of SampsonError, HomographyError, FundamentalFromMotionAndCameras) were
// dropped in favor of colmap::EssentialMatrixFromPose,
// colmap::ComputeSquaredSampsonError, colmap::ComputeSquaredHomographyError,
// and colmap::FundamentalFromEssentialMatrix respectively.

namespace colmap {
namespace glomap_ra {

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

}  // namespace glomap_ra
}  // namespace colmap
