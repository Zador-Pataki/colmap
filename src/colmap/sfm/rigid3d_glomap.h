#pragma once

#include "colmap/util/types.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/util/logging.h"

#include <Eigen/Geometry>

// TODO(dedup-glomap-vs-colmap4): DegToRad/RadToDeg duplicate
// colmap::DegToRad/RadToDeg at colmap/math/math.h:59-64,188-200.
// RotationToAngleAxis / AngleAxisToRotation duplicate
// RotationMatrixToAngleAxis / AngleAxisToRotationMatrix at
// colmap/geometry/pose.cc:131,136. The 4 short Calc* helpers and
// Rigid3dToAngleAxis are unique. See
// .claude/notes/glomap_audit/audit_glomap_files_vs_colmap4.md.

namespace colmap {
namespace glomap_ra {

// Calculate the rotation angle difference between two poses
double CalcAngle(const Rigid3d& pose1, const Rigid3d& pose2);

// Calculate the center difference between two poses
double CalcTrans(const Rigid3d& pose1, const Rigid3d& pose2);

// Calculatet the translation direction difference between two poses
double CalcTransAngle(const Rigid3d& pose1, const Rigid3d& pose2);

// Calculate the rotation angle difference between two rotations
double CalcAngle(const Eigen::Matrix3d& rotation1,
                 const Eigen::Matrix3d& rotation2);

// Convert degree to radian
double DegToRad(double degree);

// Convert radian to degree
double RadToDeg(double radian);

// Convert pose to angle axis
Eigen::Vector3d Rigid3dToAngleAxis(const Rigid3d& pose);

// Convert rotation matrix to angle axis
Eigen::Vector3d RotationToAngleAxis(const Eigen::Matrix3d& rot);

// Convert angle axis to rotation matrix
Eigen::Matrix3d AngleAxisToRotation(const Eigen::Vector3d& aa);

}  // namespace glomap_ra
}  // namespace colmap
