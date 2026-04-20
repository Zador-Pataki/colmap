#pragma once

namespace colmap::glomap {

struct InlierThresholdOptions {
  // 3D-2D thresholds.
  double max_angle_error = 1.0;           // degrees (global positioning)
  double max_reprojection_error = 1e-2;   // bundle adjustment
  double min_triangulation_angle = 1.0;   // degrees

  // image-pair thresholds.
  double max_epipolar_error_E = 1.0;
  double max_epipolar_error_F = 4.0;
  double max_epipolar_error_H = 4.0;
  double min_angle_from_epipole = 3.0;

  // edge thresholds.
  double min_inlier_num = 30.0;
  double min_inlier_ratio = 0.25;
  double max_rotation_error = 10.0;  // degrees (rotation averaging)
};

}  // namespace colmap::glomap
