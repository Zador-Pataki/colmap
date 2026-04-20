#pragma once

namespace colmap::glomap {

// Retriangulation thresholds.
struct TriangulatorOptions {
  double tri_complete_max_reproj_error = 15.0;
  double tri_merge_max_reproj_error = 15.0;
  double tri_min_angle = 1.0;

  int min_num_matches = 15;
};

}  // namespace colmap::glomap
