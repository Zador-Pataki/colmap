#pragma once

#include "colmap/estimators/glomap/view_graph_calibrator_options.h"

#include "colmap/estimators/glomap/optimization_base_options.h"
#include "colmap/glomap/image.h"
#include "colmap/glomap/image_pair.h"
#include "colmap/glomap/track.h"
#include "colmap/glomap/view_graph.h"

#include <memory>

namespace colmap::glomap {

// ViewGraphCalibratorOptions is defined in colmap/estimators/glomap/view_graph_calibrator_options.h (§07).

class ViewGraphCalibrator {
 public:
  ViewGraphCalibrator(const ViewGraphCalibratorOptions& options)
      : options_(options) {}

  // Entry point for the calibration
  bool Solve(ViewGraph& view_graph,
             std::unordered_map<camera_t, Camera>& cameras,
             std::unordered_map<image_t, Image>& images);

 private:
  // Reset the problem
  void Reset(const std::unordered_map<camera_t, Camera>& cameras);

  // Add the image pairs to the problem
  void AddImagePairsToProblem(
      const ViewGraph& view_graph,
      const std::unordered_map<camera_t, Camera>& cameras,
      const std::unordered_map<image_t, Image>& images);

  // Add a single image pair to the problem
  void AddImagePair(const ImagePair& image_pair,
                    const std::unordered_map<camera_t, Camera>& cameras,
                    const std::unordered_map<image_t, Image>& images);

  // Set the cameras to be constant if they have prior intrinsics
  size_t ParameterizeCameras(
      const std::unordered_map<camera_t, Camera>& cameras);

  // Convert the results back to the camera
  void CopyBackResults(std::unordered_map<camera_t, Camera>& cameras);

  // Filter the image pairs based on the calibration results
  size_t FilterImagePairs(ViewGraph& view_graph) const;

  ViewGraphCalibratorOptions options_;
  std::unique_ptr<ceres::Problem> problem_;
  std::unordered_map<camera_t, double> focals_;
  std::shared_ptr<ceres::LossFunction> loss_function_;
};

}  // namespace colmap::glomap
