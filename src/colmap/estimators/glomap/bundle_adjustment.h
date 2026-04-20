#pragma once

#include "colmap/estimators/glomap/bundle_adjustment_options.h"

#include "colmap/estimators/glomap/iteration_callback.h"
#include "colmap/estimators/glomap/optimization_base_options.h"
#include "colmap/glomap/image.h"
#include "colmap/glomap/image_pair.h"
#include "colmap/glomap/track.h"
#include "colmap/glomap/view_graph.h"


#include <ceres/ceres.h>
#include <functional>

namespace colmap::glomap {

// BundleAdjustmentOptions is defined in colmap/estimators/glomap/bundle_adjustment_options.h (§07).

class BundleAdjuster {
 public:
  BundleAdjuster(const BundleAdjustmentOptions& options) : options_(options) {}

  // Returns true if the optimization was a success, false if there was a
  // failure.
  // Assume tracks here are already filtered
  bool Solve(const ViewGraph& view_graph,
             std::unordered_map<camera_t, Camera>& cameras,
             std::unordered_map<image_t, Image>& images,
             std::unordered_map<track_t, Track>& tracks,
             IterationCallbackFn iteration_callback = nullptr);

  BundleAdjustmentOptions& GetOptions() { return options_; }

 private:
  // Reset the problem
  void Reset();

  // Add tracks to the problem
  void AddPointToCameraConstraints(
      const ViewGraph& view_graph,
      std::unordered_map<camera_t, Camera>& cameras,
      std::unordered_map<image_t, Image>& images,
      std::unordered_map<track_t, Track>& tracks);

  // Set the parameter groups
  void AddCamerasAndPointsToParameterGroups(
      std::unordered_map<camera_t, Camera>& cameras,
      std::unordered_map<image_t, Image>& images,
      std::unordered_map<track_t, Track>& tracks);

  // Parameterize the variables, set some variables to be constant if desired
  void ParameterizeVariables(std::unordered_map<camera_t, Camera>& cameras,
                             std::unordered_map<image_t, Image>& images,
                             std::unordered_map<track_t, Track>& tracks);

  BundleAdjustmentOptions options_;

  std::unique_ptr<ceres::Problem> problem_;
  std::shared_ptr<ceres::LossFunction> loss_function_;
};

}  // namespace colmap::glomap
