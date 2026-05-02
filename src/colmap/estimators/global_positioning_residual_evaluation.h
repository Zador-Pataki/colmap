#pragma once

#include "colmap/estimators/global_positioning_trace.h"
#include "colmap/estimators/global_positioning_tracer.h"

#include <vector>

#include <ceres/ceres.h>

namespace colmap {

struct GlobalPositioningResidualEvaluationOptions {
  const ceres::Problem& problem;
  int iteration = 0;
  const std::vector<GlobalPositioningResidualReplayEntry>& replay_entries;
  bool write_raw_jacobians = false;
};

GlobalPositioningTraceResidualValues EvaluateGlobalPositioningResiduals(
    const GlobalPositioningResidualEvaluationOptions& options);

}  // namespace colmap
