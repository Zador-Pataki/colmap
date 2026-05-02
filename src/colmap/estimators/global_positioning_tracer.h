#pragma once

#include "colmap/estimators/global_positioning_trace.h"
#include "colmap/scene/track.h"
#include "colmap/util/types.h"

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <ceres/ceres.h>

namespace colmap {

class Reconstruction;

using GlobalPositioningTraceAttrs =
    std::map<std::string, GlobalPositioningTraceValue>;

struct GlobalPositioningResidualDescriptor {
  std::string residual_type;
  std::optional<point3D_t> point3D_id;
  std::optional<image_t> image_id;
  std::optional<point2D_t> point2D_idx;
  std::optional<frame_t> frame_id;
  std::optional<camera_t> camera_id;
  std::optional<sensor_t> sensor_id;
  bool is_lc_observation = false;
  bool is_ref_in_frame = false;
  bool camera_has_prior_focal_length = false;
  std::string loss_bucket = "none";
  bool uses_keypoint_covariance = false;
  bool has_depth_prior = false;
  std::optional<double> depth_prior;
  std::optional<double> depth_sigma;
  std::optional<image_t> dmap_scale_image_id;
  std::string depth_outlier_source = "none";
};

struct GlobalPositioningResidualReplayEntry {
  std::string residual_id;
  const ceres::CostFunction* cost_function = nullptr;
  const ceres::LossFunction* loss_function = nullptr;
  size_t residual_dimension = 0;
  std::vector<int> parameter_block_sizes;
  std::vector<const double*> parameter_blocks;
  std::vector<GlobalPositioningTraceParameterBlockDescriptor>
      parameter_block_descriptors;
};

struct GlobalPositioningTraceLiveState {
  const ceres::Problem& problem;
  const Reconstruction& reconstruction;
  const std::unordered_map<frame_t, Eigen::Vector3d>& frame_centers;
  const std::vector<double>& scales;
  const std::map<image_t, double>& dmap_scales;
  const std::unordered_map<sensor_t, Eigen::Vector3d>& cams_in_rig;
};

class GlobalPositioningTracer {
 public:
  explicit GlobalPositioningTracer(
      const GlobalPositioningTraceOptions& options);
  ~GlobalPositioningTracer();

  bool Enabled() const;
  bool ResidualLedgerEnabled() const;
  bool ParameterSnapshotsEnabled() const;
  bool ResidualValuesEnabled() const;
  bool ResidualJacobiansEnabled() const;

  void WriteEvent(std::string event_type,
                  std::string stage,
                  GlobalPositioningTraceAttrs attrs = {});

  void MarkFinished();

  void ResetProblemState();

  void RecordScaleObservation(size_t scale_index,
                              point3D_t point3D_id,
                              const TrackElement& observation,
                              bool is_lc_observation);

  std::string RecordResidual(
      const GlobalPositioningResidualDescriptor& residual,
      const ceres::CostFunction* cost_function,
      const ceres::LossFunction* loss_function,
      std::vector<const double*> parameter_blocks,
      std::vector<GlobalPositioningTraceParameterBlockDescriptor>
          parameter_block_descriptors);

  void RecordSkip(const GlobalPositioningResidualDescriptor& residual,
                  std::string skip_reason);

  void RecordBucketSummaries();

  std::unique_ptr<ceres::IterationCallback> CreateIterationCallback(
      GlobalPositioningTraceLiveState live_state);

  const std::vector<GlobalPositioningResidualReplayEntry>& ReplayEntries()
      const {
    return residual_replay_entries_;
  }

 private:
  struct ScaleObservationMetadata {
    point3D_t point3D_id = kInvalidPoint3DId;
    image_t image_id = kInvalidImageId;
    point2D_t point2D_idx = kInvalidPoint2DIdx;
    bool is_lc_observation = false;
  };

  GlobalPositioningTraceRecord MakeResidualRecord(
      const GlobalPositioningResidualDescriptor& residual,
      std::string event_type,
      std::string stage) const;

  GlobalPositioningTraceOptions options_;
  std::unique_ptr<GlobalPositioningTraceRecorder> recorder_;
  bool finished_ = false;
  std::map<std::string, uint64_t> residual_bucket_counts_;
  std::vector<ScaleObservationMetadata> scale_observations_;
  std::vector<GlobalPositioningResidualReplayEntry> residual_replay_entries_;
};

}  // namespace colmap
