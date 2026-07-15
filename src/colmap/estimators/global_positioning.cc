#include "colmap/estimators/global_positioning.h"

#include "colmap/estimators/cost_functions/metric_depth.h"
#include "colmap/estimators/cost_functions/motion_averaging.h"
#include "colmap/estimators/cost_functions/utils.h"
#include "colmap/math/random.h"
#include "colmap/util/cuda.h"
#include "colmap/util/misc.h"
#include "colmap/util/threading.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <set>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace colmap {
namespace {

std::string GpObservationKey(point3D_t point3D_id,
                             image_t image_id,
                             point2D_t point2D_idx,
                             bool is_lc_observation) {
  return std::to_string(point3D_id) + ":" + std::to_string(image_id) + ":" +
         std::to_string(point2D_idx) + ":" + (is_lc_observation ? "1" : "0");
}

Eigen::Vector3d RandVector3d(double low, double high) {
  return Eigen::Vector3d(RandomUniformReal(low, high),
                         RandomUniformReal(low, high),
                         RandomUniformReal(low, high));
}

uint64_t StableHashAppend(uint64_t hash, const uint64_t value) {
  constexpr uint64_t kPrime = 1099511628211ULL;
  for (int shift = 0; shift < 64; shift += 8) {
    hash ^= static_cast<unsigned char>((value >> shift) & 0xff);
    hash *= kPrime;
  }
  return hash;
}

template <typename MapT>
std::vector<typename MapT::key_type> SortedKeys(const MapT& values) {
  std::vector<typename MapT::key_type> keys;
  keys.reserve(values.size());
  for (const auto& [key, value] : values) {
    keys.push_back(key);
  }
  std::sort(keys.begin(), keys.end());
  return keys;
}

std::string EnvValue(const char* name) {
  const char* value = std::getenv(name);
  return value == nullptr ? "" : value;
}

const char* GpStateDumpPath() {
  static const char* path = std::getenv("MPSFM_GP_STATE_DUMP");
  return path;
}

FILE* GpObsDumpFile() {
  static const char* path = std::getenv("MPSFM_GP_OBS_DUMP");
  static FILE* file = path == nullptr ? nullptr : std::fopen(path, "a");
  return file;
}

void GpObsDumpSolveMarker() {
  FILE* f = GpObsDumpFile();
  if (f == nullptr) return;
  std::fprintf(f, "solve_begin\n");
  std::fflush(f);
}

void GpObsDumpObservation(const char* tag,
                          point3D_t point3D_id,
                          const TrackElement& observation,
                          bool is_lc_observation,
                          const Eigen::Vector3d& feature_undist,
                          const Eigen::Vector3d& v_ik,
                          const Image& image,
                          const Point3D& point3D,
                          const Eigen::Vector3d& center,
                          bool use_depth,
                          bool use_soft_depth_loss,
                          bool is_runtime_depth_outlier,
                          double bata_scale,
                          double dmap_scale) {
  FILE* f = GpObsDumpFile();
  if (f == nullptr) return;
  const point2D_t feature_id = observation.point2D_idx;
  const Eigen::Vector2d angular_std = feature_id < image.angular_stddevs.size()
                                          ? image.angular_stddevs[feature_id]
                                          : Eigen::Vector2d::Constant(-1.0);
  const bool depth_valid = feature_id < image.depth_prior_validity.size() &&
                           image.depth_prior_validity[feature_id];
  const double depth_prior = feature_id < image.depth_priors.size()
                                 ? image.depth_priors[feature_id]
                                 : -1.0;
  const double depth_sigma = feature_id < image.depth_prior_stddevs.size()
                                 ? image.depth_prior_stddevs[feature_id]
                                 : -1.0;
  const Eigen::Vector4d q = image.CamFromWorld().rotation().coeffs();
  std::fprintf(
      f,
      "%s|track=%llu|image=%llu|feature=%llu|lc=%d|"
      "feature=%.17g,%.17g,%.17g|v=%.17g,%.17g,%.17g|"
      "center=%.17g,%.17g,%.17g|point=%.17g,%.17g,%.17g|"
      "q=%.17g,%.17g,%.17g,%.17g|"
      "angular=%.17g,%.17g|depth_valid=%d|depth=%.17g|depth_sigma=%.17g|"
      "use_depth=%d|soft_depth=%d|runtime_depth_outlier=%d|inlier=%d|"
      "anchor=%d|mdrp_depth_outlier=%d|bata_scale=%.17g|dmap_scale=%.17g\n",
      tag,
      static_cast<unsigned long long>(point3D_id),
      static_cast<unsigned long long>(observation.image_id),
      static_cast<unsigned long long>(feature_id),
      is_lc_observation ? 1 : 0,
      feature_undist(0),
      feature_undist(1),
      feature_undist(2),
      v_ik(0),
      v_ik(1),
      v_ik(2),
      center(0),
      center(1),
      center(2),
      point3D.xyz(0),
      point3D.xyz(1),
      point3D.xyz(2),
      q(0),
      q(1),
      q(2),
      q(3),
      angular_std(0),
      angular_std(1),
      depth_valid ? 1 : 0,
      depth_prior,
      depth_sigma,
      use_depth ? 1 : 0,
      use_soft_depth_loss ? 1 : 0,
      is_runtime_depth_outlier ? 1 : 0,
      observation.is_inlier ? 1 : 0,
      observation.is_track_anchor ? 1 : 0,
      observation.is_depth_outlier ? 1 : 0,
      bata_scale,
      dmap_scale);
}

void GpStateDump(
    const char* tag,
    const Reconstruction& reconstruction,
    const std::unordered_map<frame_t, Eigen::Vector3d>& frame_centers,
    const std::vector<double>& scales,
    const std::unordered_map<std::string, size_t>& bata_scale_indices,
    const std::map<image_t, double>& dmap_scales) {
  const char* path = GpStateDumpPath();
  if (path == nullptr) return;
  FILE* f = std::fopen(path, "a");
  if (f == nullptr) return;

  std::vector<image_t> image_ids;
  image_ids.reserve(reconstruction.Images().size());
  for (const auto& [image_id, image] : reconstruction.Images()) {
    image_ids.push_back(image_id);
  }
  std::sort(image_ids.begin(), image_ids.end());
  for (const image_t image_id : image_ids) {
    const Image& image = reconstruction.Image(image_id);
    const auto frame_center_it = frame_centers.find(image.FrameId());
    const Eigen::Vector3d center = frame_center_it == frame_centers.end()
                                       ? image.CamFromWorld().TgtOriginInSrc()
                                       : frame_center_it->second;
    const Eigen::Vector4d q = image.CamFromWorld().rotation().coeffs();
    std::fprintf(f,
                 "%s|image|%llu|t=%.17g,%.17g,%.17g|q=%.17g,%.17g,%.17g,%.17g|"
                 "nfeat=%llu|ndepth=%llu|reg=%d|frame=%llu\n",
                 tag,
                 static_cast<unsigned long long>(image_id),
                 center(0),
                 center(1),
                 center(2),
                 q(0),
                 q(1),
                 q(2),
                 q(3),
                 static_cast<unsigned long long>(image.NumPoints2D()),
                 static_cast<unsigned long long>(image.depth_priors.size()),
                 image.HasPose() ? 1 : 0,
                 static_cast<unsigned long long>(image.FrameId()));
  }

  std::vector<point3D_t> point3D_ids;
  point3D_ids.reserve(reconstruction.Points3D().size());
  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    point3D_ids.push_back(point3D_id);
  }
  std::sort(point3D_ids.begin(), point3D_ids.end());
  for (const point3D_t point3D_id : point3D_ids) {
    const Point3D& point3D = reconstruction.Point3D(point3D_id);
    std::fprintf(
        f,
        "%s|track|%llu|xyz=%.17g,%.17g,%.17g|init=%d|nobs=%llu|nlc=%llu\n",
        tag,
        static_cast<unsigned long long>(point3D_id),
        point3D.xyz(0),
        point3D.xyz(1),
        point3D.xyz(2),
        1,
        static_cast<unsigned long long>(point3D.track.Length()),
        static_cast<unsigned long long>(point3D.track.lc_elements.size()));
  }

  std::vector<std::string> scale_keys;
  scale_keys.reserve(bata_scale_indices.size());
  for (const auto& [key, index] : bata_scale_indices) {
    scale_keys.push_back(key);
  }
  std::sort(scale_keys.begin(), scale_keys.end());
  for (const std::string& key : scale_keys) {
    const size_t index = bata_scale_indices.at(key);
    if (index >= scales.size()) continue;
    std::fprintf(
        f, "%s|bata_scale|%s|value=%.17g\n", tag, key.c_str(), scales[index]);
  }

  for (const auto& [image_id, scale] : dmap_scales) {
    std::fprintf(f,
                 "%s|dmap_scale|%llu|value=%.17g\n",
                 tag,
                 static_cast<unsigned long long>(image_id),
                 scale);
  }
  std::fclose(f);
}

uint64_t StableHashAppendDouble(uint64_t hash, const double value) {
  uint64_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(value));
  return StableHashAppend(hash, bits);
}

// Per-observation depth outlier check. Returns true when the log-space
// residual |log(z_est / scaled_prior)| exceeds sigma * sigma_log.
inline bool DepthOutlierFlag(const Image& image,
                             point2D_t feature_id,
                             const Eigen::Vector3d& point3D_xyz,
                             bool use_log_scale,
                             const std::map<image_t, double>& dmap_scales,
                             image_t image_id,
                             double sigma) {
  if (feature_id >= image.depth_prior_validity.size() ||
      !image.depth_prior_validity[feature_id]) {
    return false;
  }
  if (feature_id >= image.depth_priors.size() ||
      feature_id >= image.depth_prior_stddevs.size()) {
    return false;
  }
  const double depth_prior_raw = image.depth_priors[feature_id];
  const double stddev_rel = image.depth_prior_stddevs[feature_id];
  if (depth_prior_raw <= 1e-6 || stddev_rel <= 1e-9) return false;

  // Apply per-image dmap_scale if available (else use raw prior).
  double depth_prior = depth_prior_raw;
  auto scale_it = dmap_scales.find(image_id);
  if (scale_it != dmap_scales.end()) {
    const double dmap_scale =
        use_log_scale ? std::exp(scale_it->second) : scale_it->second;
    depth_prior = dmap_scale * depth_prior_raw;
  }

  // z_est = (cam_from_world * X_world)[2]
  const Eigen::Vector3d point_cam = image.CamFromWorld() * point3D_xyz;
  const double z_est = point_cam[2];
  if (z_est <= 1e-6) return false;

  const double log_z_est = std::log(std::max(z_est, 1e-6));
  const double log_depth_prior = std::log(std::max(depth_prior, 1e-6));
  const double log_diff = std::abs(log_z_est - log_depth_prior);
  const double threshold = sigma * std::log(1.0 + std::max(stddev_rel, 1e-6));
  return log_diff >= threshold;
}

size_t NumRegularObservationsForMinViewGate(const Track& track) {
  return track.Length();
}

bool IsLossConfigOverride(const LossConfig& loss_config) {
  return loss_config.type != LossFunctionType::TRIVIAL ||
         loss_config.scale != 1.0 || loss_config.weight != 1.0;
}

uint64_t SplitMix64(uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

class PlaybackIterationCallback final : public ceres::IterationCallback {
 public:
  PlaybackIterationCallback(const int interval,
                            std::function<void(int)> capture)
      : interval_(interval), capture_(std::move(capture)) {}

  ceres::CallbackReturnType operator()(
      const ceres::IterationSummary& summary) override {
    if (summary.iteration % interval_ == 0) {
      capture_(summary.iteration);
    }
    return ceres::SOLVER_CONTINUE;
  }

 private:
  int interval_;
  std::function<void(int)> capture_;
};

}  // namespace

GlobalPositioner::GlobalPositioner(const GlobalPositionerOptions& options)
    : options_(options) {
  if (options_.random_seed >= 0) {
    SetPRNGSeed(static_cast<unsigned>(options_.random_seed));
  }
}

bool GlobalPositioner::Solve(const PoseGraph& pose_graph,
                             Reconstruction& reconstruction) {
  if (reconstruction.NumImages() == 0) {
    LOG(ERROR) << "Number of images = " << reconstruction.NumImages();
    return false;
  }
  if (reconstruction.NumPoints3D() == 0) {
    LOG(ERROR) << "Number of tracks = " << reconstruction.NumPoints3D();
    return false;
  }

  // TODO: extend rig branch in AddObservationToProblem to add MetricDepthError
  // for non-ref images. Until then, fail loud on multi-camera rigs +
  // use_metric_depth_constraint.
  if (options_.use_metric_depth_constraint) {
    for (const auto& [image_id, image] : reconstruction.Images()) {
      THROW_CHECK(image.IsRefInFrame())
          << "use_metric_depth_constraint=true is not yet supported with "
             "multi-camera rigs. Image "
          << image_id
          << " is a non-ref sensor in its frame; its depth "
             "residual would be silently dropped. Either disable "
             "use_metric_depth_constraint or run on single-camera rigs.";
    }
  }

  LOG(INFO) << "Setting up the global positioner problem";

  GpObsDumpSolveMarker();

  // Setup the problem.
  SetupProblem(pose_graph, reconstruction);

  // Initialize camera translations to be random.
  // Also, convert the camera pose translation to be the camera center.
  InitializeRandomPositions(pose_graph, reconstruction);

  // No caller-supplied seed for dmap_scales_; derive one from per-image
  // median observed z_est/depth_prior.
  if (options_.use_metric_depth_constraint && options_.use_init &&
      !options_.initial_dmap_scales.has_value()) {
    InitializeDepthMapScalesFromObservations(reconstruction);
  }

  // Add the point to camera constraints to the problem.
  AddPointToCameraConstraints(reconstruction);

  if (options_.use_parameter_block_ordering) {
    AddCamerasAndPointsToParameterGroups(reconstruction);
  }

  // Parameterize the variables, set image poses / tracks / scales to be
  // constant if desired
  ParameterizeVariables(reconstruction);

  LOG(INFO) << "Solving the global positioner problem";

  ceres::Solver::Summary summary;
  options_.solver_options.num_threads =
      GetEffectiveNumThreads(options_.solver_options.num_threads);
  options_.solver_options.minimizer_progress_to_stdout = VLOG_IS_ON(2);
  if (const char* log_path = std::getenv("MPSFM_GP_DETERMINISM_LOG")) {
    std::ofstream log(log_path, std::ios::app);
    if (log) {
      constexpr uint64_t kFnvOffset = 1469598103934665603ULL;
      uint64_t scale_order_hash = kFnvOffset;
      for (size_t idx = 0; idx < scales_.size(); ++idx) {
        scale_order_hash = StableHashAppend(scale_order_hash, idx);
      }
      uint64_t dmap_order_hash = kFnvOffset;
      for (const auto& [image_id, scale] : dmap_scales_) {
        dmap_order_hash = StableHashAppend(dmap_order_hash, image_id);
      }
      uint64_t parameter_value_hash = kFnvOffset;
      for (const double scale : scales_) {
        parameter_value_hash =
            StableHashAppendDouble(parameter_value_hash, scale);
      }
      std::vector<point3D_t> point3D_ids;
      point3D_ids.reserve(reconstruction.NumPoints3D());
      for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
        point3D_ids.push_back(point3D_id);
      }
      std::sort(point3D_ids.begin(), point3D_ids.end());
      for (const point3D_t point3D_id : point3D_ids) {
        Point3D& point3D = reconstruction.Point3D(point3D_id);
        if (problem_->HasParameterBlock(point3D.xyz.data())) {
          parameter_value_hash =
              StableHashAppend(parameter_value_hash, point3D_id);
          for (int i = 0; i < 3; ++i) {
            parameter_value_hash =
                StableHashAppendDouble(parameter_value_hash, point3D.xyz[i]);
          }
        }
      }
      std::vector<frame_t> frame_ids;
      frame_ids.reserve(frame_centers_.size());
      for (const auto& [frame_id, center] : frame_centers_) {
        frame_ids.push_back(frame_id);
      }
      std::sort(frame_ids.begin(), frame_ids.end());
      for (const frame_t frame_id : frame_ids) {
        Eigen::Vector3d& center = frame_centers_.at(frame_id);
        if (problem_->HasParameterBlock(center.data())) {
          parameter_value_hash =
              StableHashAppend(parameter_value_hash, frame_id);
          for (int i = 0; i < 3; ++i) {
            parameter_value_hash =
                StableHashAppendDouble(parameter_value_hash, center[i]);
          }
        }
      }
      std::vector<sensor_t> sensor_ids;
      sensor_ids.reserve(cams_in_rig_.size());
      for (const auto& [sensor_id, center] : cams_in_rig_) {
        sensor_ids.push_back(sensor_id);
      }
      std::sort(sensor_ids.begin(), sensor_ids.end());
      for (const sensor_t sensor_id : sensor_ids) {
        Eigen::Vector3d& center = cams_in_rig_.at(sensor_id);
        if (problem_->HasParameterBlock(center.data())) {
          parameter_value_hash = StableHashAppend(
              parameter_value_hash, static_cast<uint64_t>(sensor_id.id));
          parameter_value_hash = StableHashAppend(
              parameter_value_hash, static_cast<uint64_t>(sensor_id.type));
          for (int i = 0; i < 3; ++i) {
            parameter_value_hash =
                StableHashAppendDouble(parameter_value_hash, center[i]);
          }
        }
      }
      for (auto& [image_id, scale] : dmap_scales_) {
        if (problem_->HasParameterBlock(&scale)) {
          parameter_value_hash =
              StableHashAppend(parameter_value_hash, image_id);
          parameter_value_hash =
              StableHashAppendDouble(parameter_value_hash, scale);
        }
      }
      log << "gp_determinism_pre_solve"
          << " residual_blocks=" << problem_->NumResidualBlocks()
          << " parameter_blocks=" << problem_->NumParameterBlocks()
          << " num_threads=" << options_.solver_options.num_threads
          << " linear_solver_type="
          << static_cast<int>(options_.solver_options.linear_solver_type)
          << " preconditioner_type="
          << static_cast<int>(options_.solver_options.preconditioner_type)
          << " sparse_linear_algebra_library_type="
          << static_cast<int>(
                 options_.solver_options.sparse_linear_algebra_library_type)
          << " use_parameter_block_ordering="
          << static_cast<int>(options_.use_parameter_block_ordering)
          << " scales=" << scales_.size()
          << " frame_centers=" << frame_centers_.size()
          << " cams_in_rig=" << cams_in_rig_.size()
          << " dmap_scales=" << dmap_scales_.size()
          << " residual_order_hash=" << residual_order_hash_
          << " parameter_value_hash=" << parameter_value_hash
          << " scale_order_hash=" << scale_order_hash
          << " dmap_order_hash=" << dmap_order_hash
          << " OPENBLAS_NUM_THREADS=" << EnvValue("OPENBLAS_NUM_THREADS")
          << " OMP_NUM_THREADS=" << EnvValue("OMP_NUM_THREADS")
          << " MKL_NUM_THREADS=" << EnvValue("MKL_NUM_THREADS")
          << " BLIS_NUM_THREADS=" << EnvValue("BLIS_NUM_THREADS")
          << " VECLIB_MAXIMUM_THREADS=" << EnvValue("VECLIB_MAXIMUM_THREADS")
          << " NUMEXPR_NUM_THREADS=" << EnvValue("NUMEXPR_NUM_THREADS") << "\n";
    }
  }
  GpStateDump("pre_ceres_solve",
              reconstruction,
              frame_centers_,
              scales_,
              bata_scale_indices_,
              dmap_scales_);
  try {
    if (!options_.playback.IsEnabled()) {
      ceres::Solve(options_.solver_options, problem_.get(), &summary);
    } else {
      THROW_CHECK_GT(options_.playback.snapshot_every_n_iterations, 0)
          << "playback.snapshot_every_n_iterations must be positive";
      WritePlaybackCapture("initial", -1, reconstruction);
      ceres::Solver::Options playback_solver_options = options_.solver_options;
      PlaybackIterationCallback callback(
          options_.playback.snapshot_every_n_iterations,
          [this, &reconstruction](const int iteration) {
            WritePlaybackCapture("iteration", iteration, reconstruction);
          });
      playback_solver_options.update_state_every_iteration = true;
      playback_solver_options.callbacks.push_back(&callback);
      ceres::Solve(playback_solver_options, problem_.get(), &summary);
      if (summary.IsSolutionUsable()) {
        const int final_iteration = summary.iterations.empty()
                                        ? -1
                                        : summary.iterations.back().iteration;
        WritePlaybackCapture("final", final_iteration, reconstruction);
      }
    }
  } catch (...) {
    ConvertBackResults(reconstruction);
    throw;
  }
  if (const char* path = GpStateDumpPath()) {
    FILE* f = std::fopen(path, "a");
    if (f != nullptr) {
      std::fprintf(
          f,
          "ceres_summary|init=%.17g|final=%.17g|iters=%d|term=%d|"
          "linear_solver=%d|preconditioner=%d|threading=%d|"
          "n_resid_blocks=%d|n_param_blocks=%d|n_params=%d\n",
          summary.initial_cost,
          summary.final_cost,
          static_cast<int>(summary.iterations.size()),
          static_cast<int>(summary.termination_type),
          static_cast<int>(options_.solver_options.linear_solver_type),
          static_cast<int>(options_.solver_options.preconditioner_type),
          options_.solver_options.num_threads,
          summary.num_residual_blocks,
          summary.num_parameter_blocks,
          summary.num_parameters);
      for (size_t i = 0; i < summary.iterations.size() && i < 50; ++i) {
        std::fprintf(f,
                     "ceres_iter|%zu|cost=%.17g|step_size=%.17g|tr=%.17g\n",
                     i,
                     summary.iterations[i].cost,
                     summary.iterations[i].step_norm,
                     summary.iterations[i].trust_region_radius);
      }
      std::fclose(f);
    }
  }
  diagnostics_.num_bata_scales = static_cast<int>(scales_.size());
  diagnostics_.num_dmap_scales = static_cast<int>(dmap_scales_.size());
  diagnostics_.num_frame_centers = static_cast<int>(
      UseImageCenterBlocks() ? image_centers_.size() : frame_centers_.size());
  diagnostics_.num_point3D_xyz = static_cast<int>(initial_point3D_xyz_.size());
  diagnostics_.num_residual_blocks = summary.num_residual_blocks;
  diagnostics_.num_parameter_blocks = summary.num_parameter_blocks;
  diagnostics_.num_parameters = summary.num_parameters;
  diagnostics_.num_iterations = static_cast<int>(summary.iterations.size());
  diagnostics_.initial_cost = summary.initial_cost;
  diagnostics_.final_cost = summary.final_cost;
  diagnostics_.termination_type = static_cast<int>(summary.termination_type);

  if (VLOG_IS_ON(2)) {
    LOG(INFO) << summary.FullReport();
  } else {
    LOG(INFO) << summary.BriefReport();
  }

  GpStateDump("post_ceres_solve",
              reconstruction,
              frame_centers_,
              scales_,
              bata_scale_indices_,
              dmap_scales_);
  ConvertBackResults(reconstruction);
  return summary.IsSolutionUsable();
}

void GlobalPositioner::SetupProblem(const PoseGraph& pose_graph,
                                    const Reconstruction& reconstruction) {
  ceres::Problem::Options problem_options;
  problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  problem_ = std::make_unique<ceres::Problem>(problem_options);
  loss_function_ = options_.CreateLossFunction();
  diagnostics_ = GlobalPositionerDiagnostics();
  residual_order_hash_ = 1469598103934665603ULL;

  // Clear temporary storage from previous runs.
  frame_centers_.clear();
  image_centers_.clear();
  initial_frame_centers_.clear();
  initial_point3D_xyz_.clear();
  initial_bata_scales_.clear();
  bata_scale_indices_.clear();
  playback_observations_.clear();
  playback_image_ids_.clear();
  playback_point3D_ids_.clear();
  playback_edges_.clear();
  playback_topology_ready_ = false;
  cams_in_rig_.clear();
  per_image_scale_losses_.clear();

  // Reserve scales_ for both regular observations and lc_elements.
  // Underestimating triggers ``vector::push_back`` reallocation mid-build,
  // which invalidates the ``&scale`` data pointers that earlier residual
  // blocks already stored.
  scales_.clear();
  size_t total_observations = 0;
  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    total_observations += point3D.track.Length();
    total_observations += point3D.track.lc_elements.size();
  }
  scales_.reserve(total_observations);
}

void GlobalPositioner::InitializeRandomPositions(
    const PoseGraph& pose_graph, Reconstruction& reconstruction) {
  std::unordered_set<frame_t> constrained_positions;
  constrained_positions.reserve(reconstruction.NumFrames());
  for (const auto& [pair_id, edge] : pose_graph.ValidEdges()) {
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    constrained_positions.insert(reconstruction.Image(image_id1).FrameId());
    constrained_positions.insert(reconstruction.Image(image_id2).FrameId());
  }

  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    if (NumRegularObservationsForMinViewGate(point3D.track) <
        static_cast<size_t>(options_.min_num_view_per_track)) {
      continue;
    }
    for (const auto& observation : point3D.track.Elements()) {
      THROW_CHECK(reconstruction.ExistsImage(observation.image_id));
      const Image& image = reconstruction.Image(observation.image_id);
      if (!image.HasPose()) continue;
      constrained_positions.insert(image.FrameId());
    }
    if (options_.use_lc_observations) {
      for (const auto& observation : point3D.track.lc_elements) {
        if (!reconstruction.ExistsImage(observation.image_id)) continue;
        const Image& image = reconstruction.Image(observation.image_id);
        if (!image.HasPose()) continue;
        constrained_positions.insert(image.FrameId());
      }
    }
  }

  if (UseImageCenterBlocks()) {
    std::vector<image_t> image_ids;
    image_ids.reserve(reconstruction.Images().size());
    for (const auto& [image_id, image] : reconstruction.Images()) {
      image_ids.push_back(image_id);
    }
    std::sort(image_ids.begin(), image_ids.end());

    for (const image_t image_id : image_ids) {
      const Image& image = reconstruction.Image(image_id);
      const frame_t frame_id = image.FrameId();
      if (constrained_positions.find(frame_id) == constrained_positions.end()) {
        continue;
      }
      const Frame& frame = reconstruction.Frame(frame_id);
      Eigen::Vector3d center;
      if (options_.generate_random_positions && options_.optimize_positions &&
          !options_.use_init) {
        center = options_.random_init_scale * RandVector3d(-1, 1);
      } else {
        center = frame.RigFromWorld().TgtOriginInSrc();
      }
      const auto debug_it = options_.debug_initial_frame_centers.find(frame_id);
      if (debug_it != options_.debug_initial_frame_centers.end()) {
        center = debug_it->second;
      }
      image_centers_[image_id] = center;
      frame_centers_[frame_id] = center;
      initial_frame_centers_[frame_id] = center;
    }
    VLOG(2) << "Constrained positions: " << constrained_positions.size();
    return;
  }

  std::vector<frame_t> frame_ids;
  frame_ids.reserve(reconstruction.Frames().size());
  for (const auto& [frame_id, frame] : reconstruction.Frames()) {
    frame_ids.push_back(frame_id);
  }
  std::sort(frame_ids.begin(), frame_ids.end());

  // The reconstruction poses remain in cam_from_world convention.
  for (const frame_t frame_id : frame_ids) {
    if (constrained_positions.find(frame_id) == constrained_positions.end()) {
      continue;
    }
    const Frame& frame = reconstruction.Frame(frame_id);
    Eigen::Vector3d center;
    if (options_.generate_random_positions && options_.optimize_positions &&
        !options_.use_init) {
      center = options_.random_init_scale * RandVector3d(-1, 1);
    } else {
      center = frame.RigFromWorld().TgtOriginInSrc();
    }
    const auto debug_it = options_.debug_initial_frame_centers.find(frame_id);
    if (debug_it != options_.debug_initial_frame_centers.end()) {
      center = debug_it->second;
    }
    if (UseFrameInplaceCenterBlocks()) {
      reconstruction.Frame(frame_id).RigFromWorld().translation() = center;
    } else {
      frame_centers_[frame_id] = center;
    }
    frame_centers_[frame_id] = center;
    initial_frame_centers_[frame_id] = center;
  }

  VLOG(2) << "Constrained positions: " << constrained_positions.size();
}

void GlobalPositioner::AddPointToCameraConstraints(
    Reconstruction& reconstruction) {
  VLOG(2) << reconstruction.NumPoints3D()
          << " point to camera constraints were added to the position "
             "estimation problem.";

  // Down-weight uncalibrated cameras.
  if (options_.apply_uncalibrated_loss_downweight) {
    loss_function_ptcam_uncalibrated_ = std::make_shared<ceres::ScaledLoss>(
        loss_function_.get(),
        options_.uncalibrated_loss_downweight,
        ceres::DO_NOT_TAKE_OWNERSHIP);
  } else {
    loss_function_ptcam_uncalibrated_ = loss_function_;
  }
  loss_function_ptcam_calibrated_ = loss_function_;

  // Initialize cascade losses.
  cached_loss_normal_geometry_ =
      options_.loss_normal_geometry.CreateLossFunction();
  cached_loss_normal_depth_ = options_.loss_normal_depth.CreateLossFunction();
  cached_loss_lc_geometry_ =
      IsLossConfigOverride(options_.loss_lc_geometry)
          ? options_.loss_lc_geometry.CreateLossFunction()
          : nullptr;
  cached_loss_lc_depth_ = options_.loss_lc_depth.CreateLossFunction();
  cached_loss_normal_geometry_inlier_ =
      options_.loss_normal_geometry_inlier.CreateLossFunction();
  cached_loss_normal_depth_inlier_ =
      options_.loss_normal_depth_inlier.CreateLossFunction();
  cached_loss_normal_depth_outlier_ =
      options_.loss_normal_depth_outlier.CreateLossFunction();
  cached_loss_normal_geometry_trackstart_ =
      options_.loss_normal_geometry_trackstart.CreateLossFunction();
  cached_loss_normal_depth_trackstart_ =
      options_.loss_normal_depth_trackstart.CreateLossFunction();
  cached_loss_scale_prior_ = options_.loss_scale_prior.CreateLossFunction();
  soft_outlier_fallback_loss_.reset();

  dmap_scales_.clear();
  dmap_scale_observation_counts_.clear();

  // Seed dmap_scales_ from initial values before outlier filtering.
  if (options_.use_metric_depth_constraint &&
      options_.initial_dmap_scales.has_value()) {
    for (const auto& [image_id, linear_scale] : *options_.initial_dmap_scales) {
      const double init_value = options_.use_log_scale_for_depth_map_scales
                                    ? std::log(std::max(linear_scale, 1e-9))
                                    : linear_scale;
      dmap_scales_[image_id] = init_value;
      dmap_scale_observation_counts_[image_id] = 0;
    }
  }

  depth_outliers_.clear();
  if (options_.use_metric_depth_constraint && options_.filter_depth_outliers) {
    FilterDepthOutliers(reconstruction);
  }

  if (EnvValue("MPSFM_GP_RESIDUAL_ORDER_MODE") == "legacy_unordered") {
    for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
      if (NumRegularObservationsForMinViewGate(point3D.track) <
          static_cast<size_t>(options_.min_num_view_per_track)) {
        continue;
      }
      AddPoint3DToProblem(point3D_id, reconstruction);
    }
  } else {
    std::vector<point3D_t> point3D_ids;
    point3D_ids.reserve(reconstruction.NumPoints3D());
    for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
      point3D_ids.push_back(point3D_id);
    }
    std::sort(point3D_ids.begin(), point3D_ids.end());

    for (const point3D_t point3D_id : point3D_ids) {
      const Point3D& point3D = reconstruction.Point3D(point3D_id);
      if (NumRegularObservationsForMinViewGate(point3D.track) <
          static_cast<size_t>(options_.min_num_view_per_track)) {
        continue;
      }
      AddPoint3DToProblem(point3D_id, reconstruction);
    }
  }

  // Emit one scale-prior residual per image with depth observations,
  // weighted by obs_count so dense-depth images get stronger priors.
  if (options_.use_metric_depth_constraint) {
    for (const image_t image_id : SortedKeys(dmap_scales_)) {
      double& scale = dmap_scales_.at(image_id);
      auto count_it = dmap_scale_observation_counts_.find(image_id);
      const double obs_count =
          (count_it != dmap_scale_observation_counts_.end())
              ? static_cast<double>(count_it->second)
              : 1.0;

      // Prior: pulls log_s toward 0 (log-space) or s toward 1 (linear).
      const Eigen::Matrix<double, 1, 1> prior_vec(
          options_.use_log_scale_for_depth_map_scales ? 0.0 : 1.0);
      const Eigen::Matrix<double, 1, 1> cov_1x1(options_.scale_prior_stddev *
                                                options_.scale_prior_stddev);
      ceres::CostFunction* scale_prior_cost =
          CovarianceWeightedCostFunctor<NormalPriorCostFunctor<1>>::Create(
              cov_1x1, prior_vec);
      if (scale_prior_cost == nullptr) continue;

      ceres::LossFunction* obs_count_scaled_loss = nullptr;
      if (cached_loss_scale_prior_) {
        per_image_scale_losses_.push_back(
            std::make_unique<ceres::ScaledLoss>(cached_loss_scale_prior_.get(),
                                                obs_count,
                                                ceres::DO_NOT_TAKE_OWNERSHIP));
      } else {
        per_image_scale_losses_.push_back(std::make_unique<ceres::ScaledLoss>(
            new ceres::TrivialLoss(), obs_count, ceres::TAKE_OWNERSHIP));
      }
      obs_count_scaled_loss = per_image_scale_losses_.back().get();

      problem_->AddResidualBlock(
          scale_prior_cost, obs_count_scaled_loss, &scale);
      residual_order_hash_ = StableHashAppend(residual_order_hash_, 3);
      residual_order_hash_ = StableHashAppend(residual_order_hash_, image_id);
      ++diagnostics_.num_scale_prior_residuals;
    }
  }
  VLOG(2) << "GP: residual blocks=" << problem_->NumResidualBlocks()
          << ", parameter blocks=" << problem_->NumParameterBlocks()
          << ", scales=" << scales_.size()
          << ", frame_centers=" << frame_centers_.size()
          << ", dmap_scales=" << dmap_scales_.size();
}

void GlobalPositioner::AddPoint3DToProblem(point3D_t point3D_id,
                                           Reconstruction& reconstruction) {
  const bool random_initialization = options_.optimize_points &&
                                     options_.generate_random_points &&
                                     !options_.use_init;

  Point3D& point3D = reconstruction.Point3D(point3D_id);

  // Only set the points to be random if they are needed to be optimized
  if (random_initialization) {
    point3D.xyz = options_.random_init_scale * RandVector3d(-1, 1);
  }
  const auto debug_it = options_.debug_initial_point3D_xyz.find(point3D_id);
  if (debug_it != options_.debug_initial_point3D_xyz.end()) {
    point3D.xyz = debug_it->second;
  }
  initial_point3D_xyz_[point3D_id] = point3D.xyz;

  // Walk regular elements then LC elements as separate passes — they
  // share the residual layout but use different loss function groups.
  for (const auto& observation : point3D.track.Elements()) {
    AddObservationToProblem(
        point3D_id, observation, random_initialization, reconstruction);
  }
  if (options_.use_lc_observations) {
    for (const auto& observation : point3D.track.lc_elements) {
      AddObservationToProblem(point3D_id,
                              observation,
                              random_initialization,
                              reconstruction,
                              /*is_lc_observation=*/true);
    }
  }
}

void GlobalPositioner::RecordPlaybackObservation(
    const TrackElement& observation,
    const bool is_lc_observation,
    const ceres::ResidualBlockId residual_block_id,
    ceres::LossFunction* const loss_function) {
  if (!options_.playback.IsEnabled() || !is_lc_observation) {
    return;
  }
  THROW_CHECK_NE(observation.lc_anchor_image_id, kInvalidImageId)
      << "Playback requires exact LC image-pair provenance";
  THROW_CHECK_NE(observation.lc_anchor_point2D_idx, kInvalidPoint2DIdx)
      << "Playback requires exact LC match provenance";
  THROW_CHECK_NE(observation.image_id, observation.lc_anchor_image_id)
      << "Playback LC endpoints must belong to different images";
  playback_observations_.push_back({observation.image_id,
                                    observation.point2D_idx,
                                    observation.lc_anchor_image_id,
                                    observation.lc_anchor_point2D_idx,
                                    residual_block_id,
                                    loss_function});
}

void GlobalPositioner::WritePlaybackCapture(
    const char* const phase,
    const int iteration,
    const Reconstruction& reconstruction) {
  constexpr size_t kPointLimit = 200000;
  if (!playback_topology_ready_) {
    playback_image_ids_.reserve(reconstruction.NumImages());
    for (const auto& [image_id, image] : reconstruction.Images()) {
      if (!image.HasPose()) {
        continue;
      }
      const bool has_center =
          UseImageCenterBlocks()
              ? image_centers_.find(image_id) != image_centers_.end()
              : frame_centers_.find(image.FrameId()) != frame_centers_.end();
      if (has_center) {
        playback_image_ids_.push_back(image_id);
      }
    }
    std::sort(playback_image_ids_.begin(), playback_image_ids_.end());

    playback_point3D_ids_.reserve(reconstruction.NumPoints3D());
    for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
      playback_point3D_ids_.push_back(point3D_id);
    }
    std::sort(playback_point3D_ids_.begin(), playback_point3D_ids_.end());
    if (playback_point3D_ids_.size() > kPointLimit) {
      std::sort(playback_point3D_ids_.begin(),
                playback_point3D_ids_.end(),
                [](const point3D_t lhs, const point3D_t rhs) {
                  return std::pair(SplitMix64(static_cast<uint64_t>(lhs)),
                                   lhs) <
                         std::pair(SplitMix64(static_cast<uint64_t>(rhs)), rhs);
                });
      playback_point3D_ids_.resize(kPointLimit);
      std::sort(playback_point3D_ids_.begin(), playback_point3D_ids_.end());
    }

    const std::set<image_t> playback_images(playback_image_ids_.begin(),
                                            playback_image_ids_.end());
    using ObservationKey = std::pair<image_t, point2D_t>;
    using MatchKey = std::pair<ObservationKey, ObservationKey>;
    std::map<std::pair<image_t, image_t>, std::vector<size_t>>
        edge_observations;
    std::map<std::pair<image_t, image_t>, std::set<MatchKey>> edge_matches;
    for (size_t index = 0; index < playback_observations_.size(); ++index) {
      const PlaybackObservation& observation = playback_observations_[index];
      if (playback_images.find(observation.image_id) == playback_images.end() ||
          playback_images.find(observation.anchor_image_id) ==
              playback_images.end()) {
        continue;
      }
      const auto pair =
          std::minmax(observation.image_id, observation.anchor_image_id);
      edge_observations[pair].push_back(index);
      const auto endpoint =
          std::pair(observation.image_id, observation.point2D_idx);
      const auto anchor = std::pair(observation.anchor_image_id,
                                    observation.anchor_point2D_idx);
      edge_matches[pair].insert(std::minmax(endpoint, anchor));
    }
    for (auto& [image_pair, observation_indices] : edge_observations) {
      const auto match_it = edge_matches.find(image_pair);
      THROW_CHECK(match_it != edge_matches.end());
      playback_edges_.push_back({image_pair.first,
                                 image_pair.second,
                                 std::move(observation_indices),
                                 match_it->second.size()});
    }
    std::sort(playback_edges_.begin(),
              playback_edges_.end(),
              [](const PlaybackEdge& lhs, const PlaybackEdge& rhs) {
                return std::tuple(-static_cast<int64_t>(lhs.support_count),
                                  lhs.image_id1,
                                  lhs.image_id2) <
                       std::tuple(-static_cast<int64_t>(rhs.support_count),
                                  rhs.image_id1,
                                  rhs.image_id2);
              });
    playback_topology_ready_ = true;
  }

  GlobalPositioningPlaybackCapture capture;
  capture.phase = phase;
  capture.iteration = iteration;
  capture.image_ids.reserve(playback_image_ids_.size());
  capture.image_centers.reserve(3 * playback_image_ids_.size());
  for (const image_t image_id : playback_image_ids_) {
    const Image& image = reconstruction.Image(image_id);
    const Eigen::Vector3d center = CenterForImage(image);
    capture.image_ids.push_back(static_cast<uint64_t>(image_id));
    capture.image_centers.insert(capture.image_centers.end(),
                                 {center.x(), center.y(), center.z()});
  }
  capture.point3D_ids.reserve(playback_point3D_ids_.size());
  capture.points3D.reserve(3 * playback_point3D_ids_.size());
  for (const point3D_t point3D_id : playback_point3D_ids_) {
    const Eigen::Vector3d& xyz = reconstruction.Point3D(point3D_id).xyz;
    capture.point3D_ids.push_back(static_cast<uint64_t>(point3D_id));
    capture.points3D.insert(capture.points3D.end(),
                            {xyz.x(), xyz.y(), xyz.z()});
  }
  capture.lc_pairs.reserve(2 * playback_edges_.size());
  capture.lc_support_count.reserve(playback_edges_.size());
  capture.lc_raw_score.reserve(playback_edges_.size());
  std::unordered_map<size_t, double> observation_scores;
  for (const PlaybackEdge& edge : playback_edges_) {
    capture.lc_pairs.push_back(static_cast<uint64_t>(edge.image_id1));
    capture.lc_pairs.push_back(static_cast<uint64_t>(edge.image_id2));
    capture.lc_support_count.push_back(edge.support_count);
    double edge_score = 0.0;
    for (const size_t observation_index : edge.observation_indices) {
      const auto [score_it, inserted] =
          observation_scores.try_emplace(observation_index, 0.0);
      if (inserted) {
        const PlaybackObservation& observation =
            playback_observations_.at(observation_index);
        double cost = 0.0;
        if (problem_->EvaluateResidualBlock(observation.residual_block_id,
                                            false,
                                            &cost,
                                            nullptr,
                                            nullptr)) {
          const double squared_norm = 2.0 * cost;
          double rho[3] = {squared_norm, 1.0, 0.0};
          if (observation.loss_function != nullptr) {
            observation.loss_function->Evaluate(squared_norm, rho);
          }
          score_it->second =
              std::max(rho[1], 0.0) * std::sqrt(std::max(rho[0], 0.0));
          if (!std::isfinite(score_it->second)) {
            score_it->second = 0.0;
          }
        }
      }
      edge_score += score_it->second;
    }
    capture.lc_raw_score.push_back(edge_score);
  }
  options_.playback.callback(capture);
}

void GlobalPositioner::AddObservationToProblem(point3D_t point3D_id,
                                               const TrackElement& observation,
                                               bool random_initialization,
                                               Reconstruction& reconstruction,
                                               bool is_lc_observation) {
  Point3D& point3D = reconstruction.Point3D(point3D_id);
  if (!reconstruction.ExistsImage(observation.image_id)) return;

  Image& image = reconstruction.Image(observation.image_id);
  if (!image.HasPose()) return;

  Eigen::Vector3d feature_undist;
  if (observation.point2D_idx < image.features_undist.size()) {
    feature_undist = image.features_undist[observation.point2D_idx];
  } else {
    const std::optional<Eigen::Vector2d> cam_point =
        image.CameraPtr()->CamFromImg(
            image.Point2D(observation.point2D_idx).xy);
    if (!cam_point.has_value()) {
      LOG(WARNING)
          << "Ignoring feature because it failed to project: point3D_id="
          << point3D_id << ", image_id=" << observation.image_id
          << ", feature_id=" << observation.point2D_idx;
      return;
    }
    feature_undist = cam_point->homogeneous().normalized();
  }
  if (feature_undist.array().isNaN().any()) {
    LOG(WARNING)
        << "Ignoring feature because it failed to undistort: point3D_id="
        << point3D_id << ", image_id=" << observation.image_id
        << ", feature_id=" << observation.point2D_idx;
    return;
  }

  const Eigen::Vector3d cam_from_point3D_dir =
      image.CamFromWorld().rotation().inverse() * feature_undist;

  const std::string scale_key = GpObservationKey(point3D_id,
                                                 observation.image_id,
                                                 observation.point2D_idx,
                                                 is_lc_observation);
  if (!options_.debug_initial_bata_scales.empty() &&
      options_.debug_initial_bata_scales.find(scale_key) ==
          options_.debug_initial_bata_scales.end()) {
    return;
  }

  CHECK_GE(scales_.capacity(), scales_.size())
      << "Not enough capacity was reserved for the scales.";
  double& scale = scales_.emplace_back(1);

  if (!options_.generate_scales &&
      (random_initialization || options_.initialize_warm_start_scales)) {
    const Eigen::Vector3d cam_from_point3D_translation =
        point3D.xyz - CenterForImage(image);
    scale = std::max(1e-5,
                     cam_from_point3D_dir.dot(cam_from_point3D_translation) /
                         cam_from_point3D_translation.squaredNorm());
  }
  const auto debug_scale_it =
      options_.debug_initial_bata_scales.find(scale_key);
  if (debug_scale_it != options_.debug_initial_bata_scales.end()) {
    scale = debug_scale_it->second;
  }
  initial_bata_scales_[scale_key] = scale;
  bata_scale_indices_[scale_key] = scales_.size() - 1;
  double debug_dmap_scale = -1.0;

  // For calibrated and uncalibrated cameras, use different loss
  // functions
  // Down weight the uncalibrated cameras
  Camera& camera = reconstruction.Camera(image.CameraId());
  ceres::LossFunction* loss_function =
      (camera.has_prior_focal_length) ? loss_function_ptcam_calibrated_.get()
                                      : loss_function_ptcam_uncalibrated_.get();

  const bool image_is_track_anchor =
      observation.point2D_idx < image.is_track_anchor.size() &&
      image.is_track_anchor[observation.point2D_idx];
  const bool image_is_inlier =
      observation.point2D_idx < image.is_inlier.size() &&
      image.is_inlier[observation.point2D_idx];

  // Match the pyglomap GP loss cascade: per-observation labels are stored on
  // Image masks, not on COLMAP TrackElement flags.
  if (is_lc_observation && cached_loss_lc_geometry_) {
    loss_function = cached_loss_lc_geometry_.get();
  } else if (options_.use_metric_depth_constraint) {
    ceres::LossFunction* cascade = nullptr;
    if (image_is_track_anchor) {
      cascade = cached_loss_normal_geometry_trackstart_.get();
    } else if (image_is_inlier) {
      cascade = cached_loss_normal_geometry_inlier_.get();
    } else {
      cascade = cached_loss_normal_geometry_.get();
    }
    if (cascade != nullptr) {
      loss_function = cascade;
    }
  }

  // If the image is not part of a camera rig, use the standard BATA error
  if (image.IsRefInFrame()) {
    // Anisotropic per-keypoint covariance via CovarianceWeightedCostFunctor
    // when angular_stddevs is populated; otherwise bare BATA functor.
    // cov_world = R^T diag(sigma^2) R.
    ceres::CostFunction* cost_function = nullptr;
    if (observation.point2D_idx < image.angular_stddevs.size()) {
      const Eigen::Vector2d& angular_std =
          image.angular_stddevs[observation.point2D_idx];
      const double sigma_x = std::max(1e-9, angular_std[0]);
      const double sigma_y = std::max(1e-9, angular_std[1]);
      const double sigma_z = 0.5 * (sigma_x + sigma_y);
      cost_function = WeightedBATAPairwiseDirectionCostFunctor::Create(
          cam_from_point3D_dir,
          image.CamFromWorld().rotation(),
          sigma_x,
          sigma_y,
          sigma_z);
    }
    if (cost_function == nullptr) {
      cost_function =
          BATAPairwiseDirectionCostFunctor::Create(cam_from_point3D_dir);
    }

    const ceres::ResidualBlockId residual_block_id =
        problem_->AddResidualBlock(cost_function,
                                   loss_function,
                                   MutableCenterDataForImage(image),
                                   point3D.xyz.data(),
                                   &scale);
    RecordPlaybackObservation(
        observation, is_lc_observation, residual_block_id, loss_function);
    residual_order_hash_ = StableHashAppend(residual_order_hash_, 1);
    residual_order_hash_ = StableHashAppend(residual_order_hash_, point3D_id);
    residual_order_hash_ =
        StableHashAppend(residual_order_hash_, observation.image_id);
    residual_order_hash_ =
        StableHashAppend(residual_order_hash_, observation.point2D_idx);
    residual_order_hash_ =
        StableHashAppend(residual_order_hash_, is_lc_observation ? 1 : 0);
    ++diagnostics_.num_bata_residuals;
    if (is_lc_observation) {
      ++diagnostics_.num_lc_observations_used;
    } else {
      ++diagnostics_.num_regular_observations_used;
    }

    // 1-D MetricDepthError: anchors absolute scale via depth prior.
    if (options_.use_metric_depth_constraint) {
      AddMetricDepthResidual(
          point3D_id, observation, is_lc_observation, reconstruction);
      const auto dmap_scale_it = dmap_scales_.find(observation.image_id);
      if (dmap_scale_it != dmap_scales_.end()) {
        debug_dmap_scale = dmap_scale_it->second;
      }
    }
    const bool depth_valid =
        observation.point2D_idx < image.depth_prior_validity.size() &&
        image.depth_prior_validity[observation.point2D_idx];
    const std::pair<image_t, point2D_t> obs_key{observation.image_id,
                                                observation.point2D_idx};
    const bool is_runtime_depth_outlier = depth_outliers_.count(obs_key) > 0;
    GpObsDumpObservation("obs",
                         point3D_id,
                         observation,
                         is_lc_observation,
                         feature_undist,
                         cam_from_point3D_dir,
                         image,
                         point3D,
                         CenterForImage(image),
                         options_.use_metric_depth_constraint && depth_valid &&
                             !(is_runtime_depth_outlier && is_lc_observation),
                         is_runtime_depth_outlier && !is_lc_observation,
                         is_runtime_depth_outlier,
                         scale,
                         debug_dmap_scale);
  } else {
    // If the image is part of a camera rig, use the RigBATA error.

    const rig_t rig_id = image.FramePtr()->RigId();
    Rig& rig = reconstruction.Rig(rig_id);
    Rigid3d& cam_from_rig = rig.SensorFromRig(image.CameraPtr()->SensorId());

    if (!cam_from_rig.translation().hasNaN()) {
      const Eigen::Vector3d cam_from_rig_dir =
          image.CamFromWorld().rotation().inverse() *
          cam_from_rig.translation();

      ceres::CostFunction* cost_function =
          RigBATAPairwiseDirectionConstantRigCostFunctor::Create(
              cam_from_point3D_dir, cam_from_rig_dir);

      const ceres::ResidualBlockId residual_block_id =
          problem_->AddResidualBlock(cost_function,
                                     loss_function,
                                     point3D.xyz.data(),
                                     MutableCenterDataForImage(image),
                                     &scale);
      RecordPlaybackObservation(
          observation, is_lc_observation, residual_block_id, loss_function);
      residual_order_hash_ = StableHashAppend(residual_order_hash_, 1);
      residual_order_hash_ = StableHashAppend(residual_order_hash_, point3D_id);
      residual_order_hash_ =
          StableHashAppend(residual_order_hash_, observation.image_id);
      residual_order_hash_ =
          StableHashAppend(residual_order_hash_, observation.point2D_idx);
      residual_order_hash_ =
          StableHashAppend(residual_order_hash_, is_lc_observation ? 1 : 0);
      ++diagnostics_.num_bata_residuals;
      if (is_lc_observation) {
        ++diagnostics_.num_lc_observations_used;
      } else {
        ++diagnostics_.num_regular_observations_used;
      }
    } else {
      // If the cam_from_rig contains nan values, it needs to be re-estimated.
      // Initialize cams_in_rig_ if not already done.
      const sensor_t sensor_id = image.CameraPtr()->SensorId();
      if (cams_in_rig_.find(sensor_id) == cams_in_rig_.end()) {
        // Will be initialized to random values in ParameterizeVariables().
        cams_in_rig_[sensor_id] = Eigen::Vector3d::Zero();
      }

      ceres::CostFunction* cost_function =
          RigBATAPairwiseDirectionCostFunctor::Create(
              cam_from_point3D_dir,
              image.FramePtr()->RigFromWorld().rotation());

      const ceres::ResidualBlockId residual_block_id =
          problem_->AddResidualBlock(cost_function,
                                     loss_function,
                                     point3D.xyz.data(),
                                     MutableCenterDataForImage(image),
                                     cams_in_rig_[sensor_id].data(),
                                     &scale);
      RecordPlaybackObservation(
          observation, is_lc_observation, residual_block_id, loss_function);
      residual_order_hash_ = StableHashAppend(residual_order_hash_, 1);
      residual_order_hash_ = StableHashAppend(residual_order_hash_, point3D_id);
      residual_order_hash_ =
          StableHashAppend(residual_order_hash_, observation.image_id);
      residual_order_hash_ =
          StableHashAppend(residual_order_hash_, observation.point2D_idx);
      residual_order_hash_ =
          StableHashAppend(residual_order_hash_, is_lc_observation ? 1 : 0);
      ++diagnostics_.num_bata_residuals;
      if (is_lc_observation) {
        ++diagnostics_.num_lc_observations_used;
      } else {
        ++diagnostics_.num_regular_observations_used;
      }
    }
  }

  problem_->SetParameterLowerBound(&scale, 0, 1e-5);
}

void GlobalPositioner::AddMetricDepthResidual(point3D_t point3D_id,
                                              const TrackElement& observation,
                                              bool is_lc_observation,
                                              Reconstruction& reconstruction) {
  if (!reconstruction.ExistsImage(observation.image_id)) return;
  const Image& image = reconstruction.Image(observation.image_id);

  if (observation.point2D_idx >= image.depth_prior_validity.size() ||
      !image.depth_prior_validity[observation.point2D_idx]) {
    return;
  }
  THROW_CHECK_LT(observation.point2D_idx, image.depth_priors.size());
  THROW_CHECK_LT(observation.point2D_idx, image.depth_prior_stddevs.size());

  const double depth_prior = image.depth_priors[observation.point2D_idx];
  const double depth_sigma = image.depth_prior_stddevs[observation.point2D_idx];

  if (depth_sigma <= 1e-9) return;

  // Lazy-insert dmap_scales_ on first valid observation per image.
  if (dmap_scales_.find(observation.image_id) == dmap_scales_.end()) {
    double init_value = options_.use_log_scale_for_depth_map_scales ? 0.0 : 1.0;
    if (options_.initial_dmap_scales.has_value()) {
      const auto& init_map = *options_.initial_dmap_scales;
      auto it = init_map.find(observation.image_id);
      if (it != init_map.end()) {
        // Convert caller-supplied linear value to log if needed.
        init_value = options_.use_log_scale_for_depth_map_scales
                         ? std::log(std::max(it->second, 1e-9))
                         : it->second;
      }
    }
    dmap_scales_[observation.image_id] = init_value;
    dmap_scale_observation_counts_[observation.image_id] = 0;
  }
  dmap_scale_observation_counts_[observation.image_id]++;

  ceres::CostFunction* metric_depth_cost =
      MetricDepthError::Create(image.CamFromWorld().rotation(),
                               depth_prior,
                               depth_sigma,
                               options_.use_log_scale_for_depth_map_scales,
                               options_.metric_depth_residual_type,
                               options_.zero_residual_behind,
                               options_.log_linear_threshold);

  if (metric_depth_cost == nullptr) return;

  // Outlier dual routing:
  //   depth_outliers_ = runtime geometry-driven filter
  //     (Nσ log-residual, N = filter_depth_outlier_sigma),
  //     populated by FilterDepthOutliers between BA iterations.
  //   TrackElement::is_depth_outlier = external annotation
  //     (MDRP/boundary heuristic), populated by Python pipeline
  //     pre-solve.
  //   depth_outliers_ is checked first and is strictly more
  //   aggressive: LC observations skip the depth residual
  //   entirely, non-LC get a soft fallback loss.
  //
  // 5-way depth-loss cascade:
  //   pre-pass outlier  -> soft fallback (skip on LC)
  //   is_lc             -> cached_loss_lc_depth_
  //   TrackElement::is_track_anchor  -> cached_loss_normal_depth_trackstart_
  //   TrackElement::is_inlier        -> cached_loss_normal_depth_inlier_
  //   TrackElement::is_depth_outlier -> cached_loss_normal_depth_outlier_
  //   else              -> cached_loss_normal_depth_
  ceres::LossFunction* depth_loss = nullptr;
  const std::pair<image_t, point2D_t> obs_key{observation.image_id,
                                              observation.point2D_idx};
  const bool image_is_track_anchor =
      observation.point2D_idx < image.is_track_anchor.size() &&
      image.is_track_anchor[observation.point2D_idx];
  const bool image_is_inlier =
      observation.point2D_idx < image.is_inlier.size() &&
      image.is_inlier[observation.point2D_idx];
  const bool image_is_depth_outlier =
      observation.point2D_idx < image.is_depth_outlier.size() &&
      image.is_depth_outlier[observation.point2D_idx];
  if (depth_outliers_.count(obs_key) > 0) {
    if (is_lc_observation) {
      // LC outlier: skip depth residual entirely.
      delete metric_depth_cost;
      return;
    }
    // Non-LC outlier: soft fallback (HuberLoss(1)).
    if (!soft_outlier_fallback_loss_) {
      soft_outlier_fallback_loss_ =
          options_.loss_soft_outlier_fallback.CreateLossFunction();
    }
    depth_loss = soft_outlier_fallback_loss_.get();
  } else if (is_lc_observation) {
    depth_loss = cached_loss_lc_depth_.get();
  } else if (image_is_track_anchor) {
    depth_loss = cached_loss_normal_depth_trackstart_.get();
  } else if (image_is_inlier) {
    depth_loss = cached_loss_normal_depth_inlier_.get();
  } else if (image_is_depth_outlier) {
    depth_loss = cached_loss_normal_depth_outlier_.get();
  } else {
    depth_loss = cached_loss_normal_depth_.get();
  }

  Point3D& point3D = reconstruction.Point3D(point3D_id);
  problem_->AddResidualBlock(metric_depth_cost,
                             depth_loss,
                             MutableCenterDataForImage(image),
                             point3D.xyz.data(),
                             &dmap_scales_[observation.image_id]);
  if (!options_.use_log_scale_for_depth_map_scales) {
    problem_->SetParameterLowerBound(
        &dmap_scales_[observation.image_id], 0, 1e-5);
  }
  residual_order_hash_ = StableHashAppend(residual_order_hash_, 2);
  residual_order_hash_ = StableHashAppend(residual_order_hash_, point3D_id);
  residual_order_hash_ =
      StableHashAppend(residual_order_hash_, observation.image_id);
  residual_order_hash_ =
      StableHashAppend(residual_order_hash_, observation.point2D_idx);
  residual_order_hash_ =
      StableHashAppend(residual_order_hash_, is_lc_observation ? 1 : 0);
  ++diagnostics_.num_metric_depth_residuals;
}

void GlobalPositioner::AddCamerasAndPointsToParameterGroups(
    Reconstruction& reconstruction) {
  // Create a custom ordering for Schur-based problems.
  options_.solver_options.linear_solver_ordering.reset(
      new ceres::ParameterBlockOrdering);
  ceres::ParameterBlockOrdering* parameter_ordering =
      options_.solver_options.linear_solver_ordering.get();

  const std::string ordering_mode = EnvValue("MPSFM_GP_ORDERING_MODE");
  const bool split_scales = ordering_mode == "per_block_all";
  const bool split_points = split_scales ||
                            ordering_mode == "per_block_non_scales" ||
                            ordering_mode == "per_block_points";
  const bool split_camera_like = split_scales ||
                                 ordering_mode == "per_block_non_scales" ||
                                 ordering_mode == "per_block_camera_like";
  const bool legacy_unordered = ordering_mode == "legacy_unordered";

  // Add scale parameters first. In the default mode, keep legacy coarse
  // grouping. In strict diagnostic mode, use one group per scale block.
  int group_id = 0;
  for (double& scale : scales_) {
    parameter_ordering->AddElementToGroup(&scale, group_id);
    if (split_scales) {
      ++group_id;
    }
  }
  if (!split_scales) {
    ++group_id;
  }

  std::vector<point3D_t> point3D_ids;
  if (reconstruction.NumPoints3D() > 0) {
    point3D_ids.reserve(reconstruction.NumPoints3D());
    for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
      point3D_ids.push_back(point3D_id);
    }
    if (!legacy_unordered) {
      std::sort(point3D_ids.begin(), point3D_ids.end());
    }
    for (const point3D_t point3D_id : point3D_ids) {
      const Point3D& point3D = reconstruction.Point3D(point3D_id);
      if (problem_->HasParameterBlock(point3D.xyz.data())) {
        parameter_ordering->AddElementToGroup(
            reconstruction.Point3D(point3D_id).xyz.data(), group_id);
        if (split_points) {
          ++group_id;
        }
      }
    }
    if (!split_points) {
      ++group_id;
    }
  }

  std::vector<frame_t> frame_ids;
  std::vector<image_t> image_ids;
  if (UseImageCenterBlocks()) {
    image_ids.reserve(image_centers_.size());
    for (const auto& [image_id, center] : image_centers_) {
      image_ids.push_back(image_id);
    }
    if (!legacy_unordered) {
      std::sort(image_ids.begin(), image_ids.end());
    }
    for (const image_t image_id : image_ids) {
      Eigen::Vector3d& center = image_centers_.at(image_id);
      if (problem_->HasParameterBlock(center.data())) {
        parameter_ordering->AddElementToGroup(center.data(), group_id);
        if (split_camera_like) {
          ++group_id;
        }
      }
    }
  } else if (UseFrameInplaceCenterBlocks()) {
    frame_ids.reserve(frame_centers_.size());
    for (const auto& [frame_id, center] : frame_centers_) {
      frame_ids.push_back(frame_id);
    }
    if (!legacy_unordered) {
      std::sort(frame_ids.begin(), frame_ids.end());
    }
    for (const frame_t frame_id : frame_ids) {
      double* center_data =
          reconstruction.Frame(frame_id).RigFromWorld().translation().data();
      if (problem_->HasParameterBlock(center_data)) {
        parameter_ordering->AddElementToGroup(center_data, group_id);
        if (split_camera_like) {
          ++group_id;
        }
      }
    }
  } else {
    frame_ids.reserve(frame_centers_.size());
    for (const auto& [frame_id, center] : frame_centers_) {
      frame_ids.push_back(frame_id);
    }
    if (!legacy_unordered) {
      std::sort(frame_ids.begin(), frame_ids.end());
    }
    for (const frame_t frame_id : frame_ids) {
      Eigen::Vector3d& center = frame_centers_.at(frame_id);
      if (problem_->HasParameterBlock(center.data())) {
        parameter_ordering->AddElementToGroup(center.data(), group_id);
        if (split_camera_like) {
          ++group_id;
        }
      }
    }
  }

  // Add the cam_in_rig to be estimated into the parameter group
  std::vector<sensor_t> sensor_ids;
  sensor_ids.reserve(cams_in_rig_.size());
  for (const auto& [sensor_id, center] : cams_in_rig_) {
    sensor_ids.push_back(sensor_id);
  }
  if (!legacy_unordered) {
    std::sort(sensor_ids.begin(), sensor_ids.end());
  }
  for (const sensor_t sensor_id : sensor_ids) {
    Eigen::Vector3d& center = cams_in_rig_.at(sensor_id);
    if (problem_->HasParameterBlock(center.data())) {
      parameter_ordering->AddElementToGroup(center.data(), group_id);
      if (split_camera_like) {
        ++group_id;
      }
    }
  }

  // Legacy GLOMAP puts per-image depth-map scales in the same Schur group as
  // camera centers. Keep that ordering for native GP parity.
  for (const image_t image_id : SortedKeys(dmap_scales_)) {
    double& scale = dmap_scales_.at(image_id);
    if (problem_->HasParameterBlock(&scale)) {
      parameter_ordering->AddElementToGroup(&scale, group_id);
      if (split_camera_like) {
        ++group_id;
      }
    }
  }

  if (const char* log_path = std::getenv("MPSFM_GP_DETERMINISM_LOG")) {
    std::ofstream log(log_path, std::ios::app);
    if (log) {
      constexpr uint64_t kFnvOffset = 1469598103934665603ULL;
      uint64_t point_order_hash = kFnvOffset;
      uint64_t frame_order_hash = kFnvOffset;
      uint64_t sensor_order_hash = kFnvOffset;
      uint64_t dmap_order_hash = kFnvOffset;
      for (const point3D_t point3D_id : point3D_ids) {
        point_order_hash = StableHashAppend(point_order_hash, point3D_id);
      }
      for (const frame_t frame_id : frame_ids) {
        frame_order_hash = StableHashAppend(frame_order_hash, frame_id);
      }
      for (const sensor_t sensor_id : sensor_ids) {
        sensor_order_hash = StableHashAppend(
            sensor_order_hash, static_cast<uint64_t>(sensor_id.id));
        sensor_order_hash = StableHashAppend(
            sensor_order_hash, static_cast<uint64_t>(sensor_id.type));
      }
      for (const image_t image_id : SortedKeys(dmap_scales_)) {
        dmap_order_hash = StableHashAppend(dmap_order_hash, image_id);
      }
      log << "gp_parameter_ordering"
          << " mode="
          << (ordering_mode.empty() ? "legacy_coarse" : ordering_mode)
          << " scales=" << scales_.size() << " points=" << point3D_ids.size()
          << " frame_centers=" << frame_ids.size()
          << " image_centers=" << image_ids.size()
          << " cams_in_rig=" << sensor_ids.size()
          << " dmap_scales=" << dmap_scales_.size()
          << " point_order_hash=" << point_order_hash
          << " frame_order_hash=" << frame_order_hash
          << " sensor_order_hash=" << sensor_order_hash
          << " dmap_order_hash=" << dmap_order_hash << "\n";
    }
  }
}

void GlobalPositioner::ParameterizeVariables(Reconstruction& reconstruction) {
  // For the global positioning, do not set any camera to be constant for easier
  // convergence

  // Initialize cams_in_rig_ with random values if optimizing positions.
  if (options_.optimize_positions) {
    for (auto& [sensor_id, center] : cams_in_rig_) {
      if (problem_->HasParameterBlock(center.data())) {
        center = RandVector3d(-1, 1);
      }
    }
  }

  // If not optimizing positions, set frame centers to be constant.
  if (!options_.optimize_positions) {
    if (UseImageCenterBlocks()) {
      for (auto& [image_id, center] : image_centers_) {
        if (problem_->HasParameterBlock(center.data())) {
          problem_->SetParameterBlockConstant(center.data());
        }
      }
    } else if (UseFrameInplaceCenterBlocks()) {
      for (const auto& [frame_id, unused_center] : frame_centers_) {
        double* center_data =
            reconstruction.Frame(frame_id).RigFromWorld().translation().data();
        if (problem_->HasParameterBlock(center_data)) {
          problem_->SetParameterBlockConstant(center_data);
        }
      }
    } else {
      for (auto& [frame_id, center] : frame_centers_) {
        if (problem_->HasParameterBlock(center.data())) {
          problem_->SetParameterBlockConstant(center.data());
        }
      }
    }
  }

  // If do not optimize the rotations, set the camera rotations to be constant
  if (!options_.optimize_points) {
    for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
      if (problem_->HasParameterBlock(point3D.xyz.data())) {
        problem_->SetParameterBlockConstant(
            reconstruction.Point3D(point3D_id).xyz.data());
      }
    }
  }

  // If do not optimize the scales, set the scales to be constant
  if (!options_.optimize_scales) {
    for (double& scale : scales_) {
      if (problem_->HasParameterBlock(&scale)) {
        problem_->SetParameterBlockConstant(&scale);
      }
    }
  }

  // Lower-bound dmap_scales_ in linear space (no bound needed in log space).
  if (!options_.use_log_scale_for_depth_map_scales) {
    for (auto& [image_id, scale] : dmap_scales_) {
      if (problem_->HasParameterBlock(&scale)) {
        problem_->SetParameterLowerBound(&scale, 0, 1e-5);
      }
    }
  }
  // Pin first scale to remove gauge ambiguity. Skip when metric-depth is
  // active (depth priors + scale priors already anchor the gauge).
  if (!options_.use_metric_depth_constraint) {
    for (double& scale : scales_) {
      if (problem_->HasParameterBlock(&scale)) {
        problem_->SetParameterBlockConstant(&scale);
        break;
      }
    }
  }

#ifdef COLMAP_CUDA_ENABLED
  bool cuda_solver_enabled = false;

#if (CERES_VERSION_MAJOR >= 3 ||                                \
     (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 2)) && \
    !defined(CERES_NO_CUDA)
  if (options_.use_gpu &&
      reconstruction.NumImages() >=
          static_cast<size_t>(options_.min_num_images_gpu_solver)) {
    cuda_solver_enabled = true;
    options_.solver_options.dense_linear_algebra_library_type = ceres::CUDA;
  }
#else
  if (options_.use_gpu) {
    LOG_FIRST_N(WARNING, 1)
        << "Requested to use GPU for bundle adjustment, but Ceres was "
           "compiled without CUDA support. Falling back to CPU-based dense "
           "solvers.";
  }
#endif

#if (CERES_VERSION_MAJOR >= 3 ||                                \
     (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 3)) && \
    !defined(CERES_NO_CUDSS)
  if (options_.use_gpu &&
      reconstruction.NumImages() >=
          static_cast<size_t>(options_.min_num_images_gpu_solver)) {
    cuda_solver_enabled = true;
    options_.solver_options.sparse_linear_algebra_library_type =
        ceres::CUDA_SPARSE;
  }
#else
  if (options_.use_gpu) {
    LOG_FIRST_N(WARNING, 1)
        << "Requested to use GPU for bundle adjustment, but Ceres was "
           "compiled without cuDSS support. Falling back to CPU-based sparse "
           "solvers.";
  }
#endif

  if (cuda_solver_enabled) {
    const std::vector<int> gpu_indices = CSVToVector<int>(options_.gpu_index);
    THROW_CHECK_GT(gpu_indices.size(), 0);
    SetBestCudaDevice(gpu_indices[0]);
  }
#else
  if (options_.use_gpu) {
    LOG_FIRST_N(WARNING, 1)
        << "Requested to use GPU for bundle adjustment, but COLMAP was "
           "compiled without CUDA support. Falling back to CPU-based "
           "solvers.";
  }
#endif  // COLMAP_CUDA_ENABLED

  // Set up the options for the solver
  // Do not use iterative solvers, for its suboptimal performance.
  if (reconstruction.NumPoints3D() > 0) {
    options_.solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
    options_.solver_options.preconditioner_type = ceres::CLUSTER_TRIDIAGONAL;
  } else {
    options_.solver_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options_.solver_options.preconditioner_type = ceres::JACOBI;
  }
}

bool GlobalPositioner::UseImageCenterBlocks() const {
  return EnvValue("MPSFM_GP_CAMERA_BLOCK_MODE") == "image";
}

bool GlobalPositioner::UseFrameInplaceCenterBlocks() const {
  return EnvValue("MPSFM_GP_CAMERA_BLOCK_MODE") == "frame_inplace";
}

Eigen::Vector3d& GlobalPositioner::MutableCenterForImage(const Image& image) {
  if (UseImageCenterBlocks()) {
    return image_centers_.at(image.ImageId());
  }
  return frame_centers_.at(image.FrameId());
}

double* GlobalPositioner::MutableCenterDataForImage(const Image& image) {
  if (UseFrameInplaceCenterBlocks()) {
    return image.FramePtr()->RigFromWorld().translation().data();
  }
  return MutableCenterForImage(image).data();
}

Eigen::Vector3d GlobalPositioner::CenterForImage(const Image& image) const {
  if (UseImageCenterBlocks()) {
    return image_centers_.at(image.ImageId());
  }
  if (UseFrameInplaceCenterBlocks()) {
    return image.FramePtr()->RigFromWorld().translation();
  }
  return frame_centers_.at(image.FrameId());
}

void GlobalPositioner::ConvertBackResults(Reconstruction& reconstruction) {
  if (UseImageCenterBlocks()) {
    for (const auto& [image_id, center] : image_centers_) {
      const Image& image = reconstruction.Image(image_id);
      Rigid3d& rig_from_world =
          reconstruction.Frame(image.FrameId()).RigFromWorld();
      rig_from_world.translation() = -(rig_from_world.rotation() * center);
    }
  }

  if (UseFrameInplaceCenterBlocks()) {
    for (const auto& [frame_id, unused_center] : frame_centers_) {
      Rigid3d& rig_from_world = reconstruction.Frame(frame_id).RigFromWorld();
      rig_from_world.translation() =
          -(rig_from_world.rotation() * rig_from_world.translation());
    }
  }

  // Convert optimized frame centers back to rig_from_world translations.
  if (!UseImageCenterBlocks() && !UseFrameInplaceCenterBlocks()) {
    for (const auto& [frame_id, center] : frame_centers_) {
      Rigid3d& rig_from_world = reconstruction.Frame(frame_id).RigFromWorld();
      rig_from_world.translation() = -(rig_from_world.rotation() * center);
    }
  }

  // Convert optimized cam_in_rig back to sensor_from_rig translations.
  for (const auto& [sensor_id, center] : cams_in_rig_) {
    // Find the rig containing this sensor.
    for (const auto& [rig_id, rig] : reconstruction.Rigs()) {
      if (!rig.HasSensor(sensor_id)) {
        continue;
      }
      Rigid3d& sensor_from_rig =
          reconstruction.Rig(rig_id).SensorFromRig(sensor_id);
      sensor_from_rig.translation() = -(sensor_from_rig.rotation() * center);
      break;
    }
  }
}

void GlobalPositioner::FilterDepthOutliers(
    const Reconstruction& reconstruction) {
  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    // Regular observations
    for (const auto& observation : point3D.track.Elements()) {
      if (!reconstruction.ExistsImage(observation.image_id)) continue;
      const Image& image = reconstruction.Image(observation.image_id);
      if (!image.HasPose()) continue;
      if (DepthOutlierFlag(image,
                           observation.point2D_idx,
                           point3D.xyz,
                           options_.use_log_scale_for_depth_map_scales,
                           dmap_scales_,
                           observation.image_id,
                           options_.filter_depth_outlier_sigma)) {
        depth_outliers_.insert({observation.image_id, observation.point2D_idx});
      }
    }
    if (options_.use_lc_observations) {
      for (const auto& observation : point3D.track.lc_elements) {
        if (!reconstruction.ExistsImage(observation.image_id)) continue;
        const Image& image = reconstruction.Image(observation.image_id);
        if (!image.HasPose()) continue;
        if (DepthOutlierFlag(image,
                             observation.point2D_idx,
                             point3D.xyz,
                             options_.use_log_scale_for_depth_map_scales,
                             dmap_scales_,
                             observation.image_id,
                             options_.filter_depth_outlier_sigma)) {
          depth_outliers_.insert(
              {observation.image_id, observation.point2D_idx});
        }
      }
    }
  }
  VLOG(2) << "FilterDepthOutliers: flagged " << depth_outliers_.size()
          << " observations as depth outliers.";
}

void GlobalPositioner::InitializeDepthMapScalesFromObservations(
    const Reconstruction& reconstruction) {
  // Per-image scale estimates: scale = z_est / depth_prior.
  std::map<image_t, std::vector<double>> image_scale_estimates;

  auto consume_observation = [&](image_t image_id,
                                 point2D_t feature_id,
                                 const Eigen::Vector3d& point_world) {
    if (!reconstruction.ExistsImage(image_id)) return;
    const Image& image = reconstruction.Image(image_id);
    if (!image.HasPose()) return;

    if (feature_id >= image.depth_prior_validity.size() ||
        !image.depth_prior_validity[feature_id]) {
      return;
    }
    if (feature_id >= image.depth_priors.size()) return;

    const double depth_prior = image.depth_priors[feature_id];
    if (depth_prior <= 1e-6) return;

    // z_est = (cam_from_world * X_world)[2]
    const Eigen::Vector3d point_cam = image.CamFromWorld() * point_world;
    const double z_est = point_cam[2];
    if (z_est <= 1e-6) return;

    const double scale_estimate = z_est / depth_prior;
    if (scale_estimate > 1e-6 && scale_estimate < 1e6) {
      image_scale_estimates[image_id].push_back(scale_estimate);
    }
  };

  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    for (const auto& observation : point3D.track.Elements()) {
      consume_observation(
          observation.image_id, observation.point2D_idx, point3D.xyz);
    }
    if (options_.use_lc_observations) {
      for (const auto& observation : point3D.track.lc_elements) {
        consume_observation(
            observation.image_id, observation.point2D_idx, point3D.xyz);
      }
    }
  }

  for (auto& [image_id, scale_estimates] : image_scale_estimates) {
    if (scale_estimates.empty()) continue;

    std::sort(scale_estimates.begin(), scale_estimates.end());
    const double median_scale = scale_estimates[scale_estimates.size() / 2];

    const double initial_value = options_.use_log_scale_for_depth_map_scales
                                     ? std::log(median_scale)
                                     : median_scale;

    dmap_scales_[image_id] = initial_value;
    dmap_scale_observation_counts_[image_id] = 0;
  }

  VLOG(2) << "InitializeDepthMapScalesFromObservations: seeded "
          << dmap_scales_.size() << " image scales from observations.";
}

std::unordered_map<std::string, double> GlobalPositioner::GetFinalBataScales()
    const {
  std::unordered_map<std::string, double> out;
  out.reserve(bata_scale_indices_.size());
  for (const auto& [key, index] : bata_scale_indices_) {
    if (index < scales_.size()) {
      out.emplace(key, scales_[index]);
    }
  }
  return out;
}

bool RunGlobalPositioning(const GlobalPositionerOptions& options,
                          const PoseGraph& pose_graph,
                          Reconstruction& reconstruction) {
  GlobalPositioner positioner(options);
  return positioner.Solve(pose_graph, reconstruction);
}

}  // namespace colmap
