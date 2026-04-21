#include "colmap/estimators/global_positioning.h"

#include "colmap/estimators/cost_functions/bata_pairwise_direction_error.h"
#include "colmap/estimators/cost_functions/bata_pairwise_direction_error_with_rotation.h"
#include "colmap/estimators/cost_functions/log_scale_prior_error.h"
#include "colmap/estimators/cost_functions/mahalanobis_bata_directional_error.h"
#include "colmap/estimators/cost_functions/mahalanobis_bata_directional_error_with_rotation.h"
#include "colmap/estimators/cost_functions/manifold.h"
#include "colmap/estimators/cost_functions/metric_depth_error.h"
#include "colmap/estimators/cost_functions/metric_depth_error_with_rotation.h"
#include "colmap/estimators/cost_functions/motion_averaging.h"
#include "colmap/estimators/cost_functions/relative_translation_error.h"
#include "colmap/estimators/cost_functions/rotation_prior_error.h"
#include "colmap/estimators/cost_functions/scale_prior_error.h"
#include "colmap/estimators/cost_functions/weighted_bata_directional_error.h"
#include "colmap/estimators/cost_functions/weighted_bata_directional_error_with_rotation.h"
#include "colmap/math/random.h"
#include "colmap/util/cuda.h"
#include "colmap/util/misc.h"
#include "colmap/util/threading.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <random>

namespace colmap {
namespace {

Eigen::Vector3d RandVector3d(double low, double high) {
  return Eigen::Vector3d(RandomUniformReal(low, high),
                         RandomUniformReal(low, high),
                         RandomUniformReal(low, high));
}

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

  LOG(INFO) << "Setting up the global positioner problem";

  // Capture initial rotations for regularization.
  if (options_.optimize_rotations) {
    initial_rotations_.clear();
    for (const auto& [image_id, image] : reconstruction.Images()) {
      if (!image.HasPose()) continue;
      initial_rotations_[image_id] = image.CamFromWorld().rotation();
    }
    LOG(INFO) << "Captured " << initial_rotations_.size()
              << " initial rotations for regularization";
  }

  // Reset counters.
  geometry_only_constraints_ = 0;
  depth_constraints_ = 0;
  normal_residuals_ = 0;
  lc_residuals_ = 0;
  normal_inlier_residuals_ = 0;
  normal_outlier_residuals_ = 0;
  mdrp_depth_outlier_residuals_ = 0;
  track_anchor_residuals_ = 0;
  track_anchor_depth_residuals_ = 0;
  mahalanobis_bata_residuals_ = 0;
  diagonal_bata_residuals_ = 0;
  unweighted_bata_residuals_ = 0;
  relative_pose_constraints_ = 0;
  rotation_prior_constraints_ = 0;

  // Setup the problem.
  SetupProblem(pose_graph, reconstruction);

  // Initialize camera translations to be random.
  // Also, convert the camera pose translation to be the camera center.
  InitializeRandomPositions(pose_graph, reconstruction);

  // Consume initial_dmap_scales seeds.
  if (options_.use_depth_priors) {
    dmap_scales_.clear();
    dmap_scale_observation_counts_.clear();
    scale_prior_losses_.clear();
    depth_outliers_.clear();
    for (const auto& [image_id, scale] : options_.initial_dmap_scales) {
      if (scale <= 0.0) {
        LOG(WARNING) << "Invalid initial scale " << scale << " for image "
                     << image_id << ", skipping";
        continue;
      }
      double init_val = options_.use_log_scale_for_depth_map_scales
                            ? std::log(scale)
                            : scale;
      dmap_scales_[image_id] = init_val;
      dmap_scale_observation_counts_[image_id] = 0;
    }
    // Auto-init from 3D points when use_init=true and no seeds given.
    if (options_.use_init && options_.initial_dmap_scales.empty()) {
      LOG(INFO) << "Auto-initializing depth map scales from observed 3D points";
      InitializeDepthMapScalesFromObservations(reconstruction);
    }
  }

  // Add the point to camera constraints to the problem.
  if (!options_.debug_only_relative_pose) {
    AddPointToCameraConstraints(reconstruction);
  } else {
    LOG(INFO) << "Skipping point-to-camera constraints (debug_only_relative_pose)";
  }

  // Add relative-pose constraints.
  if (options_.use_relative_pose_constraints &&
      !options_.relative_pose_pair_ids.empty()) {
    AddRelativePoseConstraints(pose_graph, reconstruction);
  }

  // Add rotation priors.
  if (options_.optimize_rotations && options_.regularize_rotations) {
    AddRotationPriors(reconstruction);
  }

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

  // Register iteration callback.
  std::unique_ptr<ceres::IterationCallback> ceres_cb;
  if (options_.iteration_callback != nullptr) {
    auto& images_mut =
        const_cast<std::unordered_map<image_t, Image>&>(
            reconstruction.Images());
    ceres_cb = std::make_unique<glomap_ext::SfMIterationCallback>(
        options_.iteration_callback,
        images_mut,
        reconstruction,
        pose_graph,
        /*gp_mode=*/true);
    options_.solver_options.callbacks.push_back(ceres_cb.get());
    options_.solver_options.update_state_every_iteration = true;
  }

  ceres::Solve(options_.solver_options, problem_.get(), &summary);

  // Remove callback to avoid dangling pointer.
  if (ceres_cb) {
    auto& cbs = options_.solver_options.callbacks;
    cbs.erase(std::remove(cbs.begin(), cbs.end(), ceres_cb.get()), cbs.end());
    options_.solver_options.update_state_every_iteration = false;
  }

  if (VLOG_IS_ON(2)) {
    LOG(INFO) << summary.FullReport();
  } else {
    LOG(INFO) << summary.BriefReport();
  }

  // Log extended statistics.
  LOG(INFO) << "Constraint statistics: " << geometry_only_constraints_
            << " geometry-only, " << depth_constraints_ << " depth-based";
  if (options_.use_depth_priors) {
    LOG(INFO) << "Depth map scales count: " << dmap_scales_.size();
    LOG(INFO) << "  MDRP depth outlier residuals: " << mdrp_depth_outlier_residuals_;
    LOG(INFO) << "  Track anchor geometry residuals: " << track_anchor_residuals_;
    LOG(INFO) << "  Track anchor depth residuals: " << track_anchor_depth_residuals_;
    LOG(INFO) << "  BATA: " << mahalanobis_bata_residuals_ << " Mahalanobis, "
              << diagonal_bata_residuals_ << " diagonal, "
              << unweighted_bata_residuals_ << " unweighted";
  }

  ConvertBackResults(reconstruction);
  return summary.IsSolutionUsable();
}

void GlobalPositioner::SetupProblem(const PoseGraph& pose_graph,
                                    const Reconstruction& reconstruction) {
  ceres::Problem::Options problem_options;
  problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  problem_ = std::make_unique<ceres::Problem>(problem_options);
  loss_function_ = options_.CreateLossFunction();

  // Clear temporary storage from previous runs.
  frame_centers_.clear();
  cams_in_rig_.clear();

  // Allocate enough memory for the scales. One for each residual.
  // Due to possibly invalid tracks, the actual number of residuals may be
  // smaller.
  scales_.clear();
  size_t total_observations = 0;
  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    total_observations += point3D.track.Length();
    if (options_.use_lc_observations) {
      total_observations += point3D.track.LcLength();
    }
  }
  scales_.reserve(total_observations);
}

void GlobalPositioner::InitializeRandomPositions(
    const PoseGraph& pose_graph, Reconstruction& reconstruction) {
  std::unordered_set<frame_t> constrained_positions;
  constrained_positions.reserve(reconstruction.NumFrames());
  for (const auto& [pair_id, edge] : pose_graph.ValidEdges()) {
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    // filter by image_ids_to_optimize if non-empty.
    if (!options_.image_ids_to_optimize.empty()) {
      bool in1 = options_.image_ids_to_optimize.count(image_id1) > 0;
      bool in2 = options_.image_ids_to_optimize.count(image_id2) > 0;
      if (!in1 && !in2) continue;
    }
    constrained_positions.insert(reconstruction.Image(image_id1).FrameId());
    constrained_positions.insert(reconstruction.Image(image_id2).FrameId());
  }

  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    if (point3D.track.Length() <
        static_cast<size_t>(options_.min_num_view_per_track)) {
      continue;
    }
    for (const auto& observation : point3D.track.Elements()) {
      if (!options_.image_ids_to_optimize.empty() &&
          !options_.image_ids_to_optimize.count(observation.image_id))
        continue;
      THROW_CHECK(reconstruction.ExistsImage(observation.image_id));
      const Image& image = reconstruction.Image(observation.image_id);
      if (!image.HasPose()) continue;
      constrained_positions.insert(image.FrameId());
    }
  }

  // Initialize frame centers in temporary storage.
  // The reconstruction poses remain in cam_from_world convention.
  for (const auto& [frame_id, frame] : reconstruction.Frames()) {
    if (constrained_positions.find(frame_id) == constrained_positions.end()) {
      continue;
    }
    if (options_.use_init) {
      // Keep existing center (from prior GP run or external init).
      frame_centers_[frame_id] = frame.RigFromWorld().TgtOriginInSrc();
    } else if (options_.generate_random_positions && options_.optimize_positions) {
      // Use options_.random_init_scale instead of hardcoded 100.0.
      frame_centers_[frame_id] = options_.random_init_scale * RandVector3d(-1, 1);
    } else {
      frame_centers_[frame_id] = frame.RigFromWorld().TgtOriginInSrc();
    }
  }

  VLOG(2) << "Constrained positions: " << constrained_positions.size();
}

void GlobalPositioner::AddPointToCameraConstraints(
    Reconstruction& reconstruction) {
  LOG(INFO) << "AddPointToCameraConstraints: "
            << reconstruction.NumPoints3D() << " tracks";

  if (options_.use_fork_loss_dispatch) {
    // Build cached loss functions.
    cached_loss_normal_geometry_ = options_.loss_normal_geometry.Create();
    cached_loss_normal_depth_ = options_.loss_normal_depth.Create();
    cached_loss_lc_geometry_ = options_.loss_lc_geometry.Create();
    cached_loss_lc_depth_ = options_.loss_lc_depth.Create();
    cached_loss_normal_geometry_inlier_ = options_.loss_normal_geometry_inlier.Create();
    cached_loss_normal_depth_inlier_ = options_.loss_normal_depth_inlier.Create();
    cached_loss_scale_prior_ = options_.loss_scale_prior.Create();
    cached_loss_normal_geometry_trackstart_ = options_.loss_normal_geometry_trackstart.Create();
    cached_loss_normal_depth_trackstart_ = options_.loss_normal_depth_trackstart.Create();
    cached_loss_normal_depth_outlier_ = options_.loss_normal_depth_outlier.Create();
    // Fixed Huber(1,1) outlier depth — not from options.
    cached_loss_outlier_depth_ = std::shared_ptr<ceres::LossFunction>(
        new ceres::ScaledLoss(
            new ceres::HuberLoss(1.0), 1.0, ceres::TAKE_OWNERSHIP));
  } else {
    // Down-weight uncalibrated cameras (upstream default).
    loss_function_ptcam_uncalibrated_ = std::make_shared<ceres::ScaledLoss>(
        loss_function_.get(), 0.5, ceres::DO_NOT_TAKE_OWNERSHIP);
    loss_function_ptcam_calibrated_ = loss_function_;
  }

  // Preserve initial dmap_scales seeds, clear per-observation counts
  // (preserve-and-restore dance).
  if (options_.use_depth_priors) {
    std::map<image_t, double> preserved_scales = dmap_scales_;
    dmap_scales_.clear();
    dmap_scale_observation_counts_.clear();
    for (const auto& [image_id, scale] : preserved_scales) {
      dmap_scales_[image_id] = scale;
    }
    // Pre-compute depth outliers if requested.
    depth_outliers_.clear();
    if (options_.filter_depth_outliers) {
      FilterDepthOutliers(reconstruction);
    }
  }

  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    // skip tracks with no relevant images.
    if (!options_.image_ids_to_optimize.empty()) {
      bool has_relevant = false;
      for (const auto& el : point3D.track.Elements()) {
        if (options_.image_ids_to_optimize.count(el.image_id)) {
          has_relevant = true;
          break;
        }
      }
      if (!has_relevant && options_.use_lc_observations) {
        for (const auto& el : point3D.track.LcElements()) {
          if (options_.image_ids_to_optimize.count(el.image_id)) {
            has_relevant = true;
            break;
          }
        }
      }
      if (!has_relevant) continue;
    }

    if (point3D.track.Length() <
        static_cast<size_t>(options_.min_num_view_per_track)) {
      continue;
    }

    AddPoint3DToProblem(point3D_id, reconstruction);
  }

  // Add scale prior constraints after all depth observations counted.
  if (options_.use_depth_priors && options_.regularize_depth_map_scales) {
    AddScalePriorConstraints();
  }

  LOG(INFO) << "Residuals: " << normal_residuals_ << " normal, "
            << lc_residuals_ << " LC";
}

void GlobalPositioner::AddPoint3DToProblem(point3D_t point3D_id,
                                           Reconstruction& reconstruction) {
  const bool random_initialization =
      options_.optimize_points && options_.generate_random_points;

  Point3D& point3D = reconstruction.Point3D(point3D_id);

  // Only set the points to be random if they are needed to be optimized.
  // Use options_.random_init_scale instead of hardcoded 100.0.
  if (random_initialization && !options_.use_init) {
    point3D.xyz = options_.random_init_scale * RandVector3d(-1, 1);
  }

  // Add regular observations.
  for (const auto& observation : point3D.track.Elements()) {
    if (!options_.image_ids_to_optimize.empty() &&
        !options_.image_ids_to_optimize.count(observation.image_id))
      continue;
    if (options_.use_fork_loss_dispatch || options_.use_depth_priors ||
        options_.use_fork_observation_exclusion) {
      AddObservationToProblem(
          point3D_id, observation.image_id, observation.point2D_idx,
          false, reconstruction);
    } else {
      // Upstream path: fast non-dispatch branch.
      if (!reconstruction.ExistsImage(observation.image_id)) continue;
      Image& image = reconstruction.Image(observation.image_id);
      if (!image.HasPose()) continue;

      const std::optional<Eigen::Vector2d> cam_point =
          image.CameraPtr()->CamFromImg(
              image.Point2D(observation.point2D_idx).xy);
      if (!cam_point.has_value()) {
        LOG(WARNING)
            << "Ignoring feature because it failed to project: point3D_id="
            << point3D_id << ", image_id=" << observation.image_id
            << ", feature_id=" << observation.point2D_idx;
        continue;
      }

      const Eigen::Vector3d cam_from_point3D_dir =
          image.CamFromWorld().rotation().inverse() *
          cam_point->homogeneous().normalized();

      CHECK_GE(scales_.capacity(), scales_.size())
          << "Not enough capacity was reserved for the scales.";
      double& scale = scales_.emplace_back(1);

      if (!options_.generate_scales && random_initialization) {
        const Eigen::Vector3d cam_from_point3D_translation =
            point3D.xyz - frame_centers_[image.FrameId()];
        scale = std::max(1e-5,
                         cam_from_point3D_dir.dot(cam_from_point3D_translation) /
                             cam_from_point3D_translation.squaredNorm());
      }

      Camera& camera = reconstruction.Camera(image.CameraId());
      ceres::LossFunction* loss_function =
          (camera.has_prior_focal_length)
              ? loss_function_ptcam_calibrated_.get()
              : loss_function_ptcam_uncalibrated_.get();

      if (image.IsRefInFrame()) {
        ceres::CostFunction* cost_function =
            BATAPairwiseDirectionCostFunctor::Create(cam_from_point3D_dir);
        problem_->AddResidualBlock(cost_function,
                                   loss_function,
                                   frame_centers_[image.FrameId()].data(),
                                   point3D.xyz.data(),
                                   &scale);
      } else {
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
          problem_->AddResidualBlock(cost_function,
                                     loss_function,
                                     point3D.xyz.data(),
                                     frame_centers_[image.FrameId()].data(),
                                     &scale);
        } else {
          const sensor_t sensor_id = image.CameraPtr()->SensorId();
          if (cams_in_rig_.find(sensor_id) == cams_in_rig_.end()) {
            cams_in_rig_[sensor_id] = Eigen::Vector3d::Zero();
          }
          ceres::CostFunction* cost_function =
              RigBATAPairwiseDirectionCostFunctor::Create(
                  cam_from_point3D_dir,
                  image.FramePtr()->RigFromWorld().rotation());
          problem_->AddResidualBlock(cost_function,
                                     loss_function,
                                     point3D.xyz.data(),
                                     frame_centers_[image.FrameId()].data(),
                                     cams_in_rig_[sensor_id].data(),
                                     &scale);
        }
      }
      problem_->SetParameterLowerBound(&scale, 0, 1e-5);
    }
  }

  // Add LC observations.
  if (options_.use_lc_observations) {
    for (const auto& lc_obs : point3D.track.LcElements()) {
      if (!options_.image_ids_to_optimize.empty() &&
          !options_.image_ids_to_optimize.count(lc_obs.image_id))
        continue;
      AddObservationToProblem(
          point3D_id, lc_obs.image_id, lc_obs.point2D_idx,
          true, reconstruction);
    }
  }
}

void GlobalPositioner::AddCamerasAndPointsToParameterGroups(
    Reconstruction& reconstruction) {
  // Create a custom ordering for Schur-based problems.
  options_.solver_options.linear_solver_ordering.reset(
      new ceres::ParameterBlockOrdering);
  ceres::ParameterBlockOrdering* parameter_ordering =
      options_.solver_options.linear_solver_ordering.get();

  // Add scale parameters to group 0 (large and independent)
  for (double& scale : scales_) {
    parameter_ordering->AddElementToGroup(&scale, 0);
  }

  // Add point parameters to group 1.
  int group_id = 1;
  if (reconstruction.NumPoints3D() > 0) {
    for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
      if (problem_->HasParameterBlock(point3D.xyz.data()))
        parameter_ordering->AddElementToGroup(
            reconstruction.Point3D(point3D_id).xyz.data(), group_id);
    }
    group_id++;
  }

  for (auto& [frame_id, center] : frame_centers_) {
    if (problem_->HasParameterBlock(center.data())) {
      parameter_ordering->AddElementToGroup(center.data(), group_id);
    }
  }

  // Add the cam_in_rig to be estimated into the parameter group
  for (auto& [sensor_id, center] : cams_in_rig_) {
    if (problem_->HasParameterBlock(center.data())) {
      parameter_ordering->AddElementToGroup(center.data(), group_id);
    }
  }

  // add rotation pointers to camera group.
  if (options_.optimize_rotations) {
    for (const auto& [image_id, image] : reconstruction.Images()) {
      if (!image.HasPose()) continue;
      double* rot_ptr =
          image.FramePtr()->RigFromWorld().rotation().coeffs().data();
      if (problem_->HasParameterBlock(rot_ptr)) {
        parameter_ordering->AddElementToGroup(rot_ptr, group_id);
      }
    }
  }

  // add dmap_scales_ to camera group (same group as cameras).
  if (options_.use_depth_priors) {
    for (auto& [image_id, dmap_scale] : dmap_scales_) {
      if (problem_->HasParameterBlock(&dmap_scale)) {
        parameter_ordering->AddElementToGroup(&dmap_scale, group_id);
      }
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
    for (auto& [frame_id, center] : frame_centers_) {
      if (problem_->HasParameterBlock(center.data())) {
        problem_->SetParameterBlockConstant(center.data());
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
  // pin images not in image_ids_to_optimize.
  if (!options_.image_ids_to_optimize.empty()) {
    for (const auto& [image_id, image] : reconstruction.Images()) {
      if (!image.HasPose()) continue;
      if (options_.image_ids_to_optimize.count(image_id)) continue;
      const frame_t frame_id = image.FrameId();
      if (frame_centers_.count(frame_id) &&
          problem_->HasParameterBlock(frame_centers_[frame_id].data())) {
        problem_->SetParameterBlockConstant(frame_centers_[frame_id].data());
      }
    }
  }

  // apply quaternion manifold to rotation parameter blocks.
  if (options_.optimize_rotations) {
    for (const auto& [image_id, image] : reconstruction.Images()) {
      if (!image.HasPose()) continue;
      double* rot_ptr =
          image.FramePtr()->RigFromWorld().rotation().coeffs().data();
      if (problem_->HasParameterBlock(rot_ptr)) {
        SetManifold(problem_.get(), rot_ptr, CreateEigenQuaternionManifold());
      }
    }
  }

  // pin dmap_scales_ when not optimizing.
  if (options_.use_depth_priors && !options_.optimize_depth_map_scales) {
    LOG(INFO) << "Setting depth map scales to constant";
    for (auto& [id, dmap_scale] : dmap_scales_) {
      if (problem_->HasParameterBlock(&dmap_scale))
        problem_->SetParameterBlockConstant(&dmap_scale);
    }
  }

  // scale gauge fix — pin scales_[0] only when !use_depth_priors.
  // When use_depth_priors=true, dmap_scales_ anchor absolute scale.
  if (!options_.use_depth_priors) {
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

void GlobalPositioner::ConvertBackResults(Reconstruction& reconstruction) {
  // Convert optimized frame centers back to rig_from_world translations.
  for (const auto& [frame_id, center] : frame_centers_) {
    Rigid3d& rig_from_world = reconstruction.Frame(frame_id).RigFromWorld();
    rig_from_world.translation() = rig_from_world.rotation() * -center;
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
      sensor_from_rig.translation() = sensor_from_rig.rotation() * -center;
      break;
    }
  }
}

// --- Accessors ---

std::vector<double> GlobalPositioner::GetDepthMapScales() const {
  std::vector<double> result;
  for (const auto& [image_id, scale_or_log] : dmap_scales_) {
    double scale = options_.use_log_scale_for_depth_map_scales
                       ? std::exp(scale_or_log)
                       : scale_or_log;
    result.push_back(scale);
  }
  return result;
}

std::map<image_t, double> GlobalPositioner::GetDepthMapScaleMap() const {
  std::map<image_t, double> result;
  for (const auto& [image_id, scale_or_log] : dmap_scales_) {
    result[image_id] = options_.use_log_scale_for_depth_map_scales
                           ? std::exp(scale_or_log)
                           : scale_or_log;
  }
  return result;
}

std::map<image_t, double> GlobalPositioner::GetDepthMapScaleMapNested() const {
  return GetDepthMapScaleMap();
}

// --- AddObservationToProblem (sections 1–4, 7, 11) ---
//
// Used by AddPoint3DToProblem when use_fork_loss_dispatch || use_depth_priors
// || use_fork_observation_exclusion is enabled. Handles all per-channel loss
// dispatch, rotation-variant cost functions, and MDRP depth residuals.
// All parameter blocks use frame_centers_[frame_id] (upstream convention).

void GlobalPositioner::AddObservationToProblem(
    point3D_t point3D_id,
    image_t image_id,
    point2D_t point2D_idx,
    bool is_lc_observation,
    Reconstruction& reconstruction) {
  if (!reconstruction.ExistsImage(image_id)) return;
  Image& image = reconstruction.Image(image_id);
  if (!image.HasPose()) return;

  // hard exclusion.
  if (options_.use_fork_observation_exclusion &&
      point2D_idx < image.is_excluded.size() &&
      image.is_excluded[point2D_idx])
    return;

  Point3D& point3D = reconstruction.Point3D(point3D_id);
  Eigen::Vector3d& cam_center = frame_centers_[image.FrameId()];

  // Stable pointer to the frame rotation (cam_from_world quaternion).
  // RigFromWorld().rotation() == CamFromWorld().rotation() for single-image
  // frames; for rig frames this is the rig rotation, not sensor rotation.
  Eigen::Map<Eigen::Quaterniond> cam_rotation =
      image.FramePtr()->RigFromWorld().rotation();

  // Compute bearing direction.
  Eigen::Vector3d bearing_cam;
  if (point2D_idx < image.features_undist.size()) {
    bearing_cam = image.features_undist[point2D_idx];
    if (bearing_cam.array().isNaN().any()) {
      // Fallback to CamFromImg unprojection.
      const std::optional<Eigen::Vector2d> cam_pt =
          image.CameraPtr()->CamFromImg(image.Point2D(point2D_idx).xy);
      if (!cam_pt.has_value()) return;
      bearing_cam = cam_pt->homogeneous().normalized();
    }
  } else {
    const std::optional<Eigen::Vector2d> cam_pt =
        image.CameraPtr()->CamFromImg(image.Point2D(point2D_idx).xy);
    if (!cam_pt.has_value()) {
      LOG(WARNING) << "Ignoring feature: point3D_id=" << point3D_id
                   << " image_id=" << image_id
                   << " feature_id=" << point2D_idx;
      return;
    }
    bearing_cam = cam_pt->homogeneous().normalized();
  }

  // World-frame bearing (for non-WithRotation cost functions).
  const Eigen::Vector3d v_ik = cam_rotation.inverse() * bearing_cam;

  // Check outlier classification.
  bool is_outlier =
      depth_outliers_.count(std::make_pair(image_id, point2D_idx)) > 0;

  bool use_depth =
      options_.use_depth_priors &&
      options_.point_constraint_type != PointConstraintType::GEOMETRY_ONLY &&
      point2D_idx < image.depth_prior_validity.size() &&
      image.depth_prior_validity[point2D_idx] &&
      !(is_outlier && is_lc_observation);

  bool use_soft_depth_loss = options_.use_depth_priors && is_outlier &&
                              !is_lc_observation && !use_depth;

  // Lambda: build the directional BATA cost function variant.
  auto make_bata_cost = [&]() -> ceres::CostFunction* {
    if (options_.optimize_rotations) {
      if (point2D_idx < image.angular_cholesky_xy.size() &&
          point2D_idx < image.angular_stddevs_z.size()) {
        const Eigen::Vector3d& chol = image.angular_cholesky_xy[point2D_idx];
        const double sz = std::max(1e-9, image.angular_stddevs_z[point2D_idx]);
        mahalanobis_bata_residuals_++;
        return MahalanobisBATADirectionalErrorWithRotation::Create(
            bearing_cam, chol[0], chol[1], chol[2], sz);
      } else if (point2D_idx < image.angular_stddevs.size()) {
        const Eigen::Vector2d& s = image.angular_stddevs[point2D_idx];
        const double sx = std::max(1e-9, s[0]);
        const double sy = std::max(1e-9, s[1]);
        diagonal_bata_residuals_++;
        return WeightedBATADirectionalErrorWithRotation::Create(
            bearing_cam, sx, sy, 0.5 * (sx + sy));
      }
      unweighted_bata_residuals_++;
      return BATAPairwiseDirectionErrorWithRotation::Create(bearing_cam);
    } else {
      if (point2D_idx < image.angular_cholesky_xy.size() &&
          point2D_idx < image.angular_stddevs_z.size()) {
        const Eigen::Vector3d& chol = image.angular_cholesky_xy[point2D_idx];
        const double sz = std::max(1e-9, image.angular_stddevs_z[point2D_idx]);
        mahalanobis_bata_residuals_++;
        return MahalanobisBATADirectionalError::Create(
            v_ik, cam_rotation, chol[0], chol[1], chol[2], sz);
      } else if (point2D_idx < image.angular_stddevs.size()) {
        const Eigen::Vector2d& s = image.angular_stddevs[point2D_idx];
        const double sx = std::max(1e-9, s[0]);
        const double sy = std::max(1e-9, s[1]);
        diagonal_bata_residuals_++;
        return WeightedBATADirectionalError::Create(
            v_ik, cam_rotation, sx, sy, 0.5 * (sx + sy));
      }
      unweighted_bata_residuals_++;
      return BATAPairwiseDirectionError::Create(v_ik);
    }
  };

  // Pick geometry loss.
  auto pick_geom_loss = [&]() -> ceres::LossFunction* {
    if (!options_.use_fork_loss_dispatch) {
      // Upstream calibrated/uncalibrated dispatch.
      Camera& camera = reconstruction.Camera(image.CameraId());
      return camera.has_prior_focal_length
                 ? loss_function_ptcam_calibrated_.get()
                 : loss_function_ptcam_uncalibrated_.get();
    }
    if (is_lc_observation) return cached_loss_lc_geometry_.get();
    bool is_ta = point2D_idx < image.is_track_anchor.size() &&
                 image.is_track_anchor[point2D_idx];
    bool is_inlier = point2D_idx < image.is_inlier.size() &&
                     image.is_inlier[point2D_idx];
    if (is_ta) return cached_loss_normal_geometry_trackstart_.get();
    if (is_inlier) return cached_loss_normal_geometry_inlier_.get();
    return cached_loss_normal_geometry_.get();
  };

  auto pick_depth_loss = [&]() -> ceres::LossFunction* {
    if (!options_.use_fork_loss_dispatch) {
      return loss_function_ptcam_calibrated_.get();
    }
    if (use_soft_depth_loss) return cached_loss_outlier_depth_.get();
    if (is_lc_observation) return cached_loss_lc_depth_.get();
    bool is_ta = point2D_idx < image.is_track_anchor.size() &&
                 image.is_track_anchor[point2D_idx];
    bool is_inlier = point2D_idx < image.is_inlier.size() &&
                     image.is_inlier[point2D_idx];
    bool is_mdrp_outlier = !is_inlier &&
                           point2D_idx < image.is_depth_outlier.size() &&
                           image.is_depth_outlier[point2D_idx];
    if (is_ta) {
      track_anchor_depth_residuals_++;
      return cached_loss_normal_depth_trackstart_.get();
    }
    if (is_inlier) return cached_loss_normal_depth_inlier_.get();
    if (is_mdrp_outlier) {
      mdrp_depth_outlier_residuals_++;
      return cached_loss_normal_depth_outlier_.get();
    }
    return cached_loss_normal_depth_.get();
  };

  // --- Geometry-only or SPLIT_METRIC_DEPTH branch ---
  if (!use_depth && !use_soft_depth_loss) {
    geometry_only_constraints_++;
    ceres::CostFunction* cost = make_bata_cost();
    if (!cost) return;

    CHECK_GE(scales_.capacity(), scales_.size())
        << "Not enough capacity reserved for scales.";
    double& d_ik = scales_.emplace_back(1.0);
    if (!options_.generate_scales) {
      const Eigen::Vector3d dX = point3D.xyz - cam_center;
      if (dX.squaredNorm() > 1e-10)
        d_ik = std::max(1e-5, v_ik.dot(dX) / dX.squaredNorm());
    }

    ceres::LossFunction* geom_loss = pick_geom_loss();

    if (options_.optimize_rotations) {
      problem_->AddResidualBlock(cost, geom_loss,
                                 cam_rotation.coeffs().data(),
                                 cam_center.data(),
                                 point3D.xyz.data(), &d_ik);
    } else {
      problem_->AddResidualBlock(cost, geom_loss,
                                 cam_center.data(),
                                 point3D.xyz.data(), &d_ik);
    }
    problem_->SetParameterLowerBound(&d_ik, 0, 1e-5);

    if (is_lc_observation) {
      lc_residuals_++;
    } else {
      normal_residuals_++;
      bool is_ta = point2D_idx < image.is_track_anchor.size() &&
                   image.is_track_anchor[point2D_idx];
      bool is_inlier = point2D_idx < image.is_inlier.size() &&
                       image.is_inlier[point2D_idx];
      if (is_ta) track_anchor_residuals_++;
      else if (is_inlier) normal_inlier_residuals_++;
      else normal_outlier_residuals_++;
    }
  } else {
    // SPLIT_METRIC_DEPTH branch.
    depth_constraints_++;

    // 1) Directional BATA residual.
    ceres::CostFunction* cost_dir = make_bata_cost();
    if (cost_dir) {
      CHECK_GE(scales_.capacity(), scales_.size())
          << "Not enough capacity reserved for scales.";
      double& d_ik = scales_.emplace_back(1.0);
      if (!options_.generate_scales) {
        const Eigen::Vector3d dX = point3D.xyz - cam_center;
        if (dX.squaredNorm() > 1e-10)
          d_ik = std::max(1e-5, v_ik.dot(dX) / dX.squaredNorm());
      }

      ceres::LossFunction* geom_loss = pick_geom_loss();

      if (options_.optimize_rotations) {
        problem_->AddResidualBlock(cost_dir, geom_loss,
                                   cam_rotation.coeffs().data(),
                                   cam_center.data(),
                                   point3D.xyz.data(), &d_ik);
      } else {
        problem_->AddResidualBlock(cost_dir, geom_loss,
                                   cam_center.data(),
                                   point3D.xyz.data(), &d_ik);
      }
      problem_->SetParameterLowerBound(&d_ik, 0, 1e-5);

      if (is_lc_observation) {
        lc_residuals_++;
      } else {
        normal_residuals_++;
        bool is_ta = point2D_idx < image.is_track_anchor.size() &&
                     image.is_track_anchor[point2D_idx];
        bool is_inlier = point2D_idx < image.is_inlier.size() &&
                         image.is_inlier[point2D_idx];
        if (is_ta) track_anchor_residuals_++;
        else if (is_inlier) normal_inlier_residuals_++;
        else normal_outlier_residuals_++;
      }
    }

    // 2) Metric depth residual.
    const double depth_prior_val =
        point2D_idx < image.depth_priors.size()
            ? image.depth_priors[point2D_idx]
            : 0.0;
    if (depth_prior_val > 1e-6) {
      const double depth_stddev =
          point2D_idx < image.depth_prior_stddevs.size()
              ? std::max(1e-6, image.depth_prior_stddevs[point2D_idx])
              : 1.0;

      // Get or create dmap_scale parameter for this image.
      auto scale_it = dmap_scales_.find(image_id);
      if (scale_it == dmap_scales_.end()) {
        double init_val =
            options_.use_log_scale_for_depth_map_scales ? 0.0 : 1.0;
        scale_it = dmap_scales_.emplace(image_id, init_val).first;
        dmap_scale_observation_counts_[image_id] = 0;
      }
      dmap_scale_observation_counts_[image_id]++;
      double& dmap_scale = scale_it->second;

      ceres::CostFunction* cost_depth = nullptr;
      if (options_.optimize_rotations) {
        cost_depth = MetricDepthErrorWithRotation::Create(
            depth_prior_val, depth_stddev,
            options_.use_log_scale_for_depth_map_scales,
            options_.use_log_residual_for_depth,
            options_.zero_residual_behind_camera,
            options_.smooth_log_linear_transition,
            options_.log_linear_threshold);
      } else {
        cost_depth = MetricDepthError::Create(
            cam_rotation, depth_prior_val, depth_stddev,
            options_.use_log_scale_for_depth_map_scales,
            options_.use_log_residual_for_depth,
            options_.zero_residual_behind_camera,
            options_.smooth_log_linear_transition,
            options_.log_linear_threshold);
      }

      if (cost_depth) {
        ceres::LossFunction* depth_loss = pick_depth_loss();

        if (options_.optimize_rotations) {
          problem_->AddResidualBlock(cost_depth, depth_loss,
                                     cam_rotation.coeffs().data(),
                                     cam_center.data(),
                                     point3D.xyz.data(), &dmap_scale);
        } else {
          problem_->AddResidualBlock(cost_depth, depth_loss,
                                     cam_center.data(),
                                     point3D.xyz.data(), &dmap_scale);
        }
        if (!options_.use_log_scale_for_depth_map_scales) {
          problem_->SetParameterLowerBound(&dmap_scale, 0, 1e-5);
        }

        if (is_lc_observation) {
          lc_residuals_++;
        } else {
          normal_residuals_++;
          bool is_inlier = point2D_idx < image.is_inlier.size() &&
                           image.is_inlier[point2D_idx];
          if (is_inlier) normal_inlier_residuals_++;
          else normal_outlier_residuals_++;
        }
      }
    }
  }
}

// --- AddScalePriorConstraints ---

void GlobalPositioner::AddScalePriorConstraints() {
  if (!options_.regularize_depth_map_scales) {
    LOG(INFO) << "Scale prior regularization disabled";
    return;
  }

  scale_prior_losses_.clear();

  int count = 0;
  for (const auto& [img_id, obs_count] : dmap_scale_observation_counts_) {
    if (obs_count <= 0) continue;
    auto scale_it = dmap_scales_.find(img_id);
    if (scale_it == dmap_scales_.end()) continue;

    ceres::CostFunction* prior_cost = nullptr;
    if (options_.use_log_scale_for_depth_map_scales) {
      prior_cost = LogScalePriorError::Create(options_.scale_prior_stddev);
    } else {
      prior_cost = ScalePriorError::Create(1.0, options_.scale_prior_stddev);
    }
    if (!prior_cost) continue;

    const double total_weight =
        options_.loss_scale_prior.weight * static_cast<double>(obs_count);

    std::shared_ptr<ceres::LossFunction> base_loss =
        options_.loss_scale_prior.Create();

    std::shared_ptr<ceres::LossFunction> loss;
    if (std::abs(total_weight - 1.0) < 1e-9) {
      loss = base_loss;
    } else {
      loss = std::shared_ptr<ceres::LossFunction>(
          new ceres::ScaledLoss(base_loss.get(), total_weight,
                                ceres::DO_NOT_TAKE_OWNERSHIP));
      scale_prior_losses_.push_back(base_loss);
    }
    scale_prior_losses_.push_back(loss);

    problem_->AddResidualBlock(prior_cost, loss.get(), &scale_it->second);
    count++;
  }
  LOG(INFO) << "Added " << count << " scale prior residuals";
}

// --- AddRelativePoseConstraints ---

void GlobalPositioner::AddRelativePoseConstraints(
    const PoseGraph& pose_graph, Reconstruction& reconstruction) {
  if (!options_.use_relative_pose_constraints ||
      options_.relative_pose_pair_ids.empty())
    return;

  cached_loss_relative_pose_ = options_.loss_relative_pose.Create();

  LOG(INFO) << "Adding relative pose constraints for "
            << options_.relative_pose_pair_ids.size() << " consecutive pairs";

  for (const image_pair_t pair_id : options_.relative_pose_pair_ids) {
    if (pose_graph.Edges().find(pair_id) == pose_graph.Edges().end()) continue;
    const auto& edge = pose_graph.Edges().at(pair_id);
    if (!edge.valid) continue;

    const auto [id1, id2] = PairIdToImagePair(pair_id);
    if (!reconstruction.ExistsImage(id1) || !reconstruction.ExistsImage(id2))
      continue;

    const Image& image1 = reconstruction.Image(id1);
    const Image& image2 = reconstruction.Image(id2);
    if (!image1.HasPose() || !image2.HasPose()) continue;

    frame_t frame1 = image1.FrameId();
    frame_t frame2 = image2.FrameId();
    if (!frame_centers_.count(frame1) || !frame_centers_.count(frame2)) continue;

    // R_w1 = transpose of cam_from_world rotation for image1.
    const Eigen::Matrix3d R_w1 =
        image1.CamFromWorld().rotation().toRotationMatrix().transpose();
    const Eigen::Matrix3d R_21 =
        edge.cam2_from_cam1.rotation().toRotationMatrix();
    const Eigen::Vector3d t_21 = edge.cam2_from_cam1.translation();
    // R_transform = R_w1 * R_21^T, t_expected = -t_21
    // Derivation: c2 - c1 = -R_w1 * R_21^T * t_21 (FORK:1922-1954).
    const Eigen::Matrix3d R_transform = R_w1 * R_21.transpose();
    const Eigen::Vector3d t_expected = -t_21;

    Eigen::Matrix3d cov_t = edge.cov_t;
    if (cov_t.isZero() || cov_t.norm() < 1e-10) {
      const double sigma = options_.relative_pose_default_stddev;
      cov_t = Eigen::Matrix3d::Identity() * (sigma * sigma);
    }

    ceres::CostFunction* cost =
        RelativeTranslationError::Create(R_transform, t_expected, cov_t);
    problem_->AddResidualBlock(cost,
                               cached_loss_relative_pose_.get(),
                               frame_centers_[frame1].data(),
                               frame_centers_[frame2].data());
    relative_pose_constraints_++;
  }

  LOG(INFO) << "Added " << relative_pose_constraints_
            << " relative pose constraints";
}

// --- AddRotationPriors ---

void GlobalPositioner::AddRotationPriors(Reconstruction& reconstruction) {
  if (!options_.optimize_rotations || !options_.regularize_rotations) return;

  cached_loss_rotation_prior_ = options_.loss_rotation_prior.Create();

  int count = 0;
  for (const auto& [image_id, image] : reconstruction.Images()) {
    if (!image.HasPose()) continue;
    if (!options_.image_ids_to_optimize.empty() &&
        !options_.image_ids_to_optimize.count(image_id))
      continue;
    auto it = initial_rotations_.find(image_id);
    if (it == initial_rotations_.end()) continue;

    double* rot_ptr =
        image.FramePtr()->RigFromWorld().rotation().coeffs().data();
    if (!problem_->HasParameterBlock(rot_ptr)) continue;

    ceres::CostFunction* cost =
        RotationPriorError::Create(it->second, options_.rotation_prior_sigma);
    if (!cost) continue;

    problem_->AddResidualBlock(
        cost, cached_loss_rotation_prior_.get(), rot_ptr);
    rotation_prior_constraints_++;
    count++;
  }
  LOG(INFO) << "Added " << count << " rotation prior constraints";
}

// --- FilterDepthOutliers ---

void GlobalPositioner::FilterDepthOutliers(
    const Reconstruction& reconstruction) {
  int total = 0;
  int found = 0;

  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    auto process_obs = [&](const TrackElement& el) {
      if (!reconstruction.ExistsImage(el.image_id)) return;
      const Image& image = reconstruction.Image(el.image_id);
      if (!image.HasPose()) return;

      const point2D_t fidx = el.point2D_idx;
      if (fidx >= image.depth_prior_validity.size() ||
          !image.depth_prior_validity[fidx])
        return;
      if (fidx >= image.depth_priors.size() ||
          fidx >= image.depth_prior_stddevs.size())
        return;

      const double depth_prior_raw = image.depth_priors[fidx];
      const double stddev_rel = image.depth_prior_stddevs[fidx];
      if (depth_prior_raw <= 1e-6 || stddev_rel <= 1e-9) return;

      double depth_prior = depth_prior_raw;
      auto scale_it = dmap_scales_.find(el.image_id);
      if (scale_it != dmap_scales_.end()) {
        double s = options_.use_log_scale_for_depth_map_scales
                       ? std::exp(scale_it->second)
                       : scale_it->second;
        depth_prior = s * depth_prior_raw;
      }

      const frame_t frame_id = image.FrameId();
      if (!frame_centers_.count(frame_id)) return;
      const Eigen::Vector3d& cam_center = frame_centers_.at(frame_id);
      const Eigen::Quaterniond cam_rotation =
          image.FramePtr()->RigFromWorld().rotation();
      const Eigen::Vector3d pt_cam = cam_rotation * (point3D.xyz - cam_center);
      const double z = pt_cam[2];
      if (z <= 1e-6) return;

      total++;
      const double metric_std = stddev_rel * depth_prior;
      const double log_z = std::log(std::max(z, 1e-6));
      const double log_d = std::log(std::max(depth_prior, 1e-6));
      const double log_diff = std::abs(log_z - log_d);
      const double threshold =
          3.0 * std::log(1.0 + std::max(metric_std / std::max(depth_prior, 1e-6),
                                         1e-6));
      if (log_diff >= threshold) {
        depth_outliers_.insert({el.image_id, fidx});
        found++;
      }
    };

    for (const auto& el : point3D.track.Elements()) process_obs(el);
    if (options_.use_lc_observations) {
      for (const auto& el : point3D.track.LcElements()) process_obs(el);
    }
  }

  LOG(INFO) << "Depth outlier filtering: checked " << total << ", found "
            << found << " outliers";
}

// --- InitializeDepthMapScalesFromObservations ---

void GlobalPositioner::InitializeDepthMapScalesFromObservations(
    const Reconstruction& reconstruction) {
  std::map<image_t, std::vector<double>> scale_estimates;

  auto process_obs = [&](const Point3D& point3D, const TrackElement& el) {
    if (!reconstruction.ExistsImage(el.image_id)) return;
    const Image& image = reconstruction.Image(el.image_id);
    if (!image.HasPose()) return;

    const point2D_t fidx = el.point2D_idx;
    if (fidx >= image.depth_prior_validity.size() ||
        !image.depth_prior_validity[fidx])
      return;
    if (fidx >= image.depth_priors.size()) return;

    const double depth_prior = image.depth_priors[fidx];
    if (depth_prior <= 1e-6) return;

    const frame_t frame_id = image.FrameId();
    if (!frame_centers_.count(frame_id)) return;
    const Eigen::Vector3d& cam_center = frame_centers_.at(frame_id);
    const Eigen::Quaterniond cam_rotation =
        image.FramePtr()->RigFromWorld().rotation();
    const double z = (cam_rotation * (point3D.xyz - cam_center))[2];
    if (z <= 1e-6) return;

    const double s = z / depth_prior;
    if (s > 1e-6 && s < 1e6) scale_estimates[el.image_id].push_back(s);
  };

  for (const auto& [id, pt] : reconstruction.Points3D()) {
    for (const auto& el : pt.track.Elements()) process_obs(pt, el);
    if (options_.use_lc_observations) {
      for (const auto& el : pt.track.LcElements()) process_obs(pt, el);
    }
  }

  int count = 0;
  for (auto& [image_id, estimates] : scale_estimates) {
    if (estimates.empty()) continue;
    std::sort(estimates.begin(), estimates.end());
    const double median = estimates[estimates.size() / 2];
    dmap_scales_[image_id] =
        options_.use_log_scale_for_depth_map_scales ? std::log(median) : median;
    dmap_scale_observation_counts_[image_id] = 0;
    count++;
  }
  LOG(INFO) << "Auto-initialized " << count << " depth map scales (median)";
}

bool RunGlobalPositioning(const GlobalPositionerOptions& options,
                          const PoseGraph& pose_graph,
                          Reconstruction& reconstruction) {
  GlobalPositioner positioner(options);
  return positioner.Solve(pose_graph, reconstruction);
}

}  // namespace colmap
