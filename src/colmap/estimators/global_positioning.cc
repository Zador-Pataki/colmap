#include "colmap/estimators/global_positioning.h"

#include <cstdlib>

#include "colmap/estimators/cost_functions/metric_depth.h"
#include "colmap/estimators/cost_functions/motion_averaging.h"
#include "colmap/estimators/loss_config.h"
#include "colmap/math/random.h"
#include "colmap/util/cuda.h"
#include "colmap/util/misc.h"
#include "colmap/util/threading.h"

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
  // TODO(reproduce-fork, M2/Decision-9, Q8): native colmap4 only honors
  // options_.random_seed >= 0. The GP_SEED env-var fallback is a
  // transition crutch so the documented Tier-2 byte-identity recipe
  // (CLAUDE.md § "ATE byte-identity") works without patching every
  // caller. Drop this branch once all callers set random_seed
  // explicitly.
  if (options_.random_seed >= 0) {
    SetPRNGSeed(static_cast<unsigned>(options_.random_seed));
  } else if (const char* env_seed = std::getenv("GP_SEED")) {
    SetPRNGSeed(static_cast<unsigned>(std::atoi(env_seed)));
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

  // Setup the problem.
  SetupProblem(pose_graph, reconstruction);

  // Initialize camera translations to be random.
  // Also, convert the camera pose translation to be the camera center.
  InitializeRandomPositions(pose_graph, reconstruction);

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
  ceres::Solve(options_.solver_options, problem_.get(), &summary);

  if (VLOG_IS_ON(2)) {
    LOG(INFO) << summary.FullReport();
  } else {
    LOG(INFO) << summary.BriefReport();
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
  // smaller. Include both regular observations + glomap-fork lc_elements
  // (M5 two-loop iteration adds residuals + scales for both); without the
  // lc count, vector reallocation invalidates earlier &scale pointers
  // stored in Ceres residual blocks.
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
    if (point3D.track.Length() <
        static_cast<size_t>(options_.min_num_view_per_track)) {
      continue;
    }
    for (const auto& observation : point3D.track.Elements()) {
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
    if (options_.generate_random_positions && options_.optimize_positions &&
        !options_.use_init) {
      frame_centers_[frame_id] = options_.random_init_scale * RandVector3d(-1, 1);
    } else {
      frame_centers_[frame_id] = frame.RigFromWorld().TgtOriginInSrc();
    }
  }

  VLOG(2) << "Constrained positions: " << constrained_positions.size();
}

void GlobalPositioner::AddPointToCameraConstraints(
    Reconstruction& reconstruction) {
  VLOG(2) << reconstruction.NumPoints3D()
          << " point to camera constraints were added to the position "
             "estimation problem.";

  // Down-weight uncalibrated cameras.
  loss_function_ptcam_uncalibrated_ = std::make_shared<ceres::ScaledLoss>(
      loss_function_.get(), 0.5, ceres::DO_NOT_TAKE_OWNERSHIP);
  loss_function_ptcam_calibrated_ = loss_function_;

  // --- Glomap-fork loss-routing cascade pre-warm (M5) ---
  // Materialize the 10 cached losses from their LossFunctionConfig fields.
  // Default-config (all loss_*.name == "trivial", scale=1, weight=1) gives
  // unweighted residuals — equivalent to no override. The
  // AddObservationToProblem cascade picks one of these based on
  // (is_lc_observation, is_track_anchor, is_inlier, is_depth_outlier,
  // depth_outliers_) flags. ``soft_outlier_fallback_loss_`` is allocated
  // lazily on first non-LC depth-outlier observation in the cascade.
  cached_loss_normal_geometry_ =
      CreateLossFromConfig(options_.loss_normal_geometry);
  cached_loss_normal_depth_ = CreateLossFromConfig(options_.loss_normal_depth);
  cached_loss_lc_geometry_ = CreateLossFromConfig(options_.loss_lc_geometry);
  cached_loss_lc_depth_ = CreateLossFromConfig(options_.loss_lc_depth);
  cached_loss_normal_geometry_inlier_ =
      CreateLossFromConfig(options_.loss_normal_geometry_inlier);
  cached_loss_normal_depth_inlier_ =
      CreateLossFromConfig(options_.loss_normal_depth_inlier);
  cached_loss_normal_depth_outlier_ =
      CreateLossFromConfig(options_.loss_normal_depth_outlier);
  cached_loss_normal_geometry_trackstart_ =
      CreateLossFromConfig(options_.loss_normal_geometry_trackstart);
  cached_loss_normal_depth_trackstart_ =
      CreateLossFromConfig(options_.loss_normal_depth_trackstart);
  cached_loss_scale_prior_ = CreateLossFromConfig(options_.loss_scale_prior);
  soft_outlier_fallback_loss_.reset();

  // --- Glomap-fork SPLIT_METRIC_DEPTH bookkeeping (M4) ---
  // Reset per-image scale state. Lazy-inserted from observation loop in
  // AddPoint3DToProblem.
  dmap_scales_.clear();
  dmap_scale_observation_counts_.clear();

  // --- Glomap-fork initial_dmap_scales seeding + filter pre-pass (M6) ---
  // When the caller supplied initial_dmap_scales (e.g. GP2 receiving GP1's
  // solved scales), seed dmap_scales_ before FilterDepthOutliers so the
  // log-space residual check uses the right per-image scale. Lazy-insert
  // in AddObservationToProblem then skips these images. observation_count
  // starts at 0 and is incremented per observation processed.
  if (options_.point_constraint_type ==
          PointConstraintType::SPLIT_METRIC_DEPTH &&
      options_.initial_dmap_scales.has_value()) {
    for (const auto& [image_id, linear_scale] :
         *options_.initial_dmap_scales) {
      const double init_value = options_.use_log_scale_for_depth_map_scales
                                    ? std::log(std::max(linear_scale, 1e-9))
                                    : linear_scale;
      dmap_scales_[image_id] = init_value;
      dmap_scale_observation_counts_[image_id] = 0;
    }
  }

  // Pre-Solve depth outlier filter. Populates depth_outliers_ which the
  // M5 depth-loss cascade reads.
  depth_outliers_.clear();
  if (options_.point_constraint_type ==
          PointConstraintType::SPLIT_METRIC_DEPTH &&
      options_.filter_depth_outliers) {
    FilterDepthOutliers(reconstruction);
  }

  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    if (point3D.track.Length() <
        static_cast<size_t>(options_.min_num_view_per_track)) {
      continue;
    }

    AddPoint3DToProblem(point3D_id, reconstruction);
  }

  // --- Glomap-fork ScalePriorError emission (M4) ---
  // After every observation has been added, emit one scale-prior residual
  // per image with depth observations. Loss is scaled by observation count
  // so dense-depth images get proportionally stronger priors.
  if (options_.point_constraint_type ==
      PointConstraintType::SPLIT_METRIC_DEPTH) {
    for (auto& [image_id, scale] : dmap_scales_) {
      auto count_it = dmap_scale_observation_counts_.find(image_id);
      const double obs_count =
          (count_it != dmap_scale_observation_counts_.end())
              ? static_cast<double>(count_it->second)
              : 1.0;

      ceres::CostFunction* scale_prior_cost = nullptr;
      if (options_.use_log_scale_for_depth_map_scales) {
        scale_prior_cost =
            LogScalePriorError::Create(options_.scale_prior_stddev);
      } else {
        scale_prior_cost =
            ScalePriorError::Create(1.0, options_.scale_prior_stddev);
      }
      if (scale_prior_cost == nullptr) continue;

      // Per-image obs_count weighting (M4) wrapped around the cached
      // scale-prior loss (M5). When ``loss_scale_prior`` is left at default
      // (TrivialLoss / weight=1), the ScaledLoss collapses to a plain
      // 1/obs_count^-1 weighting — equivalent to fork's behaviour.
      ceres::LossFunction* obs_count_scaled_loss = nullptr;
      if (cached_loss_scale_prior_) {
        obs_count_scaled_loss = new ceres::ScaledLoss(
            cached_loss_scale_prior_.get(),
            obs_count,
            ceres::DO_NOT_TAKE_OWNERSHIP);
      } else {
        obs_count_scaled_loss = new ceres::ScaledLoss(
            new ceres::TrivialLoss(), obs_count, ceres::TAKE_OWNERSHIP);
      }

      problem_->AddResidualBlock(
          scale_prior_cost, obs_count_scaled_loss, &scale);
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
  const bool random_initialization =
      options_.optimize_points && options_.generate_random_points &&
      !options_.use_init;

  Point3D& point3D = reconstruction.Point3D(point3D_id);

  // Only set the points to be random if they are needed to be optimized
  if (random_initialization) {
    point3D.xyz = options_.random_init_scale * RandVector3d(-1, 1);
  }

  // Glomap-fork two-loop iteration (M5):
  // - First loop walks regular track elements (``is_lc_observation=false``).
  // - Second loop walks LC elements (``is_lc_observation=true``).
  // The fork keeps these as parallel collections rather than flagging
  // individual elements; the caller's loss-routing cascade picks the
  // appropriate cached loss based on the flag.
  for (const auto& observation : point3D.track.Elements()) {
    AddObservationToProblem(point3D_id,
                            observation,
                            /*is_lc_observation=*/false,
                            random_initialization,
                            reconstruction);
  }
  for (const auto& observation : point3D.track.lc_elements) {
    AddObservationToProblem(point3D_id,
                            observation,
                            /*is_lc_observation=*/true,
                            random_initialization,
                            reconstruction);
  }
}

void GlobalPositioner::AddObservationToProblem(point3D_t point3D_id,
                                               const TrackElement& observation,
                                               bool is_lc_observation,
                                               bool random_initialization,
                                               Reconstruction& reconstruction) {
  Point3D& point3D = reconstruction.Point3D(point3D_id);
    if (!reconstruction.ExistsImage(observation.image_id)) return;

    Image& image = reconstruction.Image(observation.image_id);
    if (!image.HasPose()) return;

    // --- Glomap-fork is_excluded skip (M5) ---
    if (observation.point2D_idx < image.is_excluded.size() &&
        image.is_excluded[observation.point2D_idx]) {
      return;
    }

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

    // For calibrated and uncalibrated cameras, use different loss
    // functions
    // Down weight the uncalibrated cameras
    Camera& camera = reconstruction.Camera(image.CameraId());
    ceres::LossFunction* loss_function =
        (camera.has_prior_focal_length)
            ? loss_function_ptcam_calibrated_.get()
            : loss_function_ptcam_uncalibrated_.get();

    // --- Glomap-fork loss-routing cascade (M5) ---
    // When SPLIT_METRIC_DEPTH path is active, the geometry loss is
    // selected per-observation from the 4-way cached cascade:
    //   is_lc           -> cached_loss_lc_geometry_
    //   is_track_anchor -> cached_loss_normal_geometry_trackstart_
    //   is_inlier       -> cached_loss_normal_geometry_inlier_
    //   else            -> cached_loss_normal_geometry_
    // Default-config (all cascade losses set to TrivialLoss / weight=1)
    // is equivalent to no override. Calibrated/uncalibrated discrimination
    // is preserved by composing the cascade output with native's existing
    // 0.5x ScaledLoss for !has_prior_focal_length cameras (handled by
    // wrapping cached_loss in loss_function_ptcam_uncalibrated_).
    if (options_.point_constraint_type ==
        PointConstraintType::SPLIT_METRIC_DEPTH) {
      ceres::LossFunction* cascade = nullptr;
      if (is_lc_observation) {
        cascade = cached_loss_lc_geometry_.get();
      } else if (observation.point2D_idx < image.is_track_anchor.size() &&
                 image.is_track_anchor[observation.point2D_idx]) {
        cascade = cached_loss_normal_geometry_trackstart_.get();
      } else if (observation.point2D_idx < image.is_inlier.size() &&
                 image.is_inlier[observation.point2D_idx]) {
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
      // --- Glomap-fork WeightedBATADirectionalError dispatch (M5 fix) ---
      // Use anisotropic per-keypoint weighting when image.angular_stddevs
      // is populated (fork's typical case — videosfm always populates).
      // Fall back to unweighted BATAPairwiseDirectionCostFunctor when
      // sigmas are absent. Without this dispatch the SPLIT_METRIC_DEPTH
      // path's geometry residuals lose their relative weighting against
      // the metric-depth residuals — was the dominant contributor to the
      // M7+M12 ATE drift (audit_algorithmic_semantics.md suspect #1).
      ceres::CostFunction* cost_function = nullptr;
      if (observation.point2D_idx < image.angular_stddevs.size()) {
        const Eigen::Vector2d& angular_std =
            image.angular_stddevs[observation.point2D_idx];
        const double sigma_x = std::max(1e-9, angular_std[0]);
        const double sigma_y = std::max(1e-9, angular_std[1]);
        const double sigma_z = 0.5 * (sigma_x + sigma_y);
        cost_function = WeightedBATADirectionalError::Create(
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

      problem_->AddResidualBlock(cost_function,
                                 loss_function,
                                 frame_centers_[image.FrameId()].data(),
                                 point3D.xyz.data(),
                                 &scale);

      // --- Glomap-fork SPLIT_METRIC_DEPTH addition (M4) ---
      // Add a 1-D ``MetricDepthError`` residual on
      // (frame_center, point3D.xyz, dmap_scales_[image_id]) when the image
      // has a valid depth prior at this feature. Anchors absolute scale.
      if (options_.point_constraint_type ==
              PointConstraintType::SPLIT_METRIC_DEPTH &&
          observation.point2D_idx < image.depth_prior_validity.size() &&
          image.depth_prior_validity[observation.point2D_idx]) {
        const double depth_prior = image.depth_priors[observation.point2D_idx];
        const double depth_sigma =
            image.depth_prior_stddevs[observation.point2D_idx];

        if (depth_prior > 0.0 && depth_sigma > 1e-9) {
          // Lazy-insert dmap_scales_ on first valid observation per image.
          // Initial value: from options_.initial_dmap_scales (caller-supplied
          // GP1→GP2 handoff) if provided, else 1.0 (linear) / 0.0 (log).
          if (dmap_scales_.find(observation.image_id) ==
              dmap_scales_.end()) {
            double init_value =
                options_.use_log_scale_for_depth_map_scales ? 0.0 : 1.0;
            if (options_.initial_dmap_scales.has_value()) {
              const auto& init_map = *options_.initial_dmap_scales;
              auto it = init_map.find(observation.image_id);
              if (it != init_map.end()) {
                // Caller-supplied value is in linear space; convert to
                // log if option says so.
                init_value = options_.use_log_scale_for_depth_map_scales
                                 ? std::log(std::max(it->second, 1e-9))
                                 : it->second;
              }
            }
            dmap_scales_[observation.image_id] = init_value;
            dmap_scale_observation_counts_[observation.image_id] = 0;
          }
          dmap_scale_observation_counts_[observation.image_id]++;

          ceres::CostFunction* metric_depth_cost = MetricDepthError::Create(
              image.CamFromWorld().rotation(),
              depth_prior,
              depth_sigma,
              options_.use_log_scale_for_depth_map_scales,
              options_.use_log_residual_for_depth,
              options_.zero_residual_behind,
              options_.smooth_log_linear_transition,
              options_.log_linear_threshold);

          if (metric_depth_cost != nullptr) {
            // --- Glomap-fork depth-loss cascade (M5) ---
            // 5-way cascade: pre-pass-flagged outlier (M6) -> soft fallback,
            // then is_lc -> cached_loss_lc_depth_, track_anchor ->
            // cached_loss_normal_depth_trackstart_, inlier ->
            // cached_loss_normal_depth_inlier_, MDRP-flagged outlier ->
            // cached_loss_normal_depth_outlier_, else ->
            // cached_loss_normal_depth_.
            ceres::LossFunction* depth_loss = nullptr;
            const std::pair<image_t, point2D_t> obs_key{observation.image_id,
                                                        observation.point2D_idx};
            if (depth_outliers_.count(obs_key) > 0) {
              if (is_lc_observation) {
                // LC outlier: skip the depth residual entirely (only emit
                // BATA above). Drop this metric_depth_cost.
                delete metric_depth_cost;
                metric_depth_cost = nullptr;
              } else {
                // Non-LC outlier: hardcoded soft fallback (HuberLoss(1)
                // wrapped in ScaledLoss(1)).
                if (!soft_outlier_fallback_loss_) {
                  soft_outlier_fallback_loss_ =
                      std::make_shared<ceres::ScaledLoss>(
                          new ceres::HuberLoss(1.0),
                          1.0,
                          ceres::TAKE_OWNERSHIP);
                }
                depth_loss = soft_outlier_fallback_loss_.get();
              }
            } else if (is_lc_observation) {
              depth_loss = cached_loss_lc_depth_.get();
            } else if (observation.point2D_idx < image.is_track_anchor.size() &&
                       image.is_track_anchor[observation.point2D_idx]) {
              depth_loss = cached_loss_normal_depth_trackstart_.get();
            } else if (observation.point2D_idx < image.is_inlier.size() &&
                       image.is_inlier[observation.point2D_idx]) {
              depth_loss = cached_loss_normal_depth_inlier_.get();
            } else if (observation.point2D_idx < image.is_depth_outlier.size() &&
                       image.is_depth_outlier[observation.point2D_idx]) {
              depth_loss = cached_loss_normal_depth_outlier_.get();
            } else {
              depth_loss = cached_loss_normal_depth_.get();
            }

            if (metric_depth_cost != nullptr) {
              problem_->AddResidualBlock(
                  metric_depth_cost,
                  depth_loss,
                  frame_centers_[image.FrameId()].data(),
                  point3D.xyz.data(),
                  &dmap_scales_[observation.image_id]);
            }
          }
        }
      }
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

        problem_->AddResidualBlock(cost_function,
                                   loss_function,
                                   point3D.xyz.data(),
                                   frame_centers_[image.FrameId()].data(),
                                   &scale);
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

  // --- Glomap-fork SPLIT_METRIC_DEPTH parameter group (M4) ---
  // Per-image dmap_scales_ go in a separate group (one beyond
  // frame_centers/cams_in_rig). dmap_scales_ are 1-D blocks; mixing them
  // with 3-D frame_centers in the same Schur-ordering group breaks the
  // Schur-complement preprocessor (Ceres downgrades to
  // SPARSE_NORMAL_CHOLESKY then fails to start). Separate group keeps
  // each Schur block size-uniform.
  ++group_id;
  for (auto& [image_id, scale] : dmap_scales_) {
    if (problem_->HasParameterBlock(&scale)) {
      parameter_ordering->AddElementToGroup(&scale, group_id);
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

  // --- Glomap-fork SPLIT_METRIC_DEPTH parameter bound (M4) ---
  // Lower-bound dmap_scales_ in linear space (prevents collapse to <=0).
  // No bound in log space (parameter is unbounded; positivity comes from
  // exp() in MetricDepthError).
  if (!options_.use_log_scale_for_depth_map_scales) {
    for (auto& [image_id, scale] : dmap_scales_) {
      if (problem_->HasParameterBlock(&scale)) {
        problem_->SetParameterLowerBound(&scale, 0, 1e-5);
      }
    }
  }
  // Set the first scale to be constant to remove the gauge ambiguity. Skip
  // when the metric-depth path is active: ``ScalePriorError`` (M4) plus the
  // depth-prior observations themselves anchor the gauge, and the redundant
  // pin would over-constrain the system. (Q1 / R10 — gating preserves native
  // colmap GP unit tests under default ``BATA`` mode.)
  if (options_.point_constraint_type != PointConstraintType::SPLIT_METRIC_DEPTH) {
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

// --- Glomap-fork FilterDepthOutliers (M6) ---
// Sweep regular + LC observations per track. For each, compute estimated
// camera-frame z-depth via the WORLD pose (not the in-Solve flipped
// frame_centers_ convention — Solve is about to start, frame_centers_ is
// raw initial-cam-position). Flag |log(z_est) - log(scale*prior)| >= 3 sigma_log
// where sigma_log = std::log(1 + relative_stddev). Insert (image_id,
// point2D_idx) into depth_outliers_. M5 cascade routes flagged observations
// to soft fallback (non-LC) or skip-residual (LC).
namespace {

// Helper: per-observation outlier check. Returns true to insert into
// depth_outliers_.
inline bool DepthOutlierFlag(
    const Image& image,
    point2D_t feature_id,
    const Eigen::Vector3d& point3D_xyz,
    bool use_log_scale,
    const std::map<image_t, double>& dmap_scales,
    image_t image_id) {
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
  const double threshold = 3.0 * std::log(1.0 + std::max(stddev_rel, 1e-6));
  return log_diff >= threshold;
}

}  // namespace

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
                           observation.image_id)) {
        depth_outliers_.insert(
            {observation.image_id, observation.point2D_idx});
      }
    }
    // LC observations
    for (const auto& observation : point3D.track.lc_elements) {
      if (!reconstruction.ExistsImage(observation.image_id)) continue;
      const Image& image = reconstruction.Image(observation.image_id);
      if (!image.HasPose()) continue;
      if (DepthOutlierFlag(image,
                           observation.point2D_idx,
                           point3D.xyz,
                           options_.use_log_scale_for_depth_map_scales,
                           dmap_scales_,
                           observation.image_id)) {
        depth_outliers_.insert(
            {observation.image_id, observation.point2D_idx});
      }
    }
  }
  VLOG(2) << "FilterDepthOutliers: flagged " << depth_outliers_.size()
          << " observations as depth outliers.";
}

bool RunGlobalPositioning(const GlobalPositionerOptions& options,
                          const PoseGraph& pose_graph,
                          Reconstruction& reconstruction) {
  GlobalPositioner positioner(options);
  return positioner.Solve(pose_graph, reconstruction);
}

}  // namespace colmap
