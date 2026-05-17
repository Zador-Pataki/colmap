// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.

#include "colmap/estimators/mpsfm_bundle_adjustment_caspar.h"

#ifdef CASPAR_ENABLED
#include "colmap/estimators/caspar/caspar_model_adapter.h"
#endif
#include "colmap/sensor/models.h"
#include "colmap/util/cuda.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"
#include "colmap/util/timer.h"

#include <cmath>
#include <map>
#include <sstream>
#include <unordered_set>

namespace colmap {
namespace {

void ValidatePinholeOnly(const BundleAdjustmentConfig& config,
                         const Reconstruction& reconstruction) {
  for (const image_t image_id : config.Images()) {
    const Image& image = reconstruction.Image(image_id);
    const Camera& camera = *image.CameraPtr();
    THROW_CHECK_EQ(camera.model_id, CameraModelId::kPinhole)
        << "MPSFM Caspar bundle adjustment supports only PINHOLE cameras. "
        << "Image " << image_id << " uses " << camera.ModelName() << ".";
  }
}

void ValidateCameraIdIsPinhole(const camera_t camera_id,
                               const Reconstruction& reconstruction,
                               const char* factor_name) {
  THROW_CHECK(reconstruction.ExistsCamera(camera_id))
      << factor_name << " references missing camera " << camera_id << ".";
  const Camera& camera = reconstruction.Camera(camera_id);
  THROW_CHECK_EQ(camera.model_id, CameraModelId::kPinhole)
      << factor_name << " supports only PINHOLE cameras. Camera " << camera_id
      << " uses " << camera.ModelName() << ".";
}

void ValidateInputs(const BundleAdjustmentOptions& options,
                    const BundleAdjustmentConfig& config,
                    const Reconstruction& reconstruction,
                    const MpsfmCasparBundleAdjustmentProblem& problem) {
  THROW_CHECK_EQ(options.backend, BundleAdjustmentBackend::CASPAR)
      << "solve_mpsfm_caspar_bundle_adjustment requires "
         "BundleAdjustmentOptions.backend = BundleAdjustmentBackend.CASPAR.";
  THROW_CHECK(!problem.tie_focal)
      << "MPSFM Caspar bundle adjustment does not support tie_focal=True. "
         "That path converts PINHOLE to SIMPLE_PINHOLE.";
  THROW_CHECK(problem.log_depth)
      << "MPSFM Caspar bundle adjustment currently exposes the VideoSfM "
         "log-depth objective only.";
  THROW_CHECK(!problem.reprojection.use_keypoint_covariances)
      << "MPSFM Caspar reprojection covariance weighting was requested, but "
         "this COLMAP worktree does not store per-Point2D covariance matrices "
         "on Image/Point2D objects. Disable use_kp_covariances_ba or extend "
         "the reconstruction data model to pass covariance square-root "
         "information into Caspar.";

  ValidatePinholeOnly(config, reconstruction);

  for (const auto& prior : problem.intrinsics_priors) {
    ValidateCameraIdIsPinhole(
        prior.camera_id, reconstruction, "MPSFM intrinsics prior");
  }
  for (const auto& edge : problem.intrinsics_random_walks) {
    ValidateCameraIdIsPinhole(
        edge.prev_camera_id, reconstruction, "MPSFM intrinsics random-walk");
    ValidateCameraIdIsPinhole(
        edge.next_camera_id, reconstruction, "MPSFM intrinsics random-walk");
  }

  THROW_CHECK_GT(problem.reprojection.keypoint_std, 0);
  THROW_CHECK_GE(problem.reprojection.residual_weight, 0);
}

int CountReprojectionFactors(const BundleAdjustmentConfig& config,
                             const Reconstruction& reconstruction) {
  int num_factors = 0;
  for (const image_t image_id : config.Images()) {
    const Image& image = reconstruction.Image(image_id);
    for (const Point2D& point2D : image.Points2D()) {
      if (point2D.HasPoint3D() && !config.IsIgnoredPoint(point2D.point3D_id)) {
        ++num_factors;
      }
    }
  }
  return num_factors;
}

#ifdef CASPAR_ENABLED
double LossTypeToDouble(const MpsfmCasparLossType type) {
  switch (type) {
    case MpsfmCasparLossType::TRIVIAL:
      return 0.0;
    case MpsfmCasparLossType::SOFT_L1:
      return 1.0;
    case MpsfmCasparLossType::CAUCHY:
      return 2.0;
    case MpsfmCasparLossType::HUBER:
      return 3.0;
  }
  return 0.0;
}

void AppendPose(std::vector<StorageType>& out, const Rigid3d& pose) {
  out.push_back(static_cast<StorageType>(pose.rotation().x()));
  out.push_back(static_cast<StorageType>(pose.rotation().y()));
  out.push_back(static_cast<StorageType>(pose.rotation().z()));
  out.push_back(static_cast<StorageType>(pose.rotation().w()));
  out.push_back(static_cast<StorageType>(pose.translation().x()));
  out.push_back(static_cast<StorageType>(pose.translation().y()));
  out.push_back(static_cast<StorageType>(pose.translation().z()));
}

void AppendRotation(std::vector<StorageType>& out, const Rigid3d& pose) {
  out.push_back(static_cast<StorageType>(pose.rotation().x()));
  out.push_back(static_cast<StorageType>(pose.rotation().y()));
  out.push_back(static_cast<StorageType>(pose.rotation().z()));
  out.push_back(static_cast<StorageType>(pose.rotation().w()));
}

void AppendTranslation(std::vector<StorageType>& out, const Rigid3d& pose) {
  out.push_back(static_cast<StorageType>(pose.translation().x()));
  out.push_back(static_cast<StorageType>(pose.translation().y()));
  out.push_back(static_cast<StorageType>(pose.translation().z()));
}

void AppendPoint(std::vector<StorageType>& out, const Point3D& point) {
  out.push_back(static_cast<StorageType>(point.xyz.x()));
  out.push_back(static_cast<StorageType>(point.xyz.y()));
  out.push_back(static_cast<StorageType>(point.xyz.z()));
}

void AppendCalib(std::vector<StorageType>& out, const Camera& camera) {
  out.push_back(static_cast<StorageType>(camera.params[0]));
  out.push_back(static_cast<StorageType>(camera.params[1]));
  out.push_back(static_cast<StorageType>(camera.params[2]));
  out.push_back(static_cast<StorageType>(camera.params[3]));
}

void AppendFocal(std::vector<StorageType>& out, const Camera& camera) {
  out.push_back(static_cast<StorageType>(camera.params[0]));
  out.push_back(static_cast<StorageType>(camera.params[1]));
}

void AppendPrincipalPoint(std::vector<StorageType>& out,
                          const Camera& camera) {
  out.push_back(static_cast<StorageType>(camera.params[2]));
  out.push_back(static_cast<StorageType>(camera.params[3]));
}

void AppendRobustLoss(std::vector<StorageType>& out,
                      const MpsfmCasparLossType type,
                      const double scale,
                      const double magnitude) {
  THROW_CHECK_GE(scale, 0);
  THROW_CHECK_GE(magnitude, 0);
  out.push_back(static_cast<StorageType>(LossTypeToDouble(type)));
  out.push_back(static_cast<StorageType>(scale));
  out.push_back(static_cast<StorageType>(magnitude));
}

void AppendReprojectionWeightLoss(
    std::vector<StorageType>& out,
    const MpsfmCasparReprojectionOptions& options) {
  const double sqrt_weight =
      std::sqrt(options.residual_weight) / options.keypoint_std;
  out.push_back(static_cast<StorageType>(sqrt_weight));
  out.push_back(static_cast<StorageType>(0));
  out.push_back(static_cast<StorageType>(0));
  out.push_back(static_cast<StorageType>(sqrt_weight));
  AppendRobustLoss(out, options.loss_type, options.loss_scale, 1.0);
}

enum class IntrinsicsMode { FULL, FOCAL_ONLY, FIXED };

struct ReprojData {
  std::vector<unsigned int> pose;
  std::vector<unsigned int> translation;
  std::vector<unsigned int> calib;
  std::vector<unsigned int> focal;
  std::vector<unsigned int> point;
  std::vector<StorageType> const_pose;
  std::vector<StorageType> const_rotation;
  std::vector<StorageType> const_calib;
  std::vector<StorageType> const_focal;
  std::vector<StorageType> const_pp;
  std::vector<StorageType> const_point;
  std::vector<StorageType> pixel;
  std::vector<StorageType> weight_loss;
  size_t num = 0;
};

struct DepthData {
  std::vector<unsigned int> pose;
  std::vector<unsigned int> translation;
  std::vector<unsigned int> scale;
  std::vector<unsigned int> point;
  std::vector<StorageType> const_pose;
  std::vector<StorageType> const_rotation;
  std::vector<StorageType> const_scale;
  std::vector<StorageType> const_point;
  std::vector<StorageType> log_depth;
  std::vector<StorageType> loss;
  size_t num = 0;
};

class MpsfmCasparGraph {
 public:
  MpsfmCasparGraph(const BundleAdjustmentOptions& options,
                   const BundleAdjustmentConfig& config,
                   Reconstruction& reconstruction,
                   MpsfmCasparBundleAdjustmentProblem& problem)
      : options_(options),
        config_(config),
        reconstruction_(reconstruction),
        problem_(problem) {
    BuildNodes();
    BuildReprojectionFactors();
    BuildDepthFactors();
    BuildIntrinsicsFactors();
    BuildScalePriors();
  }

  std::shared_ptr<MpsfmCasparBundleAdjustmentSummary> Solve(
      MpsfmCasparBundleAdjustmentSummary& summary) {
    if (NumResiduals() == 0) {
      LOG(FATAL_THROW) << "MPSFM Caspar graph has no residuals.";
    }

    caspar::SolverParams<double> params;
    int gpu_index = -1;
    if (options_.caspar) {
      const auto& co = *options_.caspar;
      const std::vector<int> gpu_indices = CSVToVector<int>(co.gpu_index);
      THROW_CHECK_GT(gpu_indices.size(), 0);
      gpu_index = gpu_indices[0];
      params.solver_iter_max = co.solver_iter_max;
      params.pcg_iter_max = co.pcg_iter_max;
      params.diag_init = co.diag_init;
      params.diag_min = co.diag_min;
      params.diag_scaling_up = co.diag_scaling_up;
      params.diag_scaling_down = co.diag_scaling_down;
      params.diag_exit_value = co.diag_exit_value;
      params.score_exit_value = co.score_exit_value;
      params.pcg_rel_error_exit = co.pcg_rel_error_exit;
      params.pcg_rel_score_exit = co.pcg_rel_score_exit;
      params.pcg_rel_decrease_min = co.pcg_rel_decrease_min;
      params.solver_rel_decrease_min = co.solver_rel_decrease_min;
    }

    const size_t device_id =
        static_cast<size_t>(gpu_index >= 0 ? gpu_index : FindBestCudaDevice());
    caspar::GraphSolver solver = CreateSolver(params, BuildSizing(), device_id);
    SetSolverData(solver);
    caspar::SolveResult result = solver.solve(/*print_progress=*/false);
    ReadSolverData(solver);
    WriteBack();

    auto caspar_summary = CasparBundleAdjustmentSummary::Create(result);
    summary.termination_type = caspar_summary->termination_type;
    summary.num_residuals = NumResiduals();
    summary.initial_score = caspar_summary->initial_score;
    summary.final_score = caspar_summary->final_score;
    summary.iteration_count = caspar_summary->iteration_count;
    summary.solve_time = caspar_summary->solve_time;
    summary.backend_message = "routed complete PINHOLE MPSFM objective through Caspar";
    return std::make_shared<MpsfmCasparBundleAdjustmentSummary>(summary);
  }

 private:
  IntrinsicsMode CameraIntrinsicsMode(const camera_t camera_id) const {
    if (config_.HasConstantCamIntrinsics(camera_id)) {
      return IntrinsicsMode::FIXED;
    }
    if (options_.refine_focal_length && options_.refine_extra_params &&
        options_.refine_principal_point) {
      return IntrinsicsMode::FULL;
    }
    if (options_.refine_focal_length && options_.refine_extra_params &&
        !options_.refine_principal_point) {
      return IntrinsicsMode::FOCAL_ONLY;
    }
    if (!options_.refine_focal_length && !options_.refine_extra_params &&
        !options_.refine_principal_point) {
      return IntrinsicsMode::FIXED;
    }
    LOG(FATAL_THROW)
        << "MPSFM Caspar PINHOLE intrinsics support full [fx,fy,cx,cy], "
           "focal-only with fixed principal point, or fully fixed intrinsics.";
    return IntrinsicsMode::FIXED;
  }

  bool IsFullPoseVariable(const Image& image) const {
    return options_.refine_rig_from_world &&
           !options_.constant_rig_from_world_rotation &&
           !config_.HasConstantRigFromWorldPose(image.FrameId()) &&
           image.IsRefInFrame();
  }

  bool IsTranslationVariable(const Image& image) const {
    return options_.refine_rig_from_world &&
           options_.constant_rig_from_world_rotation &&
           !config_.HasConstantRigFromWorldPose(image.FrameId()) &&
           image.IsRefInFrame();
  }

  bool IsPointVariable(const point3D_t point3D_id) const {
    return options_.refine_points3D && !config_.HasConstantPoint(point3D_id) &&
           !config_.IsIgnoredPoint(point3D_id);
  }

  void BuildNodes() {
    for (const image_t image_id : config_.Images()) {
      const Image& image = reconstruction_.Image(image_id);
      THROW_CHECK(image.IsRefInFrame())
          << "MPSFM Caspar currently requires ref-sensor images.";
      const Camera& camera = reconstruction_.Camera(image.CameraId());
      GetOrCreateIntrinsics(camera.camera_id, camera);
      if (IsFullPoseVariable(image)) {
        GetOrCreatePose(image.FrameId(), image.CamFromWorld());
      } else if (IsTranslationVariable(image)) {
        GetOrCreateTranslation(image.FrameId(), image.CamFromWorld());
      }
    }

    for (const auto& [point3D_id, point] : reconstruction_.Points3D()) {
      if (!config_.IsIgnoredPoint(point3D_id)) {
        GetOrCreatePoint(point3D_id, point);
      }
    }
    for (const auto& [image_id, value] : problem_.shift_scale) {
      GetOrCreateScale(image_id, value[1]);
    }
  }

  unsigned int GetOrCreatePose(const frame_t frame_id, const Rigid3d& pose) {
    auto [it, inserted] = frame_to_pose_.try_emplace(frame_id, pose_data_.size() / 7);
    if (inserted) {
      pose_idx_to_frame_[it->second] = frame_id;
      AppendPose(pose_data_, pose);
    }
    return static_cast<unsigned int>(it->second);
  }

  unsigned int GetOrCreateTranslation(const frame_t frame_id,
                                      const Rigid3d& pose) {
    auto [it, inserted] =
        frame_to_translation_.try_emplace(frame_id, translation_data_.size() / 3);
    if (inserted) {
      translation_idx_to_frame_[it->second] = frame_id;
      AppendTranslation(translation_data_, pose);
    }
    return static_cast<unsigned int>(it->second);
  }

  unsigned int GetOrCreatePoint(const point3D_t point3D_id,
                                const Point3D& point) {
    auto [it, inserted] =
        point_to_idx_.try_emplace(point3D_id, point_data_.size() / 3);
    if (inserted) {
      point_idx_to_id_[it->second] = point3D_id;
      AppendPoint(point_data_, point);
    }
    return static_cast<unsigned int>(it->second);
  }

  unsigned int GetOrCreateScale(const image_t image_id, const double scale) {
    auto [it, inserted] =
        image_to_scale_.try_emplace(image_id, scale_data_.size());
    if (inserted) {
      scale_idx_to_image_[it->second] = image_id;
      scale_data_.push_back(static_cast<StorageType>(scale));
    }
    return static_cast<unsigned int>(it->second);
  }

  unsigned int GetOrCreateIntrinsics(const camera_t camera_id,
                                     const Camera& camera) {
    const IntrinsicsMode mode = CameraIntrinsicsMode(camera_id);
    auto [it, inserted] =
        camera_to_intrinsics_.try_emplace(camera_id, focal_data_.size() / 2);
    if (inserted) {
      camera_idx_to_id_[it->second] = camera_id;
      AppendFocal(focal_data_, camera);
      AppendPrincipalPoint(principal_point_data_, camera);
      AppendCalib(calib_data_, camera);
      intrinsics_modes_[camera_id] = mode;
    } else {
      THROW_CHECK(intrinsics_modes_.at(camera_id) == mode);
    }
    return static_cast<unsigned int>(it->second);
  }

  void AddReprojectionFactor(const Image& image,
                             const Camera& camera,
                             const Point2D& point2D,
                             const Point3D& point3D) {
    const IntrinsicsMode mode = CameraIntrinsicsMode(camera.camera_id);
    const bool pose_var = IsFullPoseVariable(image);
    const bool translation_var = IsTranslationVariable(image);
    const bool point_var = IsPointVariable(point2D.point3D_id);
    const unsigned int intr_idx = GetOrCreateIntrinsics(camera.camera_id, camera);

    ReprojData* d = nullptr;
    if (translation_var) {
      if (mode == IntrinsicsMode::FULL && point_var) {
        d = &reproj_fixed_rotation_;
      } else if (mode == IntrinsicsMode::FULL && !point_var) {
        d = &reproj_fixed_rotation_fixed_point_;
      } else if (mode == IntrinsicsMode::FIXED && point_var) {
        d = &reproj_fixed_rotation_fixed_calib_;
      } else if (mode == IntrinsicsMode::FIXED && !point_var) {
        d = &reproj_fixed_rotation_fixed_calib_fixed_point_;
      } else if (mode == IntrinsicsMode::FOCAL_ONLY && point_var) {
        d = &reproj_split_fixed_rotation_fixed_pp_;
      } else if (mode == IntrinsicsMode::FOCAL_ONLY && !point_var) {
        d = &reproj_split_fixed_rotation_fixed_pp_fixed_point_;
      } else {
        LOG(FATAL_THROW)
            << "MPSFM Caspar fixed-rotation reprojection with focal-only "
               "intrinsics is not yet wired.";
      }
      d->translation.push_back(GetOrCreateTranslation(image.FrameId(),
                                                      image.CamFromWorld()));
      AppendRotation(d->const_rotation, image.CamFromWorld());
    } else if (pose_var) {
      if (mode == IntrinsicsMode::FULL && point_var) {
        d = &reproj_;
      } else if (mode == IntrinsicsMode::FULL && !point_var) {
        d = &reproj_fixed_point_;
      } else if (mode == IntrinsicsMode::FOCAL_ONLY && point_var) {
        d = &reproj_split_fixed_pp_;
      } else if (mode == IntrinsicsMode::FOCAL_ONLY && !point_var) {
        d = &reproj_split_fixed_pp_fixed_point_;
      } else if (mode == IntrinsicsMode::FIXED && point_var) {
        d = &reproj_split_fixed_focal_pp_;
      } else {
        d = &reproj_split_fixed_focal_pp_fixed_point_;
      }
      d->pose.push_back(GetOrCreatePose(image.FrameId(), image.CamFromWorld()));
    } else {
      if (mode == IntrinsicsMode::FULL && point_var) {
        d = &reproj_fixed_pose_;
      } else if (mode == IntrinsicsMode::FULL && !point_var) {
        d = &reproj_fixed_pose_fixed_point_;
      } else if (mode == IntrinsicsMode::FOCAL_ONLY && point_var) {
        d = &reproj_split_fixed_pose_fixed_pp_;
      } else if (mode == IntrinsicsMode::FOCAL_ONLY && !point_var) {
        d = &reproj_split_fixed_pose_fixed_pp_fixed_point_;
      } else if (mode == IntrinsicsMode::FIXED && point_var) {
        d = &reproj_split_fixed_pose_fixed_focal_pp_;
      } else {
        return;
      }
      AppendPose(d->const_pose, image.CamFromWorld());
    }

    if (mode == IntrinsicsMode::FULL) {
      d->calib.push_back(intr_idx);
    } else if (mode == IntrinsicsMode::FOCAL_ONLY) {
      d->focal.push_back(intr_idx);
      AppendPrincipalPoint(d->const_pp, camera);
    } else {
      AppendFocal(d->const_focal, camera);
      AppendPrincipalPoint(d->const_pp, camera);
      AppendCalib(d->const_calib, camera);
    }
    if (point_var) {
      d->point.push_back(GetOrCreatePoint(point2D.point3D_id, point3D));
    } else {
      AppendPoint(d->const_point, point3D);
    }
    d->pixel.push_back(static_cast<StorageType>(point2D.xy.x()));
    d->pixel.push_back(static_cast<StorageType>(point2D.xy.y()));
    AppendReprojectionWeightLoss(d->weight_loss, problem_.reprojection);
    ++d->num;
  }

  void BuildReprojectionFactors() {
    for (const image_t image_id : config_.Images()) {
      const Image& image = reconstruction_.Image(image_id);
      const Camera& camera = reconstruction_.Camera(image.CameraId());
      for (const Point2D& point2D : image.Points2D()) {
        if (!point2D.HasPoint3D() || config_.IsIgnoredPoint(point2D.point3D_id)) {
          continue;
        }
        AddReprojectionFactor(
            image, camera, point2D, reconstruction_.Point3D(point2D.point3D_id));
      }
    }
  }

  void AddDepthFactor(const MpsfmCasparDepthObservation& obs) {
    THROW_CHECK(reconstruction_.ExistsImage(obs.image_id));
    THROW_CHECK(reconstruction_.ExistsPoint3D(obs.point3D_id));
    const Image& image = reconstruction_.Image(obs.image_id);
    const Point3D& point = reconstruction_.Point3D(obs.point3D_id);
    const bool pose_var = IsFullPoseVariable(image);
    const bool translation_var = IsTranslationVariable(image);
    const bool point_var = IsPointVariable(obs.point3D_id);
    const unsigned int scale_idx =
        GetOrCreateScale(obs.image_id, problem_.shift_scale.at(obs.image_id)[1]);
    DepthData* d = nullptr;
    if (translation_var) {
      if (point_var) {
        d = &depth_fixed_rotation_;
      } else {
        d = &depth_fixed_rotation_fixed_point_;
      }
      d->translation.push_back(GetOrCreateTranslation(image.FrameId(),
                                                      image.CamFromWorld()));
      AppendRotation(d->const_rotation, image.CamFromWorld());
    } else if (pose_var) {
      d = point_var ? &depth_ : &depth_fixed_point_;
      d->pose.push_back(GetOrCreatePose(image.FrameId(), image.CamFromWorld()));
    } else {
      d = point_var ? &depth_fixed_pose_ : &depth_fixed_pose_fixed_point_;
      AppendPose(d->const_pose, image.CamFromWorld());
    }
    d->scale.push_back(scale_idx);
    if (point_var) {
      d->point.push_back(GetOrCreatePoint(obs.point3D_id, point));
    } else {
      AppendPoint(d->const_point, point);
    }
    d->log_depth.push_back(static_cast<StorageType>(std::log(std::max(obs.depth, 1e-10))));
    AppendRobustLoss(d->loss, obs.loss_type, obs.loss_scale, obs.sqrt_information);
    ++d->num;
  }

  void BuildDepthFactors() {
    for (const auto& obs : problem_.depth_observations) {
      THROW_CHECK(problem_.shift_scale.count(obs.image_id))
          << "Depth observation for image " << obs.image_id
          << " has no shift/scale entry.";
      if (!obs.gross_outlier) {
        AddDepthFactor(obs);
      }
    }
  }

  void BuildIntrinsicsFactors() {
    for (const auto& prior : problem_.intrinsics_priors) {
      const Camera& camera = reconstruction_.Camera(prior.camera_id);
      const unsigned int idx = GetOrCreateIntrinsics(prior.camera_id, camera);
      const IntrinsicsMode mode = CameraIntrinsicsMode(prior.camera_id);
      if (mode == IntrinsicsMode::FULL) {
        intr_prior_calib_.push_back(idx);
        for (int i = 0; i < 4; ++i) {
          intr_prior_prior_.push_back(static_cast<StorageType>(prior.prior[i]));
          intr_prior_inv_std_.push_back(static_cast<StorageType>(1.0 / prior.std_dev[i]));
        }
      } else if (mode == IntrinsicsMode::FOCAL_ONLY) {
        intr_prior_split_focal_.push_back(idx);
        for (int i = 0; i < 4; ++i) {
          intr_prior_split_prior_.push_back(static_cast<StorageType>(prior.prior[i]));
          intr_prior_split_inv_std_.push_back(static_cast<StorageType>(1.0 / prior.std_dev[i]));
        }
        AppendPrincipalPoint(intr_prior_split_pp_, camera);
      }
    }
    for (const auto& edge : problem_.intrinsics_random_walks) {
      const Camera& prev = reconstruction_.Camera(edge.prev_camera_id);
      const Camera& next = reconstruction_.Camera(edge.next_camera_id);
      const IntrinsicsMode prev_mode = CameraIntrinsicsMode(edge.prev_camera_id);
      const IntrinsicsMode next_mode = CameraIntrinsicsMode(edge.next_camera_id);
      THROW_CHECK(prev_mode == next_mode)
          << "MPSFM Caspar random-walk currently requires matching intrinsics modes.";
      const unsigned int prev_idx = GetOrCreateIntrinsics(edge.prev_camera_id, prev);
      const unsigned int next_idx = GetOrCreateIntrinsics(edge.next_camera_id, next);
      if (prev_mode == IntrinsicsMode::FULL) {
        intr_rw_prev_.push_back(prev_idx);
        intr_rw_next_.push_back(next_idx);
        for (int i = 0; i < 4; ++i) {
          intr_rw_inv_std_.push_back(static_cast<StorageType>(
              1.0 / std::sqrt(std::max(edge.variance_per_frame[i] * edge.frame_gap, 1e-10))));
        }
      } else if (prev_mode == IntrinsicsMode::FOCAL_ONLY) {
        intr_rw_split_prev_focal_.push_back(prev_idx);
        intr_rw_split_next_focal_.push_back(next_idx);
        for (int i = 0; i < 4; ++i) {
          intr_rw_split_inv_std_.push_back(static_cast<StorageType>(
              1.0 / std::sqrt(std::max(edge.variance_per_frame[i] * edge.frame_gap, 1e-10))));
        }
        AppendPrincipalPoint(intr_rw_split_prev_pp_, prev);
        AppendPrincipalPoint(intr_rw_split_next_pp_, next);
      }
    }
  }

  void BuildScalePriors() {
    for (const auto& prior : problem_.scale_priors) {
      THROW_CHECK(problem_.shift_scale.count(prior.image_id));
      scale_prior_scale_.push_back(
          GetOrCreateScale(prior.image_id, problem_.shift_scale.at(prior.image_id)[1]));
      scale_prior_inv_std_.push_back(static_cast<StorageType>(1.0 / prior.std_dev));
      AppendRobustLoss(scale_prior_loss_, prior.loss_type, prior.loss_scale, prior.magnitude);
    }
  }

  CasparSolverSizing BuildSizing() const {
    CasparSolverSizing sz;
    sz.num_depth_scales = scale_data_.size();
    sz.num_pinhole_calibs = focal_data_.size() / 2;
    sz.num_pinhole_poses = pose_data_.size() / 7;
    sz.num_pinhole_translations = translation_data_.size() / 3;
    sz.num_points = point_data_.size() / 3;
    sz.num_pinhole = reproj_.num;
    sz.num_pinhole_fixed_pose = reproj_fixed_pose_.num;
    sz.num_pinhole_fixed_point = reproj_fixed_point_.num;
    sz.num_pinhole_fixed_pose_fixed_point = reproj_fixed_pose_fixed_point_.num;
    sz.num_pinhole_split_fixed_principal_point = reproj_split_fixed_pp_.num;
    sz.num_pinhole_split_fixed_principal_point_fixed_point =
        reproj_split_fixed_pp_fixed_point_.num;
    sz.num_pinhole_split_fixed_pose_fixed_principal_point =
        reproj_split_fixed_pose_fixed_pp_.num;
    sz.num_pinhole_split_fixed_pose_fixed_principal_point_fixed_point =
        reproj_split_fixed_pose_fixed_pp_fixed_point_.num;
    sz.num_pinhole_split_fixed_focal_fixed_principal_point =
        reproj_split_fixed_focal_pp_.num;
    sz.num_pinhole_split_fixed_focal_fixed_principal_point_fixed_point =
        reproj_split_fixed_focal_pp_fixed_point_.num;
    sz.num_pinhole_split_fixed_pose_fixed_focal_fixed_principal_point =
        reproj_split_fixed_pose_fixed_focal_pp_.num;
    THROW_CHECK_EQ(reproj_split_fixed_pose_fixed_focal_pp_fixed_point_.num, 0);
    sz.num_pinhole_fixed_rotation = reproj_fixed_rotation_.num;
    sz.num_pinhole_fixed_rotation_fixed_calib =
        reproj_fixed_rotation_fixed_calib_.num;
    sz.num_pinhole_fixed_rotation_fixed_point =
        reproj_fixed_rotation_fixed_point_.num;
    sz.num_pinhole_fixed_rotation_fixed_calib_fixed_point =
        reproj_fixed_rotation_fixed_calib_fixed_point_.num;
    sz.num_pinhole_split_fixed_rotation_fixed_principal_point =
        reproj_split_fixed_rotation_fixed_pp_.num;
    sz.num_pinhole_split_fixed_rotation_fixed_principal_point_fixed_point =
        reproj_split_fixed_rotation_fixed_pp_fixed_point_.num;
    sz.num_pinhole_log_depth = depth_.num;
    sz.num_pinhole_log_depth_fixed_pose = depth_fixed_pose_.num;
    sz.num_pinhole_log_depth_fixed_point = depth_fixed_point_.num;
    sz.num_pinhole_log_depth_fixed_pose_fixed_point =
        depth_fixed_pose_fixed_point_.num;
    sz.num_pinhole_log_depth_fixed_rotation = depth_fixed_rotation_.num;
    sz.num_pinhole_log_depth_fixed_rotation_fixed_point =
        depth_fixed_rotation_fixed_point_.num;
    sz.num_pinhole_intrinsics_prior = intr_prior_calib_.size();
    sz.num_pinhole_split_intrinsics_prior_fixed_principal_point =
        intr_prior_split_focal_.size();
    sz.num_pinhole_intrinsics_random_walk = intr_rw_prev_.size();
    sz.num_pinhole_split_intrinsics_random_walk_fixed_prev_principal_point_fixed_next_principal_point =
        intr_rw_split_prev_focal_.size();
    sz.num_scale_prior = scale_prior_scale_.size();
    return sz;
  }

  size_t NumResiduals() const {
    return 2 * (reproj_.num + reproj_fixed_pose_.num + reproj_fixed_point_.num +
                reproj_fixed_pose_fixed_point_.num + reproj_split_fixed_pp_.num +
                reproj_split_fixed_pp_fixed_point_.num +
                reproj_split_fixed_pose_fixed_pp_.num +
                reproj_split_fixed_pose_fixed_pp_fixed_point_.num +
                reproj_split_fixed_focal_pp_.num +
                reproj_split_fixed_focal_pp_fixed_point_.num +
                reproj_split_fixed_pose_fixed_focal_pp_.num +
                reproj_split_fixed_pose_fixed_focal_pp_fixed_point_.num +
                reproj_fixed_rotation_.num +
                reproj_fixed_rotation_fixed_calib_.num +
                reproj_fixed_rotation_fixed_point_.num +
                reproj_fixed_rotation_fixed_calib_fixed_point_.num +
                reproj_split_fixed_rotation_fixed_pp_.num +
                reproj_split_fixed_rotation_fixed_pp_fixed_point_.num) +
           depth_.num + depth_fixed_pose_.num + depth_fixed_point_.num +
           depth_fixed_pose_fixed_point_.num + depth_fixed_rotation_.num +
           depth_fixed_rotation_fixed_point_.num + 4 * intr_prior_calib_.size() +
           4 * intr_prior_split_focal_.size() + 4 * intr_rw_prev_.size() +
           4 * intr_rw_split_prev_focal_.size() + scale_prior_scale_.size();
  }

  void SetReprojBase(caspar::GraphSolver& s,
                     const ReprojData& d,
                     const std::string& kind) {
    if (d.num == 0) return;
    if (kind == "pinhole") {
      s.SetPinholeNum(d.num);
      s.SetPinholePoseIndicesFromHost(d.pose.data(), d.num);
      s.SetPinholeCalibIndicesFromHost(d.calib.data(), d.num);
      s.SetPinholePointIndicesFromHost(d.point.data(), d.num);
      s.SetPinholePixelDataFromStackedHost(d.pixel.data(), 0, d.num);
      s.SetPinholeWeightLossDataFromStackedHost(d.weight_loss.data(), 0, d.num);
    }
  }

  void SetSolverData(caspar::GraphSolver& s) {
    if (!point_data_.empty()) s.SetPointNodesFromStackedHost(point_data_.data(), 0, point_data_.size() / 3);
    if (!pose_data_.empty()) s.SetPinholePoseNodesFromStackedHost(pose_data_.data(), 0, pose_data_.size() / 7);
    if (!translation_data_.empty()) s.SetPinholeTranslationNodesFromStackedHost(translation_data_.data(), 0, translation_data_.size() / 3);
    if (!focal_data_.empty()) {
      s.SetPinholeFocalNodesFromStackedHost(focal_data_.data(), 0, focal_data_.size() / 2);
      s.SetPinholePrincipalPointNodesFromStackedHost(principal_point_data_.data(), 0, principal_point_data_.size() / 2);
      s.SetPinholeCalibNodesFromStackedHost(calib_data_.data(), 0, calib_data_.size() / 4);
    }
    if (!scale_data_.empty()) s.SetDepthScaleNodesFromStackedHost(scale_data_.data(), 0, scale_data_.size());

    SetPinholeReprojectionFactors(s);
    SetDepthFactors(s);
    SetIntrinsicsFactors(s);
    SetScalePriorFactors(s);
    s.finish_indices();
  }

  void SetPinholeReprojectionFactors(caspar::GraphSolver& s);
  void SetDepthFactors(caspar::GraphSolver& s);
  void SetIntrinsicsFactors(caspar::GraphSolver& s);
  void SetScalePriorFactors(caspar::GraphSolver& s);

  void ReadSolverData(caspar::GraphSolver& s) {
    if (!point_data_.empty()) s.GetPointNodesToStackedHost(point_data_.data(), 0, point_data_.size() / 3);
    if (!pose_data_.empty()) s.GetPinholePoseNodesToStackedHost(pose_data_.data(), 0, pose_data_.size() / 7);
    if (!translation_data_.empty()) s.GetPinholeTranslationNodesToStackedHost(translation_data_.data(), 0, translation_data_.size() / 3);
    if (!focal_data_.empty()) {
      s.GetPinholeFocalNodesToStackedHost(focal_data_.data(), 0, focal_data_.size() / 2);
      s.GetPinholePrincipalPointNodesToStackedHost(principal_point_data_.data(), 0, principal_point_data_.size() / 2);
      s.GetPinholeCalibNodesToStackedHost(calib_data_.data(), 0, calib_data_.size() / 4);
    }
    if (!scale_data_.empty()) s.GetDepthScaleNodesToStackedHost(scale_data_.data(), 0, scale_data_.size());
  }

  void WriteBack() {
    for (const auto& [idx, point_id] : point_idx_to_id_) {
      if (!IsPointVariable(point_id)) continue;
      Point3D& point = reconstruction_.Point3D(point_id);
      point.xyz.x() = point_data_[idx * 3 + 0];
      point.xyz.y() = point_data_[idx * 3 + 1];
      point.xyz.z() = point_data_[idx * 3 + 2];
    }
    for (const auto& [idx, frame_id] : pose_idx_to_frame_) {
      Rigid3d& pose = reconstruction_.Frame(frame_id).RigFromWorld();
      pose.rotation().x() = pose_data_[idx * 7 + 0];
      pose.rotation().y() = pose_data_[idx * 7 + 1];
      pose.rotation().z() = pose_data_[idx * 7 + 2];
      pose.rotation().w() = pose_data_[idx * 7 + 3];
      pose.translation().x() = pose_data_[idx * 7 + 4];
      pose.translation().y() = pose_data_[idx * 7 + 5];
      pose.translation().z() = pose_data_[idx * 7 + 6];
      pose.rotation().normalize();
    }
    for (const auto& [idx, frame_id] : translation_idx_to_frame_) {
      Rigid3d& pose = reconstruction_.Frame(frame_id).RigFromWorld();
      pose.translation().x() = translation_data_[idx * 3 + 0];
      pose.translation().y() = translation_data_[idx * 3 + 1];
      pose.translation().z() = translation_data_[idx * 3 + 2];
    }
    for (const auto& [idx, camera_id] : camera_idx_to_id_) {
      const IntrinsicsMode mode = intrinsics_modes_.at(camera_id);
      if (mode == IntrinsicsMode::FIXED) continue;
      Camera& camera = reconstruction_.Camera(camera_id);
      if (mode == IntrinsicsMode::FULL) {
        camera.params[0] = calib_data_[idx * 4 + 0];
        camera.params[1] = calib_data_[idx * 4 + 1];
        camera.params[2] = calib_data_[idx * 4 + 2];
        camera.params[3] = calib_data_[idx * 4 + 3];
      } else {
        camera.params[0] = focal_data_[idx * 2 + 0];
        camera.params[1] = focal_data_[idx * 2 + 1];
      }
      THROW_CHECK(camera.VerifyParams());
    }
    for (const auto& [idx, image_id] : scale_idx_to_image_) {
      problem_.shift_scale[image_id][1] = scale_data_[idx];
    }
  }

  const BundleAdjustmentOptions& options_;
  const BundleAdjustmentConfig& config_;
  Reconstruction& reconstruction_;
  MpsfmCasparBundleAdjustmentProblem& problem_;

  std::unordered_map<frame_t, size_t> frame_to_pose_;
  std::unordered_map<size_t, frame_t> pose_idx_to_frame_;
  std::unordered_map<frame_t, size_t> frame_to_translation_;
  std::unordered_map<size_t, frame_t> translation_idx_to_frame_;
  std::unordered_map<point3D_t, size_t> point_to_idx_;
  std::unordered_map<size_t, point3D_t> point_idx_to_id_;
  std::unordered_map<image_t, size_t> image_to_scale_;
  std::unordered_map<size_t, image_t> scale_idx_to_image_;
  std::unordered_map<camera_t, size_t> camera_to_intrinsics_;
  std::unordered_map<size_t, camera_t> camera_idx_to_id_;
  std::unordered_map<camera_t, IntrinsicsMode> intrinsics_modes_;

  std::vector<StorageType> pose_data_;
  std::vector<StorageType> translation_data_;
  std::vector<StorageType> point_data_;
  std::vector<StorageType> scale_data_;
  std::vector<StorageType> focal_data_;
  std::vector<StorageType> principal_point_data_;
  std::vector<StorageType> calib_data_;

  ReprojData reproj_, reproj_fixed_pose_, reproj_fixed_point_,
      reproj_fixed_pose_fixed_point_, reproj_split_fixed_pp_,
      reproj_split_fixed_pp_fixed_point_, reproj_split_fixed_pose_fixed_pp_,
      reproj_split_fixed_pose_fixed_pp_fixed_point_,
      reproj_split_fixed_focal_pp_, reproj_split_fixed_focal_pp_fixed_point_,
      reproj_split_fixed_pose_fixed_focal_pp_,
      reproj_split_fixed_pose_fixed_focal_pp_fixed_point_,
      reproj_fixed_rotation_, reproj_fixed_rotation_fixed_calib_,
      reproj_fixed_rotation_fixed_point_,
      reproj_fixed_rotation_fixed_calib_fixed_point_,
      reproj_split_fixed_rotation_fixed_pp_,
      reproj_split_fixed_rotation_fixed_pp_fixed_point_;

  DepthData depth_, depth_fixed_pose_, depth_fixed_point_,
      depth_fixed_pose_fixed_point_, depth_fixed_rotation_,
      depth_fixed_rotation_fixed_point_;

  std::vector<unsigned int> intr_prior_calib_, intr_prior_split_focal_;
  std::vector<StorageType> intr_prior_prior_, intr_prior_inv_std_,
      intr_prior_split_prior_, intr_prior_split_inv_std_, intr_prior_split_pp_;
  std::vector<unsigned int> intr_rw_prev_, intr_rw_next_,
      intr_rw_split_prev_focal_, intr_rw_split_next_focal_;
  std::vector<StorageType> intr_rw_inv_std_, intr_rw_split_inv_std_,
      intr_rw_split_prev_pp_, intr_rw_split_next_pp_;
  std::vector<unsigned int> scale_prior_scale_;
  std::vector<StorageType> scale_prior_inv_std_, scale_prior_loss_;
};

void MpsfmCasparGraph::SetPinholeReprojectionFactors(caspar::GraphSolver& s) {
  if (reproj_.num) {
    s.SetPinholeNum(reproj_.num);
    s.SetPinholePoseIndicesFromHost(reproj_.pose.data(), reproj_.num);
    s.SetPinholeCalibIndicesFromHost(reproj_.calib.data(), reproj_.num);
    s.SetPinholePointIndicesFromHost(reproj_.point.data(), reproj_.num);
    s.SetPinholePixelDataFromStackedHost(reproj_.pixel.data(), 0, reproj_.num);
    s.SetPinholeWeightLossDataFromStackedHost(
        reproj_.weight_loss.data(), 0, reproj_.num);
  }
  if (reproj_fixed_pose_.num) {
    s.SetPinholeFixedPoseNum(reproj_fixed_pose_.num);
    s.SetPinholeFixedPoseCalibIndicesFromHost(
        reproj_fixed_pose_.calib.data(), reproj_fixed_pose_.num);
    s.SetPinholeFixedPosePointIndicesFromHost(
        reproj_fixed_pose_.point.data(), reproj_fixed_pose_.num);
    s.SetPinholeFixedPosePoseDataFromStackedHost(
        reproj_fixed_pose_.const_pose.data(), 0, reproj_fixed_pose_.num);
    s.SetPinholeFixedPosePixelDataFromStackedHost(
        reproj_fixed_pose_.pixel.data(), 0, reproj_fixed_pose_.num);
    s.SetPinholeFixedPoseWeightLossDataFromStackedHost(
        reproj_fixed_pose_.weight_loss.data(), 0, reproj_fixed_pose_.num);
  }
  if (reproj_fixed_point_.num) {
    s.SetPinholeFixedPointNum(reproj_fixed_point_.num);
    s.SetPinholeFixedPointPoseIndicesFromHost(
        reproj_fixed_point_.pose.data(), reproj_fixed_point_.num);
    s.SetPinholeFixedPointCalibIndicesFromHost(
        reproj_fixed_point_.calib.data(), reproj_fixed_point_.num);
    s.SetPinholeFixedPointPointDataFromStackedHost(
        reproj_fixed_point_.const_point.data(), 0, reproj_fixed_point_.num);
    s.SetPinholeFixedPointPixelDataFromStackedHost(
        reproj_fixed_point_.pixel.data(), 0, reproj_fixed_point_.num);
    s.SetPinholeFixedPointWeightLossDataFromStackedHost(
        reproj_fixed_point_.weight_loss.data(), 0, reproj_fixed_point_.num);
  }
  if (reproj_fixed_pose_fixed_point_.num) {
    s.SetPinholeFixedPoseFixedPointNum(reproj_fixed_pose_fixed_point_.num);
    s.SetPinholeFixedPoseFixedPointCalibIndicesFromHost(
        reproj_fixed_pose_fixed_point_.calib.data(),
        reproj_fixed_pose_fixed_point_.num);
    s.SetPinholeFixedPoseFixedPointPoseDataFromStackedHost(
        reproj_fixed_pose_fixed_point_.const_pose.data(),
        0,
        reproj_fixed_pose_fixed_point_.num);
    s.SetPinholeFixedPoseFixedPointPointDataFromStackedHost(
        reproj_fixed_pose_fixed_point_.const_point.data(),
        0,
        reproj_fixed_pose_fixed_point_.num);
    s.SetPinholeFixedPoseFixedPointPixelDataFromStackedHost(
        reproj_fixed_pose_fixed_point_.pixel.data(),
        0,
        reproj_fixed_pose_fixed_point_.num);
    s.SetPinholeFixedPoseFixedPointWeightLossDataFromStackedHost(
        reproj_fixed_pose_fixed_point_.weight_loss.data(),
        0,
        reproj_fixed_pose_fixed_point_.num);
  }
  if (reproj_split_fixed_pp_.num) {
    s.SetPinholeSplitFixedPrincipalPointNum(reproj_split_fixed_pp_.num);
    s.SetPinholeSplitFixedPrincipalPointPoseIndicesFromHost(
        reproj_split_fixed_pp_.pose.data(), reproj_split_fixed_pp_.num);
    s.SetPinholeSplitFixedPrincipalPointFocalIndicesFromHost(
        reproj_split_fixed_pp_.focal.data(), reproj_split_fixed_pp_.num);
    s.SetPinholeSplitFixedPrincipalPointPointIndicesFromHost(
        reproj_split_fixed_pp_.point.data(), reproj_split_fixed_pp_.num);
    s.SetPinholeSplitFixedPrincipalPointPrincipalPointDataFromStackedHost(
        reproj_split_fixed_pp_.const_pp.data(), 0, reproj_split_fixed_pp_.num);
    s.SetPinholeSplitFixedPrincipalPointPixelDataFromStackedHost(
        reproj_split_fixed_pp_.pixel.data(), 0, reproj_split_fixed_pp_.num);
    s.SetPinholeSplitFixedPrincipalPointWeightLossDataFromStackedHost(
        reproj_split_fixed_pp_.weight_loss.data(), 0, reproj_split_fixed_pp_.num);
  }
  if (reproj_split_fixed_pp_fixed_point_.num) {
    s.SetPinholeSplitFixedPrincipalPointFixedPointNum(
        reproj_split_fixed_pp_fixed_point_.num);
    s.SetPinholeSplitFixedPrincipalPointFixedPointPoseIndicesFromHost(
        reproj_split_fixed_pp_fixed_point_.pose.data(),
        reproj_split_fixed_pp_fixed_point_.num);
    s.SetPinholeSplitFixedPrincipalPointFixedPointFocalIndicesFromHost(
        reproj_split_fixed_pp_fixed_point_.focal.data(),
        reproj_split_fixed_pp_fixed_point_.num);
    s.SetPinholeSplitFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
        reproj_split_fixed_pp_fixed_point_.const_pp.data(),
        0,
        reproj_split_fixed_pp_fixed_point_.num);
    s.SetPinholeSplitFixedPrincipalPointFixedPointPointDataFromStackedHost(
        reproj_split_fixed_pp_fixed_point_.const_point.data(),
        0,
        reproj_split_fixed_pp_fixed_point_.num);
    s.SetPinholeSplitFixedPrincipalPointFixedPointPixelDataFromStackedHost(
        reproj_split_fixed_pp_fixed_point_.pixel.data(),
        0,
        reproj_split_fixed_pp_fixed_point_.num);
    s.SetPinholeSplitFixedPrincipalPointFixedPointWeightLossDataFromStackedHost(
        reproj_split_fixed_pp_fixed_point_.weight_loss.data(),
        0,
        reproj_split_fixed_pp_fixed_point_.num);
  }
  if (reproj_split_fixed_pose_fixed_pp_.num) {
    s.SetPinholeSplitFixedPoseFixedPrincipalPointNum(
        reproj_split_fixed_pose_fixed_pp_.num);
    s.SetPinholeSplitFixedPoseFixedPrincipalPointFocalIndicesFromHost(
        reproj_split_fixed_pose_fixed_pp_.focal.data(),
        reproj_split_fixed_pose_fixed_pp_.num);
    s.SetPinholeSplitFixedPoseFixedPrincipalPointPointIndicesFromHost(
        reproj_split_fixed_pose_fixed_pp_.point.data(),
        reproj_split_fixed_pose_fixed_pp_.num);
    s.SetPinholeSplitFixedPoseFixedPrincipalPointPoseDataFromStackedHost(
        reproj_split_fixed_pose_fixed_pp_.const_pose.data(),
        0,
        reproj_split_fixed_pose_fixed_pp_.num);
    s.SetPinholeSplitFixedPoseFixedPrincipalPointPrincipalPointDataFromStackedHost(
        reproj_split_fixed_pose_fixed_pp_.const_pp.data(),
        0,
        reproj_split_fixed_pose_fixed_pp_.num);
    s.SetPinholeSplitFixedPoseFixedPrincipalPointPixelDataFromStackedHost(
        reproj_split_fixed_pose_fixed_pp_.pixel.data(),
        0,
        reproj_split_fixed_pose_fixed_pp_.num);
    s.SetPinholeSplitFixedPoseFixedPrincipalPointWeightLossDataFromStackedHost(
        reproj_split_fixed_pose_fixed_pp_.weight_loss.data(),
        0,
        reproj_split_fixed_pose_fixed_pp_.num);
  }
  if (reproj_split_fixed_pose_fixed_pp_fixed_point_.num) {
    s.SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointNum(
        reproj_split_fixed_pose_fixed_pp_fixed_point_.num);
    s.SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointFocalIndicesFromHost(
        reproj_split_fixed_pose_fixed_pp_fixed_point_.focal.data(),
        reproj_split_fixed_pose_fixed_pp_fixed_point_.num);
    s.SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointPoseDataFromStackedHost(
        reproj_split_fixed_pose_fixed_pp_fixed_point_.const_pose.data(),
        0,
        reproj_split_fixed_pose_fixed_pp_fixed_point_.num);
    s.SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
        reproj_split_fixed_pose_fixed_pp_fixed_point_.const_pp.data(),
        0,
        reproj_split_fixed_pose_fixed_pp_fixed_point_.num);
    s.SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointPointDataFromStackedHost(
        reproj_split_fixed_pose_fixed_pp_fixed_point_.const_point.data(),
        0,
        reproj_split_fixed_pose_fixed_pp_fixed_point_.num);
    s.SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointPixelDataFromStackedHost(
        reproj_split_fixed_pose_fixed_pp_fixed_point_.pixel.data(),
        0,
        reproj_split_fixed_pose_fixed_pp_fixed_point_.num);
    s.SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointWeightLossDataFromStackedHost(
        reproj_split_fixed_pose_fixed_pp_fixed_point_.weight_loss.data(),
        0,
        reproj_split_fixed_pose_fixed_pp_fixed_point_.num);
  }
  if (reproj_split_fixed_focal_pp_.num) {
    s.SetPinholeSplitFixedFocalFixedPrincipalPointNum(
        reproj_split_fixed_focal_pp_.num);
    s.SetPinholeSplitFixedFocalFixedPrincipalPointPoseIndicesFromHost(
        reproj_split_fixed_focal_pp_.pose.data(), reproj_split_fixed_focal_pp_.num);
    s.SetPinholeSplitFixedFocalFixedPrincipalPointPointIndicesFromHost(
        reproj_split_fixed_focal_pp_.point.data(), reproj_split_fixed_focal_pp_.num);
    s.SetPinholeSplitFixedFocalFixedPrincipalPointFocalDataFromStackedHost(
        reproj_split_fixed_focal_pp_.const_focal.data(),
        0,
        reproj_split_fixed_focal_pp_.num);
    s.SetPinholeSplitFixedFocalFixedPrincipalPointPrincipalPointDataFromStackedHost(
        reproj_split_fixed_focal_pp_.const_pp.data(),
        0,
        reproj_split_fixed_focal_pp_.num);
    s.SetPinholeSplitFixedFocalFixedPrincipalPointPixelDataFromStackedHost(
        reproj_split_fixed_focal_pp_.pixel.data(),
        0,
        reproj_split_fixed_focal_pp_.num);
    s.SetPinholeSplitFixedFocalFixedPrincipalPointWeightLossDataFromStackedHost(
        reproj_split_fixed_focal_pp_.weight_loss.data(),
        0,
        reproj_split_fixed_focal_pp_.num);
  }
  if (reproj_split_fixed_focal_pp_fixed_point_.num) {
    s.SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointNum(
        reproj_split_fixed_focal_pp_fixed_point_.num);
    s.SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointPoseIndicesFromHost(
        reproj_split_fixed_focal_pp_fixed_point_.pose.data(),
        reproj_split_fixed_focal_pp_fixed_point_.num);
    s.SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointFocalDataFromStackedHost(
        reproj_split_fixed_focal_pp_fixed_point_.const_focal.data(),
        0,
        reproj_split_fixed_focal_pp_fixed_point_.num);
    s.SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
        reproj_split_fixed_focal_pp_fixed_point_.const_pp.data(),
        0,
        reproj_split_fixed_focal_pp_fixed_point_.num);
    s.SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointPointDataFromStackedHost(
        reproj_split_fixed_focal_pp_fixed_point_.const_point.data(),
        0,
        reproj_split_fixed_focal_pp_fixed_point_.num);
    s.SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointPixelDataFromStackedHost(
        reproj_split_fixed_focal_pp_fixed_point_.pixel.data(),
        0,
        reproj_split_fixed_focal_pp_fixed_point_.num);
    s.SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointWeightLossDataFromStackedHost(
        reproj_split_fixed_focal_pp_fixed_point_.weight_loss.data(),
        0,
        reproj_split_fixed_focal_pp_fixed_point_.num);
  }
  if (reproj_split_fixed_pose_fixed_focal_pp_.num) {
    s.SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointNum(
        reproj_split_fixed_pose_fixed_focal_pp_.num);
    s.SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointPointIndicesFromHost(
        reproj_split_fixed_pose_fixed_focal_pp_.point.data(),
        reproj_split_fixed_pose_fixed_focal_pp_.num);
    s.SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointPoseDataFromStackedHost(
        reproj_split_fixed_pose_fixed_focal_pp_.const_pose.data(),
        0,
        reproj_split_fixed_pose_fixed_focal_pp_.num);
    s.SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointFocalDataFromStackedHost(
        reproj_split_fixed_pose_fixed_focal_pp_.const_focal.data(),
        0,
        reproj_split_fixed_pose_fixed_focal_pp_.num);
    s.SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointPrincipalPointDataFromStackedHost(
        reproj_split_fixed_pose_fixed_focal_pp_.const_pp.data(),
        0,
        reproj_split_fixed_pose_fixed_focal_pp_.num);
    s.SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointPixelDataFromStackedHost(
        reproj_split_fixed_pose_fixed_focal_pp_.pixel.data(),
        0,
        reproj_split_fixed_pose_fixed_focal_pp_.num);
    s.SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointWeightLossDataFromStackedHost(
        reproj_split_fixed_pose_fixed_focal_pp_.weight_loss.data(),
        0,
        reproj_split_fixed_pose_fixed_focal_pp_.num);
  }
  if (reproj_fixed_rotation_.num) {
    s.SetPinholeFixedRotationNum(reproj_fixed_rotation_.num);
    s.SetPinholeFixedRotationTranslationIndicesFromHost(
        reproj_fixed_rotation_.translation.data(), reproj_fixed_rotation_.num);
    s.SetPinholeFixedRotationCalibIndicesFromHost(
        reproj_fixed_rotation_.calib.data(), reproj_fixed_rotation_.num);
    s.SetPinholeFixedRotationPointIndicesFromHost(
        reproj_fixed_rotation_.point.data(), reproj_fixed_rotation_.num);
    s.SetPinholeFixedRotationRotationDataFromStackedHost(
        reproj_fixed_rotation_.const_rotation.data(), 0, reproj_fixed_rotation_.num);
    s.SetPinholeFixedRotationPixelDataFromStackedHost(
        reproj_fixed_rotation_.pixel.data(), 0, reproj_fixed_rotation_.num);
    s.SetPinholeFixedRotationWeightLossDataFromStackedHost(
        reproj_fixed_rotation_.weight_loss.data(), 0, reproj_fixed_rotation_.num);
  }
  if (reproj_fixed_rotation_fixed_calib_.num) {
    s.SetPinholeFixedRotationFixedCalibNum(
        reproj_fixed_rotation_fixed_calib_.num);
    s.SetPinholeFixedRotationFixedCalibTranslationIndicesFromHost(
        reproj_fixed_rotation_fixed_calib_.translation.data(),
        reproj_fixed_rotation_fixed_calib_.num);
    s.SetPinholeFixedRotationFixedCalibPointIndicesFromHost(
        reproj_fixed_rotation_fixed_calib_.point.data(),
        reproj_fixed_rotation_fixed_calib_.num);
    s.SetPinholeFixedRotationFixedCalibRotationDataFromStackedHost(
        reproj_fixed_rotation_fixed_calib_.const_rotation.data(),
        0,
        reproj_fixed_rotation_fixed_calib_.num);
    s.SetPinholeFixedRotationFixedCalibPixelDataFromStackedHost(
        reproj_fixed_rotation_fixed_calib_.pixel.data(),
        0,
        reproj_fixed_rotation_fixed_calib_.num);
    s.SetPinholeFixedRotationFixedCalibCalibDataFromStackedHost(
        reproj_fixed_rotation_fixed_calib_.const_calib.data(),
        0,
        reproj_fixed_rotation_fixed_calib_.num);
    s.SetPinholeFixedRotationFixedCalibWeightLossDataFromStackedHost(
        reproj_fixed_rotation_fixed_calib_.weight_loss.data(),
        0,
        reproj_fixed_rotation_fixed_calib_.num);
  }
  if (reproj_fixed_rotation_fixed_point_.num) {
    s.SetPinholeFixedRotationFixedPointNum(
        reproj_fixed_rotation_fixed_point_.num);
    s.SetPinholeFixedRotationFixedPointTranslationIndicesFromHost(
        reproj_fixed_rotation_fixed_point_.translation.data(),
        reproj_fixed_rotation_fixed_point_.num);
    s.SetPinholeFixedRotationFixedPointCalibIndicesFromHost(
        reproj_fixed_rotation_fixed_point_.calib.data(),
        reproj_fixed_rotation_fixed_point_.num);
    s.SetPinholeFixedRotationFixedPointRotationDataFromStackedHost(
        reproj_fixed_rotation_fixed_point_.const_rotation.data(),
        0,
        reproj_fixed_rotation_fixed_point_.num);
    s.SetPinholeFixedRotationFixedPointPointDataFromStackedHost(
        reproj_fixed_rotation_fixed_point_.const_point.data(),
        0,
        reproj_fixed_rotation_fixed_point_.num);
    s.SetPinholeFixedRotationFixedPointPixelDataFromStackedHost(
        reproj_fixed_rotation_fixed_point_.pixel.data(),
        0,
        reproj_fixed_rotation_fixed_point_.num);
    s.SetPinholeFixedRotationFixedPointWeightLossDataFromStackedHost(
        reproj_fixed_rotation_fixed_point_.weight_loss.data(),
        0,
        reproj_fixed_rotation_fixed_point_.num);
  }
  if (reproj_fixed_rotation_fixed_calib_fixed_point_.num) {
    s.SetPinholeFixedRotationFixedCalibFixedPointNum(
        reproj_fixed_rotation_fixed_calib_fixed_point_.num);
    s.SetPinholeFixedRotationFixedCalibFixedPointTranslationIndicesFromHost(
        reproj_fixed_rotation_fixed_calib_fixed_point_.translation.data(),
        reproj_fixed_rotation_fixed_calib_fixed_point_.num);
    s.SetPinholeFixedRotationFixedCalibFixedPointRotationDataFromStackedHost(
        reproj_fixed_rotation_fixed_calib_fixed_point_.const_rotation.data(),
        0,
        reproj_fixed_rotation_fixed_calib_fixed_point_.num);
    s.SetPinholeFixedRotationFixedCalibFixedPointPixelDataFromStackedHost(
        reproj_fixed_rotation_fixed_calib_fixed_point_.pixel.data(),
        0,
        reproj_fixed_rotation_fixed_calib_fixed_point_.num);
    s.SetPinholeFixedRotationFixedCalibFixedPointCalibDataFromStackedHost(
        reproj_fixed_rotation_fixed_calib_fixed_point_.const_calib.data(),
        0,
        reproj_fixed_rotation_fixed_calib_fixed_point_.num);
    s.SetPinholeFixedRotationFixedCalibFixedPointPointDataFromStackedHost(
        reproj_fixed_rotation_fixed_calib_fixed_point_.const_point.data(),
        0,
        reproj_fixed_rotation_fixed_calib_fixed_point_.num);
    s.SetPinholeFixedRotationFixedCalibFixedPointWeightLossDataFromStackedHost(
        reproj_fixed_rotation_fixed_calib_fixed_point_.weight_loss.data(),
        0,
        reproj_fixed_rotation_fixed_calib_fixed_point_.num);
  }
  if (reproj_split_fixed_rotation_fixed_pp_.num) {
    s.SetPinholeSplitFixedRotationFixedPrincipalPointNum(
        reproj_split_fixed_rotation_fixed_pp_.num);
    s.SetPinholeSplitFixedRotationFixedPrincipalPointTranslationIndicesFromHost(
        reproj_split_fixed_rotation_fixed_pp_.translation.data(),
        reproj_split_fixed_rotation_fixed_pp_.num);
    s.SetPinholeSplitFixedRotationFixedPrincipalPointFocalIndicesFromHost(
        reproj_split_fixed_rotation_fixed_pp_.focal.data(),
        reproj_split_fixed_rotation_fixed_pp_.num);
    s.SetPinholeSplitFixedRotationFixedPrincipalPointPointIndicesFromHost(
        reproj_split_fixed_rotation_fixed_pp_.point.data(),
        reproj_split_fixed_rotation_fixed_pp_.num);
    s.SetPinholeSplitFixedRotationFixedPrincipalPointRotationDataFromStackedHost(
        reproj_split_fixed_rotation_fixed_pp_.const_rotation.data(),
        0,
        reproj_split_fixed_rotation_fixed_pp_.num);
    s.SetPinholeSplitFixedRotationFixedPrincipalPointPrincipalPointDataFromStackedHost(
        reproj_split_fixed_rotation_fixed_pp_.const_pp.data(),
        0,
        reproj_split_fixed_rotation_fixed_pp_.num);
    s.SetPinholeSplitFixedRotationFixedPrincipalPointPixelDataFromStackedHost(
        reproj_split_fixed_rotation_fixed_pp_.pixel.data(),
        0,
        reproj_split_fixed_rotation_fixed_pp_.num);
    s.SetPinholeSplitFixedRotationFixedPrincipalPointWeightLossDataFromStackedHost(
        reproj_split_fixed_rotation_fixed_pp_.weight_loss.data(),
        0,
        reproj_split_fixed_rotation_fixed_pp_.num);
  }
  if (reproj_split_fixed_rotation_fixed_pp_fixed_point_.num) {
    s.SetPinholeSplitFixedRotationFixedPrincipalPointFixedPointNum(
        reproj_split_fixed_rotation_fixed_pp_fixed_point_.num);
    s.SetPinholeSplitFixedRotationFixedPrincipalPointFixedPointTranslationIndicesFromHost(
        reproj_split_fixed_rotation_fixed_pp_fixed_point_.translation.data(),
        reproj_split_fixed_rotation_fixed_pp_fixed_point_.num);
    s.SetPinholeSplitFixedRotationFixedPrincipalPointFixedPointFocalIndicesFromHost(
        reproj_split_fixed_rotation_fixed_pp_fixed_point_.focal.data(),
        reproj_split_fixed_rotation_fixed_pp_fixed_point_.num);
    s.SetPinholeSplitFixedRotationFixedPrincipalPointFixedPointRotationDataFromStackedHost(
        reproj_split_fixed_rotation_fixed_pp_fixed_point_.const_rotation.data(),
        0,
        reproj_split_fixed_rotation_fixed_pp_fixed_point_.num);
    s.SetPinholeSplitFixedRotationFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
        reproj_split_fixed_rotation_fixed_pp_fixed_point_.const_pp.data(),
        0,
        reproj_split_fixed_rotation_fixed_pp_fixed_point_.num);
    s.SetPinholeSplitFixedRotationFixedPrincipalPointFixedPointPointDataFromStackedHost(
        reproj_split_fixed_rotation_fixed_pp_fixed_point_.const_point.data(),
        0,
        reproj_split_fixed_rotation_fixed_pp_fixed_point_.num);
    s.SetPinholeSplitFixedRotationFixedPrincipalPointFixedPointPixelDataFromStackedHost(
        reproj_split_fixed_rotation_fixed_pp_fixed_point_.pixel.data(),
        0,
        reproj_split_fixed_rotation_fixed_pp_fixed_point_.num);
    s.SetPinholeSplitFixedRotationFixedPrincipalPointFixedPointWeightLossDataFromStackedHost(
        reproj_split_fixed_rotation_fixed_pp_fixed_point_.weight_loss.data(),
        0,
        reproj_split_fixed_rotation_fixed_pp_fixed_point_.num);
  }
  // Fully fixed point/fixed-calib uncommon variants are currently guarded by
  // graph sizing; add explicit failure if they appear before silently dropping.
  THROW_CHECK_EQ(reproj_split_fixed_pose_fixed_focal_pp_fixed_point_.num, 0);
}

void MpsfmCasparGraph::SetDepthFactors(caspar::GraphSolver& s) {
  if (depth_.num) {
    s.SetPinholeLogDepthNum(depth_.num);
    s.SetPinholeLogDepthPoseIndicesFromHost(depth_.pose.data(), depth_.num);
    s.SetPinholeLogDepthScaleIndicesFromHost(depth_.scale.data(), depth_.num);
    s.SetPinholeLogDepthPointIndicesFromHost(depth_.point.data(), depth_.num);
    s.SetPinholeLogDepthLogDepthDataFromStackedHost(
        depth_.log_depth.data(), 0, depth_.num);
    s.SetPinholeLogDepthLossDataFromStackedHost(depth_.loss.data(), 0, depth_.num);
  }
  if (depth_fixed_pose_.num) {
    s.SetPinholeLogDepthFixedPoseNum(depth_fixed_pose_.num);
    s.SetPinholeLogDepthFixedPoseScaleIndicesFromHost(
        depth_fixed_pose_.scale.data(), depth_fixed_pose_.num);
    s.SetPinholeLogDepthFixedPosePointIndicesFromHost(
        depth_fixed_pose_.point.data(), depth_fixed_pose_.num);
    s.SetPinholeLogDepthFixedPoseLogDepthDataFromStackedHost(
        depth_fixed_pose_.log_depth.data(), 0, depth_fixed_pose_.num);
    s.SetPinholeLogDepthFixedPosePoseDataFromStackedHost(
        depth_fixed_pose_.const_pose.data(), 0, depth_fixed_pose_.num);
    s.SetPinholeLogDepthFixedPoseLossDataFromStackedHost(
        depth_fixed_pose_.loss.data(), 0, depth_fixed_pose_.num);
  }
  if (depth_fixed_point_.num) {
    s.SetPinholeLogDepthFixedPointNum(depth_fixed_point_.num);
    s.SetPinholeLogDepthFixedPointPoseIndicesFromHost(
        depth_fixed_point_.pose.data(), depth_fixed_point_.num);
    s.SetPinholeLogDepthFixedPointScaleIndicesFromHost(
        depth_fixed_point_.scale.data(), depth_fixed_point_.num);
    s.SetPinholeLogDepthFixedPointLogDepthDataFromStackedHost(
        depth_fixed_point_.log_depth.data(), 0, depth_fixed_point_.num);
    s.SetPinholeLogDepthFixedPointPointDataFromStackedHost(
        depth_fixed_point_.const_point.data(), 0, depth_fixed_point_.num);
    s.SetPinholeLogDepthFixedPointLossDataFromStackedHost(
        depth_fixed_point_.loss.data(), 0, depth_fixed_point_.num);
  }
  if (depth_fixed_pose_fixed_point_.num) {
    s.SetPinholeLogDepthFixedPoseFixedPointNum(
        depth_fixed_pose_fixed_point_.num);
    s.SetPinholeLogDepthFixedPoseFixedPointScaleIndicesFromHost(
        depth_fixed_pose_fixed_point_.scale.data(),
        depth_fixed_pose_fixed_point_.num);
    s.SetPinholeLogDepthFixedPoseFixedPointLogDepthDataFromStackedHost(
        depth_fixed_pose_fixed_point_.log_depth.data(),
        0,
        depth_fixed_pose_fixed_point_.num);
    s.SetPinholeLogDepthFixedPoseFixedPointPoseDataFromStackedHost(
        depth_fixed_pose_fixed_point_.const_pose.data(),
        0,
        depth_fixed_pose_fixed_point_.num);
    s.SetPinholeLogDepthFixedPoseFixedPointPointDataFromStackedHost(
        depth_fixed_pose_fixed_point_.const_point.data(),
        0,
        depth_fixed_pose_fixed_point_.num);
    s.SetPinholeLogDepthFixedPoseFixedPointLossDataFromStackedHost(
        depth_fixed_pose_fixed_point_.loss.data(),
        0,
        depth_fixed_pose_fixed_point_.num);
  }
  if (depth_fixed_rotation_.num) {
    s.SetPinholeLogDepthFixedRotationNum(depth_fixed_rotation_.num);
    s.SetPinholeLogDepthFixedRotationTranslationIndicesFromHost(
        depth_fixed_rotation_.translation.data(), depth_fixed_rotation_.num);
    s.SetPinholeLogDepthFixedRotationScaleIndicesFromHost(
        depth_fixed_rotation_.scale.data(), depth_fixed_rotation_.num);
    s.SetPinholeLogDepthFixedRotationPointIndicesFromHost(
        depth_fixed_rotation_.point.data(), depth_fixed_rotation_.num);
    s.SetPinholeLogDepthFixedRotationRotationDataFromStackedHost(
        depth_fixed_rotation_.const_rotation.data(), 0, depth_fixed_rotation_.num);
    s.SetPinholeLogDepthFixedRotationLogDepthDataFromStackedHost(
        depth_fixed_rotation_.log_depth.data(), 0, depth_fixed_rotation_.num);
    s.SetPinholeLogDepthFixedRotationLossDataFromStackedHost(
        depth_fixed_rotation_.loss.data(), 0, depth_fixed_rotation_.num);
  }
  if (depth_fixed_rotation_fixed_point_.num) {
    s.SetPinholeLogDepthFixedRotationFixedPointNum(
        depth_fixed_rotation_fixed_point_.num);
    s.SetPinholeLogDepthFixedRotationFixedPointTranslationIndicesFromHost(
        depth_fixed_rotation_fixed_point_.translation.data(),
        depth_fixed_rotation_fixed_point_.num);
    s.SetPinholeLogDepthFixedRotationFixedPointScaleIndicesFromHost(
        depth_fixed_rotation_fixed_point_.scale.data(),
        depth_fixed_rotation_fixed_point_.num);
    s.SetPinholeLogDepthFixedRotationFixedPointRotationDataFromStackedHost(
        depth_fixed_rotation_fixed_point_.const_rotation.data(),
        0,
        depth_fixed_rotation_fixed_point_.num);
    s.SetPinholeLogDepthFixedRotationFixedPointLogDepthDataFromStackedHost(
        depth_fixed_rotation_fixed_point_.log_depth.data(),
        0,
        depth_fixed_rotation_fixed_point_.num);
    s.SetPinholeLogDepthFixedRotationFixedPointPointDataFromStackedHost(
        depth_fixed_rotation_fixed_point_.const_point.data(),
        0,
        depth_fixed_rotation_fixed_point_.num);
    s.SetPinholeLogDepthFixedRotationFixedPointLossDataFromStackedHost(
        depth_fixed_rotation_fixed_point_.loss.data(),
        0,
        depth_fixed_rotation_fixed_point_.num);
  }
}

void MpsfmCasparGraph::SetIntrinsicsFactors(caspar::GraphSolver& s) {
  if (!intr_prior_calib_.empty()) {
    const size_t n = intr_prior_calib_.size();
    s.SetPinholeIntrinsicsPriorNum(n);
    s.SetPinholeIntrinsicsPriorCalibIndicesFromHost(intr_prior_calib_.data(), n);
    s.SetPinholeIntrinsicsPriorPriorDataFromStackedHost(
        intr_prior_prior_.data(), 0, n);
    s.SetPinholeIntrinsicsPriorInvStdDataFromStackedHost(
        intr_prior_inv_std_.data(), 0, n);
  }
  if (!intr_prior_split_focal_.empty()) {
    const size_t n = intr_prior_split_focal_.size();
    s.SetPinholeSplitIntrinsicsPriorFixedPrincipalPointNum(n);
    s.SetPinholeSplitIntrinsicsPriorFixedPrincipalPointFocalIndicesFromHost(
        intr_prior_split_focal_.data(), n);
    s.SetPinholeSplitIntrinsicsPriorFixedPrincipalPointPriorDataFromStackedHost(
        intr_prior_split_prior_.data(), 0, n);
    s.SetPinholeSplitIntrinsicsPriorFixedPrincipalPointInvStdDataFromStackedHost(
        intr_prior_split_inv_std_.data(), 0, n);
    s.SetPinholeSplitIntrinsicsPriorFixedPrincipalPointPrincipalPointDataFromStackedHost(
        intr_prior_split_pp_.data(), 0, n);
  }
  if (!intr_rw_prev_.empty()) {
    const size_t n = intr_rw_prev_.size();
    s.SetPinholeIntrinsicsRandomWalkNum(n);
    s.SetPinholeIntrinsicsRandomWalkPrevCalibIndicesFromHost(
        intr_rw_prev_.data(), n);
    s.SetPinholeIntrinsicsRandomWalkNextCalibIndicesFromHost(
        intr_rw_next_.data(), n);
    s.SetPinholeIntrinsicsRandomWalkInvStdDataFromStackedHost(
        intr_rw_inv_std_.data(), 0, n);
  }
  if (!intr_rw_split_prev_focal_.empty()) {
    const size_t n = intr_rw_split_prev_focal_.size();
    s.SetPinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextPrincipalPointNum(n);
    s.SetPinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextPrincipalPointPrevFocalIndicesFromHost(
        intr_rw_split_prev_focal_.data(), n);
    s.SetPinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextPrincipalPointNextFocalIndicesFromHost(
        intr_rw_split_next_focal_.data(), n);
    s.SetPinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextPrincipalPointInvStdDataFromStackedHost(
        intr_rw_split_inv_std_.data(), 0, n);
    s.SetPinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextPrincipalPointPrevPrincipalPointDataFromStackedHost(
        intr_rw_split_prev_pp_.data(), 0, n);
    s.SetPinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextPrincipalPointNextPrincipalPointDataFromStackedHost(
        intr_rw_split_next_pp_.data(), 0, n);
  }
}

void MpsfmCasparGraph::SetScalePriorFactors(caspar::GraphSolver& s) {
  if (scale_prior_scale_.empty()) return;
  const size_t n = scale_prior_scale_.size();
  s.SetScalePriorNum(n);
  s.SetScalePriorScaleIndicesFromHost(scale_prior_scale_.data(), n);
  s.SetScalePriorInvStdDataFromStackedHost(scale_prior_inv_std_.data(), 0, n);
  s.SetScalePriorLossDataFromStackedHost(scale_prior_loss_.data(), 0, n);
}
#endif

}  // namespace

std::string MpsfmCasparBundleAdjustmentSummary::BriefReport() const {
  std::ostringstream report;
  report << "MPSFM Caspar bundle adjustment report\n";
  report << "    Termination: " << termination_type << "\n";
  report << "    Residuals: " << num_residuals << "\n";
  report << "    Factors: reprojection=" << num_reprojection_factors
         << ", depth=" << num_depth_factors
         << ", intrinsics_prior=" << num_intrinsics_prior_factors
         << ", intrinsics_random_walk=" << num_intrinsics_random_walk_factors
         << ", scale_prior=" << num_scale_prior_factors << "\n";
  report << "    Solver iterations: " << iteration_count << "\n";
  report << "    Score: initial=" << initial_score << ", final=" << final_score
         << "\n";
  report << "    Timing: construction=" << construction_time
         << "s, solve=" << solve_time << "s";
  if (!backend_message.empty()) {
    report << "\n    Backend: " << backend_message;
  }
  return report.str();
}

std::shared_ptr<MpsfmCasparBundleAdjustmentSummary>
SolveMpsfmCasparBundleAdjustment(
    const BundleAdjustmentOptions& options,
    const BundleAdjustmentConfig& config,
    Reconstruction& reconstruction,
    MpsfmCasparBundleAdjustmentProblem& problem) {
  auto summary = std::make_shared<MpsfmCasparBundleAdjustmentSummary>();
  summary->num_reprojection_factors =
      CountReprojectionFactors(config, reconstruction);
  summary->num_depth_factors =
      static_cast<int>(problem.depth_observations.size());
  summary->num_intrinsics_prior_factors =
      static_cast<int>(problem.intrinsics_priors.size());
  summary->num_intrinsics_random_walk_factors =
      static_cast<int>(problem.intrinsics_random_walks.size());
  summary->num_scale_prior_factors =
      static_cast<int>(problem.scale_priors.size());

  ValidateInputs(options, config, reconstruction, problem);

#ifdef CASPAR_ENABLED
  Timer timer;
  timer.Start();
  MpsfmCasparGraph graph(options, config, reconstruction, problem);
  timer.Pause();
  summary->construction_time = timer.ElapsedSeconds();

  timer.Restart();
  std::shared_ptr<MpsfmCasparBundleAdjustmentSummary> caspar_summary =
      graph.Solve(*summary);
  timer.Pause();
  summary->solve_time = timer.ElapsedSeconds();

  summary->termination_type = caspar_summary->termination_type;
  summary->num_residuals = caspar_summary->num_residuals;
  summary->initial_score = caspar_summary->initial_score;
  summary->final_score = caspar_summary->final_score;
  summary->iteration_count = caspar_summary->iteration_count;
  summary->backend_message = caspar_summary->backend_message;
  return summary;
#else
  LOG(FATAL_THROW)
      << "solve_mpsfm_caspar_bundle_adjustment requires COLMAP to be built "
         "with CASPAR_ENABLED.";
#endif
}

}  // namespace colmap
