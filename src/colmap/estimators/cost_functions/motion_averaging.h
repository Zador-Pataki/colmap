
#pragma once

#include "colmap/util/logging.h"

#include <Eigen/Core>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace colmap {

// Computes the error between a translation direction and the direction formed
// from two positions such that: t_ij - scale * (p_j - p_i) is minimized.
// The positions can either be two camera centers or one camera center and one
// 3D point.
// Reference: Zhuang et al., "Baseline Desensitizing In Translation Averaging",
// CVPR 2018.
struct BATAPairwiseDirectionCostFunctor {
  explicit BATAPairwiseDirectionCostFunctor(
      const Eigen::Vector3d& pos2_from_pos1_dir)
      : pos2_from_pos1_dir_(pos2_from_pos1_dir) {}

  template <typename T>
  bool operator()(const T* pos1,
                  const T* pos2,
                  const T* scale,
                  T* residuals) const {
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals_vec(residuals);
    residuals_vec = pos2_from_pos1_dir_.cast<T>() -
                    scale[0] * (Eigen::Map<const Eigen::Matrix<T, 3, 1>>(pos2) -
                                Eigen::Map<const Eigen::Matrix<T, 3, 1>>(pos1));
    return true;
  }

  static ceres::CostFunction* Create(
      const Eigen::Vector3d& pos2_from_pos1_dir) {
    return (
        new ceres::
            AutoDiffCostFunction<BATAPairwiseDirectionCostFunctor, 3, 3, 3, 1>(
                new BATAPairwiseDirectionCostFunctor(pos2_from_pos1_dir)));
  }

  const Eigen::Vector3d pos2_from_pos1_dir_;
};

// Computes the error between a translation direction and the direction formed
// from a camera (c) and 3D point (p) with constant rig extrinsics, such that:
// t_ij - scale * (p - c + t_rig) is minimized.
struct RigBATAPairwiseDirectionConstantRigCostFunctor {
  RigBATAPairwiseDirectionConstantRigCostFunctor(
      const Eigen::Vector3d& cam_from_point3D_dir,
      const Eigen::Vector3d& cam_from_rig_translation)
      : cam_from_point3D_dir_(cam_from_point3D_dir),
        cam_from_rig_translation_(cam_from_rig_translation) {}

  template <typename T>
  bool operator()(const T* point3D,
                  const T* rig_in_world,
                  const T* scale,
                  T* residuals) const {
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals_vec(residuals);
    residuals_vec =
        cam_from_point3D_dir_.cast<T>() -
        scale[0] * (Eigen::Map<const Eigen::Matrix<T, 3, 1>>(point3D) -
                    Eigen::Map<const Eigen::Matrix<T, 3, 1>>(rig_in_world) +
                    cam_from_rig_translation_.cast<T>());
    return true;
  }

  static ceres::CostFunction* Create(
      const Eigen::Vector3d& cam_from_point3D_dir,
      const Eigen::Vector3d& cam_from_rig_translation) {
    return (new ceres::AutoDiffCostFunction<
            RigBATAPairwiseDirectionConstantRigCostFunctor,
            3,
            3,
            3,
            1>(new RigBATAPairwiseDirectionConstantRigCostFunctor(
        cam_from_point3D_dir, cam_from_rig_translation)));
  }

  const Eigen::Vector3d cam_from_point3D_dir_;
  const Eigen::Vector3d cam_from_rig_translation_;
};

// Computes the error between a translation direction and the direction formed
// from a camera (c) and 3D point (p) with variable rig extrinsics, such that:
// t_ij - scale * (p - c + t_rig) is minimized.
struct RigBATAPairwiseDirectionCostFunctor {
  RigBATAPairwiseDirectionCostFunctor(
      const Eigen::Vector3d& cam_from_point3D_dir,
      const Eigen::Quaterniond& rig_from_world_rot)
      : cam_from_point3D_dir_(cam_from_point3D_dir),
        world_from_rig_rot_(rig_from_world_rot.inverse()) {}

  template <typename T>
  bool operator()(const T* point3D,
                  const T* rig_in_world,
                  const T* cam_in_rig,
                  const T* scale,
                  T* residuals) const {
    const Eigen::Matrix<T, 3, 1> cam_from_rig_translation =
        world_from_rig_rot_.cast<T>() *
        Eigen::Map<const Eigen::Matrix<T, 3, 1>>(cam_in_rig);

    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals_vec(residuals);
    residuals_vec =
        cam_from_point3D_dir_.cast<T>() -
        scale[0] * (Eigen::Map<const Eigen::Matrix<T, 3, 1>>(point3D) -
                    Eigen::Map<const Eigen::Matrix<T, 3, 1>>(rig_in_world) -
                    cam_from_rig_translation);
    return true;
  }

  static ceres::CostFunction* Create(
      const Eigen::Vector3d& cam_from_point3D_dir,
      const Eigen::Quaterniond& rig_from_world_rot) {
    return (new ceres::AutoDiffCostFunction<RigBATAPairwiseDirectionCostFunctor,
                                            3,
                                            3,
                                            3,
                                            3,
                                            1>(
        new RigBATAPairwiseDirectionCostFunctor(cam_from_point3D_dir,
                                                rig_from_world_rot)));
  }

  const Eigen::Vector3d cam_from_point3D_dir_;
  const Eigen::Quaterniond world_from_rig_rot_;
};

// Glomap-fork addition. Like ``BATAPairwiseDirectionCostFunctor`` but
// rotates the residual into camera frame via constant ``rotation`` and
// applies anisotropic per-axis weighting ``(1/sigma_x, 1/sigma_y,
// 1/sigma_z)``. ``GlobalPositioner`` substitutes this for the unweighted
// variant whenever ``image.angular_stddevs[fid]`` is populated. Caller
// computes ``sigma_z = (sigma_x + sigma_y) / 2`` (or pulls from
// ``image.angular_stddevs_z[fid]``) when constructing.
struct WeightedBATADirectionalError {
  WeightedBATADirectionalError(const Eigen::Vector3d& translation_obs,
                               const Eigen::Quaterniond& rotation,
                               double sigma_x,
                               double sigma_y,
                               double sigma_z)
      : translation_obs_(translation_obs),
        rotation_(rotation),
        inv_sigma_x_(1.0 / sigma_x),
        inv_sigma_y_(1.0 / sigma_y),
        inv_sigma_z_(1.0 / sigma_z) {
    THROW_CHECK_GT(sigma_x, 0.0);
    THROW_CHECK_GT(sigma_y, 0.0);
    THROW_CHECK_GT(sigma_z, 0.0);
  }

  template <typename T>
  bool operator()(const T* position1,
                  const T* position2,
                  const T* scale,
                  T* residuals) const {
    using Vec3T = Eigen::Matrix<T, 3, 1>;
    const Vec3T r_world = translation_obs_.cast<T>() -
                          scale[0] * (Eigen::Map<const Vec3T>(position2) -
                                      Eigen::Map<const Vec3T>(position1));
    const Vec3T r_cam = rotation_.cast<T>() * r_world;
    residuals[0] = T(inv_sigma_x_) * r_cam[0];
    residuals[1] = T(inv_sigma_y_) * r_cam[1];
    residuals[2] = T(inv_sigma_z_) * r_cam[2];
    return true;
  }

  static ceres::CostFunction* Create(const Eigen::Vector3d& translation_obs,
                                     const Eigen::Quaterniond& rotation,
                                     double sigma_x,
                                     double sigma_y,
                                     double sigma_z) {
    return new ceres::AutoDiffCostFunction<WeightedBATADirectionalError,
                                           3,
                                           3,
                                           3,
                                           1>(
        new WeightedBATADirectionalError(
            translation_obs, rotation, sigma_x, sigma_y, sigma_z));
  }

  const Eigen::Vector3d translation_obs_;
  const Eigen::Quaterniond rotation_;
  const double inv_sigma_x_;
  const double inv_sigma_y_;
  const double inv_sigma_z_;
};

}  // namespace colmap
