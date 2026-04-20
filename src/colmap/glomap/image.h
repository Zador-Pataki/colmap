#pragma once

#include "colmap/glomap/gravity_info.h"
#include "colmap/glomap/types.h"

#include <string>
#include <vector>

#include <Eigen/Core>

namespace colmap::glomap {

struct Image {
  Image() : image_id(-1), file_name("") {}
  Image(image_t img_id, camera_t cam_id, std::string file_name)
      : image_id(img_id),
        file_name(std::move(file_name)),
        camera_id(cam_id),
        log_scale(0),
        log_scale_stddev(0) {}

  // Basic information.
  // NOTE: image_id and file_name are ctor-only; no assignment (unordered_map
  // callers must use emplace / try_emplace / insert({id, Image(...)})).
  const image_t image_id;
  const std::string file_name;

  // The id of the camera.
  camera_t camera_id;

  // Whether the image is within the largest connected component.
  bool is_registered = false;
  int cluster_id = -1;

  // The pose of the image (transformation from world to camera).
  colmap::Rigid3d cam_from_world;

  // Gravity information.
  GravityInfo gravity_info;

  // Distorted feature points in pixels.
  std::vector<Eigen::Vector2d> features;
  // Normalized feature rays; populated by UndistortImages.
  std::vector<Eigen::Vector3d> features_undist;

  // --- Monocular depth priors (constant per image load) ---
  std::vector<double> depth_priors;
  std::vector<double> depth_prior_stddevs;
  std::vector<bool> depth_prior_validity;

  // --- Inlier flag for second GP (set from Python after first GP) ---
  // If true, use trivial loss for this observation in second GP.
  std::vector<bool> is_inlier;

  // --- MDRP depth outlier flag (set from Python after MDRP) ---
  // If true, use robust loss for depth constraint of this observation.
  std::vector<bool> is_depth_outlier;

  // --- Track-anchor flag (set from Python after track establishment) ---
  // If true, use loss_normal_geometry_trackstart for geometry constraint.
  std::vector<bool> is_track_anchor;

  // --- Hard exclusion flag (set from Python after GP stage) ---
  // If true, this observation is NOT added to the problem at all. Used to
  // filter egregious outliers between GP stages.
  std::vector<bool> is_excluded;

  // --- Angular uncertainties (anisotropic weighting) ---
  // 2D vector (sigma_x, sigma_y) in radians, typically (pix_sigma_x / fx,
  // pix_sigma_y / fy).
  std::vector<Eigen::Vector2d> angular_stddevs;

  // --- Mahalanobis weighting (full-covariance support) ---
  // Cholesky factor of XY precision matrix: stored as (L00, L10, L11) for
  // lower-triangular 2x2 with L * L^T = Sigma^{-1}.
  std::vector<Eigen::Vector3d> angular_cholesky_xy;
  // Z-component stddev (sqrt of average trace of XY covariance).
  std::vector<double> angular_stddevs_z;

  // --- Per-image scale parameters (optimizable) ---
  double log_scale;
  double log_scale_stddev;

  // --- Gravity prior uncertainty (radians; 0 = use option default) ---
  double gravity_sigma = 0;

  // Methods.
  Eigen::Vector3d Center() const;
};

}  // namespace colmap::glomap
