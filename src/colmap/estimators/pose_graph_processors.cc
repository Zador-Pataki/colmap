// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
// (BSD-3-Clause license, see LICENSE)

#include "colmap/estimators/pose_graph_processors.h"

#include "colmap/estimators/two_view_geometry.h"
#include "colmap/geometry/essential_matrix.h"
#include "colmap/scene/two_view_geometry.h"
#include "colmap/util/threading.h"

#include <glog/logging.h>
#include <vector>

namespace colmap {
namespace {

// Compute F from cam2_from_cam1 and camera intrinsics.
// F = K2^{-T} * [t]_x * R * K1^{-1}
Eigen::Matrix3d FundamentalFromPoseAndCameras(
    const Camera& camera1,
    const Camera& camera2,
    const Rigid3d& cam2_from_cam1) {
  const Eigen::Vector3d t = cam2_from_cam1.translation();
  const Eigen::Matrix3d R = cam2_from_cam1.rotation().toRotationMatrix();
  Eigen::Matrix3d t_x;
  t_x << 0, -t.z(), t.y(),
         t.z(), 0, -t.x(),
         -t.y(), t.x(), 0;
  const Eigen::Matrix3d E = t_x * R;
  const Eigen::Matrix3d K1 = camera1.CalibrationMatrix();
  const Eigen::Matrix3d K2 = camera2.CalibrationMatrix();
  return K2.inverse().transpose() * E * K1.inverse();
}

// Extract 2D feature positions from an Image's points2D_.
std::vector<Eigen::Vector2d> ExtractFeatures(const Image& image) {
  std::vector<Eigen::Vector2d> features;
  features.reserve(image.NumPoints2D());
  for (const auto& p : image.Points2D()) {
    features.push_back(p.xy);
  }
  return features;
}

}  // namespace

void DecomposeRelPose(PoseGraph& pose_graph,
                      const std::unordered_map<camera_t, Camera>& cameras,
                      const std::unordered_map<image_t, Image>& images) {
  std::vector<image_pair_t> pair_ids;
  for (const auto& [pid, edge] : pose_graph.Edges()) {
    if (!edge.valid) continue;
    const auto [id1, id2] = PairIdToImagePair(pid);
    if (images.count(id1) == 0 || images.count(id2) == 0) continue;
    const camera_t cam_id1 = images.at(id1).CameraId();
    const camera_t cam_id2 = images.at(id2).CameraId();
    if (!cameras.at(cam_id1).has_prior_focal_length) continue;
    if (!cameras.at(cam_id2).has_prior_focal_length) continue;
    pair_ids.push_back(pid);
  }

  LOG(INFO) << "DecomposeRelPose: " << pair_ids.size() << " pairs";

  ThreadPool thread_pool(ThreadPool::kMaxNumThreads);

  for (const image_pair_t pid : pair_ids) {
    thread_pool.AddTask([&, pid]() {
      PoseGraph::Edge& edge = pose_graph.Edges().at(pid);
      const auto [id1, id2] = PairIdToImagePair(pid);
      const Camera& cam1 = cameras.at(images.at(id1).CameraId());
      const Camera& cam2 = cameras.at(images.at(id2).CameraId());

      const std::vector<Eigen::Vector2d> features1 = ExtractFeatures(images.at(id1));
      const std::vector<Eigen::Vector2d> features2 = ExtractFeatures(images.at(id2));
      if (features1.empty() || features2.empty()) return;

      TwoViewGeometry geom;
      geom.config = edge.config;
      if (edge.E.has_value()) geom.E = *edge.E;
      if (edge.F.has_value()) geom.F = *edge.F;
      if (edge.H.has_value()) geom.H = *edge.H;
      geom.cam2_from_cam1 = edge.cam2_from_cam1;

      if (!EstimateTwoViewGeometryPose(cam1, features1, cam2, features2, &geom)) {
        return;
      }

      // Handle planar: upgrade config but keep existing pose.
      if (edge.config == TwoViewGeometry::PLANAR &&
          cam1.has_prior_focal_length && cam2.has_prior_focal_length) {
        edge.config = TwoViewGeometry::CALIBRATED;
        return;
      }

      if (!cam1.has_prior_focal_length || !cam2.has_prior_focal_length) return;

      edge.config = geom.config;
      if (geom.cam2_from_cam1.has_value()) {
        edge.cam2_from_cam1 = *geom.cam2_from_cam1;
        if (edge.cam2_from_cam1.translation().norm() > 1e-10) {
          edge.cam2_from_cam1.translation() =
              edge.cam2_from_cam1.translation().normalized();
        }
      }
    });
  }

  thread_pool.Wait();
}

void UpdateImagePairsConfig(
    PoseGraph& pose_graph,
    const std::unordered_map<camera_t, Camera>& cameras,
    const std::unordered_map<image_t, Image>& images) {
  // Count per-camera: (total pairs with prior-FL, calibrated pairs).
  std::unordered_map<camera_t, std::pair<int, int>> camera_counter;

  for (const auto& [pid, edge] : pose_graph.Edges()) {
    if (!edge.valid) continue;
    const auto [id1, id2] = PairIdToImagePair(pid);
    if (images.count(id1) == 0 || images.count(id2) == 0) continue;
    const camera_t cam_id1 = images.at(id1).CameraId();
    const camera_t cam_id2 = images.at(id2).CameraId();
    if (!cameras.at(cam_id1).has_prior_focal_length ||
        !cameras.at(cam_id2).has_prior_focal_length) continue;

    const bool is_calib =
        (edge.config == TwoViewGeometry::CALIBRATED);
    const bool is_uncalib =
        (edge.config == TwoViewGeometry::UNCALIBRATED);

    if (is_calib) {
      camera_counter[cam_id1].first++;
      camera_counter[cam_id2].first++;
      camera_counter[cam_id1].second++;
      camera_counter[cam_id2].second++;
    } else if (is_uncalib) {
      camera_counter[cam_id1].first++;
      camera_counter[cam_id2].first++;
    }
  }

  // Camera is "valid" if >50% of its pairs are CALIBRATED.
  std::unordered_map<camera_t, bool> camera_valid;
  for (const auto& [cam_id, counts] : camera_counter) {
    camera_valid[cam_id] =
        (counts.first > 0 && counts.second * 1.0 / counts.first > 0.5);
  }

  // Upgrade UNCALIBRATED pairs whose both cameras are valid.
  for (auto& [pid, edge] : pose_graph.Edges()) {
    if (!edge.valid) continue;
    if (edge.config != TwoViewGeometry::UNCALIBRATED) continue;
    const auto [id1, id2] = PairIdToImagePair(pid);
    if (images.count(id1) == 0 || images.count(id2) == 0) continue;
    const camera_t cam_id1 = images.at(id1).CameraId();
    const camera_t cam_id2 = images.at(id2).CameraId();

    if (!camera_valid.count(cam_id1) || !camera_valid[cam_id1]) continue;
    if (!camera_valid.count(cam_id2) || !camera_valid[cam_id2]) continue;

    edge.config = TwoViewGeometry::CALIBRATED;
    edge.F = FundamentalFromPoseAndCameras(
        cameras.at(cam_id1), cameras.at(cam_id2), edge.cam2_from_cam1);
  }
}

void UndistortImages(const std::unordered_map<camera_t, Camera>& cameras,
                     std::unordered_map<image_t, Image>& images,
                     bool clean_points) {
  std::vector<image_t> to_process;
  for (auto& [img_id, img] : images) {
    if (img.features_undist.size() == img.NumPoints2D() && !clean_points)
      continue;
    to_process.push_back(img_id);
  }

  LOG(INFO) << "UndistortImages: " << to_process.size() << " images";

  ThreadPool thread_pool(ThreadPool::kMaxNumThreads);

  for (const image_t img_id : to_process) {
    Image& img = images.at(img_id);
    const Camera& camera = cameras.at(img.CameraId());
    const size_t n = img.NumPoints2D();

    thread_pool.AddTask([&img, &camera, n]() {
      img.features_undist.clear();
      img.features_undist.reserve(n);
      for (size_t i = 0; i < n; ++i) {
        img.features_undist.emplace_back(
            camera.CamFromImg(img.Point2D(i).xy).value().homogeneous().normalized());
      }
    });
  }

  thread_pool.Wait();
}

}  // namespace colmap
