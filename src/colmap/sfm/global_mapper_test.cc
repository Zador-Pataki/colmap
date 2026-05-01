#include "colmap/sfm/global_mapper.h"

#include "colmap/feature/types.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/scene/database_cache.h"
#include "colmap/scene/frame.h"
#include "colmap/scene/image.h"
#include "colmap/scene/reconstruction_matchers.h"
#include "colmap/scene/synthetic.h"
#include "colmap/scene/two_view_geometry.h"
#include "colmap/sensor/models.h"
#include "colmap/sensor/rig.h"
#include "colmap/util/testing.h"

#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

namespace colmap {
namespace {

// TODO(jsch): Add tests for pose priors.

std::shared_ptr<DatabaseCache> CreateDatabaseCache(const Database& database) {
  DatabaseCache::Options options;
  return DatabaseCache::Create(database, options);
}

std::shared_ptr<DatabaseCache> CreateThreeViewTrackCache(
    const point2D_t num_tracks) {
  auto cache = std::make_shared<DatabaseCache>();

  Camera camera = Camera::CreateFromModelId(
      /*camera_id=*/1, CameraModelId::kSimplePinhole, 1000.0, 1024, 768);
  const sensor_t camera_sensor = camera.SensorId();
  cache->AddCamera(std::move(camera));

  Rig rig;
  rig.SetRigId(1);
  rig.AddRefSensor(camera_sensor);
  cache->AddRig(std::move(rig));

  for (image_t image_id = 1; image_id <= 3; ++image_id) {
    Image image;
    image.SetImageId(image_id);
    image.SetName(std::to_string(image_id) + ".jpg");
    image.SetCameraId(1);
    image.SetFrameId(image_id);

    std::vector<Eigen::Vector2d> keypoints;
    keypoints.reserve(num_tracks);
    for (point2D_t point2D_idx = 0; point2D_idx < num_tracks; ++point2D_idx) {
      keypoints.emplace_back(100.0 * point2D_idx, 100.0 * point2D_idx);
    }
    image.SetPoints2D(keypoints);

    Frame frame;
    frame.SetFrameId(image_id);
    frame.SetRigId(1);
    frame.AddDataId(image.DataId());
    frame.SetRigFromWorld(Rigid3d());
    cache->AddFrame(std::move(frame));
    cache->AddImage(std::move(image));
  }

  auto add_image_pair = [&](const image_t image_id1, const image_t image_id2) {
    TwoViewGeometry two_view_geometry;
    two_view_geometry.config = TwoViewGeometry::CALIBRATED;
    two_view_geometry.cam2_from_cam1 = Rigid3d();
    two_view_geometry.inlier_matches.reserve(num_tracks);
    for (point2D_t point2D_idx = 0; point2D_idx < num_tracks; ++point2D_idx) {
      two_view_geometry.inlier_matches.emplace_back(point2D_idx, point2D_idx);
    }
    cache->CorrespondenceGraph()->AddTwoViewGeometry(
        image_id1, image_id2, std::move(two_view_geometry));
  };
  add_image_pair(1, 2);
  add_image_pair(1, 3);
  add_image_pair(2, 3);

  return cache;
}

TEST(GlobalMapper, WithoutNoise) {
  SetPRNGSeed(1);
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  auto reconstruction = std::make_shared<Reconstruction>();

  GlobalMapper global_mapper(CreateDatabaseCache(*database));
  global_mapper.BeginReconstruction(reconstruction);

  global_mapper.Solve(GlobalMapperOptions());

  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(*reconstruction,
                                 /*max_rotation_error_deg=*/1e-2,
                                 /*max_proj_center_error=*/1e-4));
}

TEST(GlobalMapper, WithoutNoiseWithNonTrivialKnownRig) {
  SetPRNGSeed(1);
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 2;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.sensor_from_rig_translation_stddev =
      0.1;                                                         // No noise
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 5.;  // No noise
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  auto reconstruction = std::make_shared<Reconstruction>();

  GlobalMapper global_mapper(CreateDatabaseCache(*database));
  global_mapper.BeginReconstruction(reconstruction);

  global_mapper.Solve(GlobalMapperOptions());

  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(*reconstruction,
                                 /*max_rotation_error_deg=*/1e-2,
                                 /*max_proj_center_error=*/1e-4));
}

TEST(GlobalMapper, WithoutNoiseWithNonTrivialUnknownRig) {
  SetPRNGSeed(1);
  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 3;
  synthetic_dataset_options.num_frames_per_rig = 7;
  synthetic_dataset_options.num_points3D = 50;
  synthetic_dataset_options.sensor_from_rig_translation_stddev =
      0.1;                                                         // No noise
  synthetic_dataset_options.sensor_from_rig_rotation_stddev = 5.;  // No noise

  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());

  auto reconstruction = std::make_shared<Reconstruction>();

  GlobalMapper global_mapper(CreateDatabaseCache(*database));
  global_mapper.BeginReconstruction(reconstruction);

  // Set the rig sensors to be unknown
  for (const auto& [rig_id, rig] : reconstruction->Rigs()) {
    for (const auto& [sensor_id, sensor] : rig.NonRefSensors()) {
      if (sensor.has_value()) {
        reconstruction->Rig(rig_id).ResetSensorFromRig(sensor_id);
      }
    }
  }

  global_mapper.Solve(GlobalMapperOptions());

  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(*reconstruction,
                                 /*max_rotation_error_deg=*/1e-2,
                                 /*max_proj_center_error=*/1e-4));
}

TEST(GlobalMapper, WithNoiseAndOutliers) {
  SetPRNGSeed(1);

  const auto database_path = CreateTestDir() / "database.db";

  auto database = Database::Open(database_path);
  Reconstruction gt_reconstruction;
  SyntheticDatasetOptions synthetic_dataset_options;
  synthetic_dataset_options.num_rigs = 2;
  synthetic_dataset_options.num_cameras_per_rig = 1;
  synthetic_dataset_options.num_frames_per_rig = 4;
  synthetic_dataset_options.num_points3D = 100;
  synthetic_dataset_options.inlier_match_ratio = 0.7;
  synthetic_dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(
      synthetic_dataset_options, &gt_reconstruction, database.get());
  SyntheticNoiseOptions synthetic_noise_options;
  synthetic_noise_options.point2D_stddev = 0.5;
  SynthesizeNoise(synthetic_noise_options, &gt_reconstruction, database.get());

  auto reconstruction = std::make_shared<Reconstruction>();

  GlobalMapper global_mapper(CreateDatabaseCache(*database));
  global_mapper.BeginReconstruction(reconstruction);

  global_mapper.Solve(GlobalMapperOptions());

  EXPECT_THAT(gt_reconstruction,
              ReconstructionNear(*reconstruction,
                                 /*max_rotation_error_deg=*/1e-1,
                                 /*max_proj_center_error=*/1e-1,
                                 /*max_scale_error=*/std::nullopt,
                                 /*num_obs_tolerance=*/0.02));
}

TEST(GlobalMapper, EstablishTracksAppliesRequiredTracksPerView) {
  auto reconstruction = std::make_shared<Reconstruction>();
  GlobalMapper global_mapper(CreateThreeViewTrackCache(/*num_tracks=*/5));
  global_mapper.BeginReconstruction(reconstruction);

  GlobalMapperOptions options;
  options.track_required_tracks_per_view = 1;
  options.track_min_num_views_per_track = 3;

  global_mapper.EstablishTracks(options);
  EXPECT_EQ(reconstruction->NumPoints3D(), 2);
}

}  // namespace
}  // namespace colmap
