// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/sfm/global_mapper.h"

#include "colmap/scene/database_cache.h"
#include "colmap/scene/reconstruction_matchers.h"
#include "colmap/scene/synthetic.h"
#include "colmap/util/testing.h"

#include <gtest/gtest.h>

namespace colmap {
namespace {

// TODO(jsch): Add tests for pose priors.

std::shared_ptr<DatabaseCache> CreateDatabaseCache(const Database& database) {
  DatabaseCache::Options options;
  return DatabaseCache::Create(database, options);
}

TEST(GlobalMapper, WithoutNoise) {
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

TEST(GlobalMapper, TrackConstructionBeforeScaleGeneration) {
  auto database = Database::Open(CreateTestDir() / "database.db");
  Reconstruction original;
  SyntheticDatasetOptions dataset_options;
  dataset_options.num_rigs = 1;
  dataset_options.num_cameras_per_rig = 1;
  dataset_options.num_frames_per_rig = 3;
  dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(dataset_options, &original, database.get());
  auto cache = CreateDatabaseCache(*database);
  for (const bool skip_global_positioning : {false, true}) {
    for (const int min_track_length : {0, 3, 4}) {
      auto reconstruction = std::make_shared<Reconstruction>(original);
      for (const auto point_id : reconstruction->Point3DIds()) {
        reconstruction->DeletePoint3D(point_id);
      }
      GlobalMapper mapper(cache);
      mapper.BeginReconstruction(reconstruction);
      GlobalMapperOptions options;
      options.skip_global_positioning = skip_global_positioning;
      options.global_positioning.min_num_view_per_track = min_track_length;
      mapper.EstablishTracks(options);
      const size_t expected = skip_global_positioning || min_track_length <= 3
                                  ? original.NumPoints3D()
                                  : 0;
      EXPECT_EQ(reconstruction->NumPoints3D(), expected);
      const auto scales = GenerateObservationScales(*reconstruction);
      EXPECT_EQ(scales.size(), 3 * expected);
    }
  }
}

TEST(GlobalMapper, GlobalPositioningTrackFiltering) {
  auto database = Database::Open(CreateTestDir() / "database.db");
  auto reconstruction = std::make_shared<Reconstruction>();
  SyntheticDatasetOptions dataset_options;
  dataset_options.num_rigs = 1;
  dataset_options.num_cameras_per_rig = 1;
  dataset_options.num_frames_per_rig = 3;
  dataset_options.two_view_geometry_has_relative_pose = true;
  SynthesizeDataset(dataset_options, reconstruction.get(), database.get());
  GlobalMapper mapper(CreateDatabaseCache(*database));
  mapper.BeginReconstruction(reconstruction);
  const auto point_id = reconstruction->Points3D().begin()->first;
  const auto observation = reconstruction->Point3D(point_id).track.Element(0);
  reconstruction->DeleteObservation(observation.image_id,
                                    observation.point2D_idx);
  const Reconstruction original = *reconstruction;
  GlobalMapperOptions mapper_options;
  mapper_options.skip_rotation_averaging = true;
  mapper_options.skip_track_establishment = true;
  mapper_options.skip_bundle_adjustment = true;
  mapper_options.skip_retriangulation = true;
  mapper_options.min_tri_angle_deg = 0.0;
  auto& options = mapper_options.global_positioning;
  options.use_gpu = false;
  options.generate_random_positions = false;
  options.generate_random_points = false;
  options.initialize_scales_from_geometry = true;
  for (const int min_track_length : {0, 3, 4}) {
    options.min_num_view_per_track = min_track_length;
    *reconstruction = original;
    {
      auto scales = GenerateObservationScales(*reconstruction, true);
      auto positioner =
          GlobalPositioner::CreateDefault(options, *reconstruction, scales);
      for (auto& [id, point3D] : reconstruction->Points3D()) {
        EXPECT_TRUE(
            positioner->Problem().HasParameterBlock(point3D.xyz.data()));
      }
    }
    EXPECT_EQ(mapper.Solve(mapper_options), min_track_length <= 3);
    for (const auto& [id, point3D] : original.Points3D()) {
      EXPECT_EQ(
          reconstruction->ExistsPoint3D(id),
          point3D.track.Length() >= static_cast<size_t>(min_track_length));
    }
  }
}

TEST(GlobalMapper, BundleAdjustmentTrackFiltering) {
  auto database = Database::Open(CreateTestDir() / "database.db");
  auto reconstruction = std::make_shared<Reconstruction>();
  SyntheticDatasetOptions dataset_options;
  dataset_options.num_rigs = 1;
  dataset_options.num_cameras_per_rig = 1;
  dataset_options.num_frames_per_rig = 3;
  SynthesizeDataset(dataset_options, reconstruction.get(), database.get());
  GlobalMapper mapper(CreateDatabaseCache(*database));
  mapper.BeginReconstruction(reconstruction);
  const auto point_id = reconstruction->Points3D().begin()->first;
  const auto observation = reconstruction->Point3D(point_id).track.Element(0);
  reconstruction->DeleteObservation(observation.image_id,
                                    observation.point2D_idx);
  const Reconstruction original = *reconstruction;
  BundleAdjustmentOptions options;
  options.print_summary = false;
  for (const int min_track_length : {0, 3}) {
    options.min_track_length = min_track_length;
    *reconstruction = original;
    bool checked = false;
    EXPECT_TRUE(mapper.IterativeBundleAdjustment(
        options, 1e-2, 1.0, 1, true, false, [&]() {
          EXPECT_EQ(reconstruction->ExistsPoint3D(point_id),
                    min_track_length == 0);
          checked = true;
          return true;
        }));
    EXPECT_TRUE(checked);
  }
  *reconstruction = original;
  options.min_track_length = 4;
  EXPECT_FALSE(mapper.IterativeBundleAdjustment(options, 1e-2, 1.0, 1));
  EXPECT_EQ(reconstruction->NumPoints3D(), 0);

  *reconstruction = original;
  options.min_track_length = 3;
  auto other_point = reconstruction->Points3D().begin();
  if (other_point->first == point_id) ++other_point;
  const auto other_id = other_point->first;
  const auto outlier = reconstruction->Point3D(other_id).track.Element(0);
  reconstruction->Image(outlier.image_id).Point2D(outlier.point2D_idx).xy +=
      Eigen::Vector2d(1000, 1000);
  EXPECT_TRUE(mapper.IterativeBundleAdjustment(options, 1e-2, 1.0, 0));
  EXPECT_FALSE(reconstruction->ExistsPoint3D(other_id));
}

TEST(GlobalMapperOptions, RefineSensorFromRigPropagatesToSubOptions) {
  GlobalMapperOptions options;
  options.refine_sensor_from_rig = false;
  // Sub-options keep their own defaults (true) until accessed.
  EXPECT_TRUE(options.rotation_averaging.refine_sensor_from_rig);
  EXPECT_TRUE(options.global_positioning.refine_sensor_from_rig);
  EXPECT_TRUE(options.bundle_adjustment.refine_sensor_from_rig);
  // Accessors return resolved sub-options with the top-level flag applied.
  EXPECT_FALSE(options.RotationAveraging().refine_sensor_from_rig);
  EXPECT_FALSE(options.GlobalPositioning().refine_sensor_from_rig);
  EXPECT_FALSE(options.BundleAdjustment().refine_sensor_from_rig);
}

}  // namespace
}  // namespace colmap
