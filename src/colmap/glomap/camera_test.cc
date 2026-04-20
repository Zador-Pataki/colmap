#include "colmap/glomap/camera.h"

#include <colmap/sensor/models.h>

#include <gtest/gtest.h>

namespace colmap::glomap {
namespace {

colmap::Camera MakePinhole() {
  return colmap::Camera::CreateFromModelId(
      /*camera_id=*/1,
      colmap::CameraModelId::kPinhole,
      /*focal_length=*/500.0,
      /*width=*/640,
      /*height=*/480);
}

TEST(Camera, CompositionHoldsColmapCamera) {
  Camera cam;
  cam.camera.camera_id = 7;
  EXPECT_EQ(cam.camera.camera_id, 7u);
}

TEST(Camera, HasRefinedFocalLength_DefaultFalse) {
  Camera cam;
  EXPECT_FALSE(cam.has_refined_focal_length);
}

TEST(Camera, HasRefinedFocalLength_Writable) {
  Camera cam;
  cam.has_refined_focal_length = true;
  EXPECT_TRUE(cam.has_refined_focal_length);
}

TEST(Camera, Focal_ReturnsAverage) {
  Camera cam(MakePinhole());
  // Pinhole: params = [fx, fy, cx, cy] with fx == fy == 500.
  EXPECT_NEAR(cam.Focal(), 500.0, 1e-12);
}

TEST(Camera, PrincipalPoint_ReturnsCxCy) {
  Camera cam(MakePinhole());
  const Eigen::Vector2d pp = cam.PrincipalPoint();
  EXPECT_NEAR(pp.x(), 320.0, 1e-12);
  EXPECT_NEAR(pp.y(), 240.0, 1e-12);
}

TEST(Camera, GetK_Returns3x3IntrinsicMatrix) {
  Camera cam(MakePinhole());
  const Eigen::Matrix3d K = cam.GetK();
  EXPECT_NEAR(K(0, 0), 500.0, 1e-12);
  EXPECT_NEAR(K(1, 1), 500.0, 1e-12);
  EXPECT_NEAR(K(0, 2), 320.0, 1e-12);
  EXPECT_NEAR(K(1, 2), 240.0, 1e-12);
  EXPECT_NEAR(K(2, 2), 1.0, 1e-12);
}

}  // namespace
}  // namespace colmap::glomap
