#pragma once

#include "colmap/glomap/camera.h"
#include "colmap/glomap/image.h"
#include "colmap/glomap/types.h"

#include <unordered_map>

namespace colmap::glomap {

void UndistortImages(std::unordered_map<camera_t, Camera>& cameras,
                     std::unordered_map<image_t, Image>& images,
                     bool clean_points = true);

}  // namespace colmap::glomap
