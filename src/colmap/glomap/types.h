#pragma once

#include <colmap/geometry/rigid3.h>
#include <colmap/util/types.h>

#include <cstdint>
#include <limits>
#include <unordered_map>
#include <unordered_set>

namespace colmap::glomap {

// Unique identifier for cameras.
using colmap::camera_t;

// Unique identifier for images.
using colmap::image_t;

// Each image pair gets a unique ID, see `Database::ImagePairToPairId`.
typedef uint64_t image_pair_t;

// Index per image, i.e. determines maximum number of 2D points per image.
typedef uint32_t feature_t;

// Unique identifier per added 3D point.
typedef uint64_t track_t;

using colmap::Rigid3d;

const image_t kMaxNumImages = std::numeric_limits<std::int32_t>::max();
const image_pair_t kInvalidImagePairId = -1;

}  // namespace colmap::glomap
