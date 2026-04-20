#pragma once

#include "colmap/glomap/camera.h"
#include "colmap/glomap/image.h"
#include "colmap/glomap/image_pair.h"
#include "colmap/glomap/track.h"
#include "colmap/glomap/view_graph.h"

#include <unordered_map>

namespace colmap::glomap {

// Python map aliases. Binding them with py::bind_map gives mpsfm direct
// iteration semantics identical to the fork pyglomap bindings.
using ImageMap = std::unordered_map<image_t, Image>;
using TrackMap = std::unordered_map<track_t, Track>;
using ImagePairMap = std::unordered_map<image_pair_t, ImagePair>;
using CameraMap = std::unordered_map<camera_t, Camera>;

}  // namespace colmap::glomap
