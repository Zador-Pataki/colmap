
#pragma once

#include "colmap/estimators/glomap/triangulator_options.h"

#include "colmap/glomap/image.h"
#include "colmap/glomap/image_pair.h"
#include "colmap/glomap/track.h"
#include "colmap/glomap/view_graph.h"

#include <colmap/scene/database.h>

namespace colmap::glomap {

// TriangulatorOptions is defined in colmap/estimators/glomap/triangulator_options.h (§07).

bool RetriangulateTracks(const TriangulatorOptions& options,
                         const colmap::Database& database,
                         std::unordered_map<camera_t, Camera>& cameras,
                         std::unordered_map<image_t, Image>& images,
                         std::unordered_map<track_t, Track>& tracks);

}  // namespace colmap::glomap
