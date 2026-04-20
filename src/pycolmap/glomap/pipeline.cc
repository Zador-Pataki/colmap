// §12 Python bindings for the glomap pipeline run_* free functions.
#include "colmap/estimators/glomap/pipeline.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace colmap::glomap;
using namespace pybind11::literals;

void BindGlomapPipeline(py::module& m) {
  m.def("run_rotation_averaging",
        &run_rotation_averaging,
        "options"_a, "view_graph"_a, "images"_a,
        "Estimate global rotations from relative rotations in the view "
        "graph. Returns True on success.");

  m.def("run_global_positioning",
        &run_global_positioning,
        "options"_a, "view_graph"_a, "cameras"_a, "images"_a, "tracks"_a,
        "Estimate global camera positions and 3D point positions. Returns "
        "True on success.");

  m.def("run_relative_pose_estimation",
        &run_relative_pose_estimation,
        "options"_a, "view_graph"_a, "cameras"_a, "images"_a,
        "Estimate relative poses for every valid pair in the view graph.");

  m.def("run_view_graph_calibration",
        &run_view_graph_calibration,
        "options"_a, "view_graph"_a, "cameras"_a, "images"_a,
        "Estimate focal lengths consistently across the view graph.");

  m.def("run_track_establishment",
        &run_track_establishment,
        "options"_a, "view_graph"_a, "images"_a, "tracks"_a,
        "Establish consistent feature tracks from the view graph. Returns "
        "the number of tracks.");

  m.def("run_track_filter",
        &run_track_filter,
        "options"_a, "view_graph"_a, "cameras"_a, "images"_a, "tracks"_a,
        "Filter tracks by reprojection error. Returns the number of "
        "observations removed.");

  m.def("run_bundle_adjustment",
        &run_bundle_adjustment,
        "options"_a, "view_graph"_a, "cameras"_a, "images"_a, "tracks"_a,
        "Run bundle adjustment over poses + tracks + intrinsics. Returns "
        "True on success.");

  // run_track_retriangulation deferred — impl not yet wired.
}
