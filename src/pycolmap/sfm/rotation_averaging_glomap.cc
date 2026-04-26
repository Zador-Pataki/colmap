#include "colmap/sfm/rotation_averaging_glomap.h"

#include "colmap/estimators/rotation_averaging.h"
#include "colmap/scene/correspondence_graph.h"
#include "colmap/scene/image.h"
#include "colmap/util/types.h"

#include "pycolmap/helpers.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <unordered_map>
#include <utility>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

namespace {

// Glomap-style entry point: takes a CorrespondenceGraph plus a Python dict
// of pycolmap.Image, runs rotation averaging via the fork-port in
// colmap::glomap_ra::RotationEstimator, and returns
// `(images_dict, weights_dict)` to match pyglomap's
// `run_rotation_averaging` signature.
py::tuple RunRotationAveragingGlomap(
    CorrespondenceGraph& view_graph,
    py::dict images_py,
    const RotationEstimatorOptions& options) {
  std::unordered_map<image_t, Image> images;
  images.reserve(images_py.size());
  for (auto item : images_py) {
    images.emplace(py::cast<image_t>(item.first),
                   py::cast<Image>(item.second));
  }

  std::unordered_map<image_pair_t, double> weights;
  {
    py::gil_scoped_release release;
    // Mirror pyglomap.run_rotation_averaging: filter to largest connected
    // component first (this is what sets is_registered=true on the surviving
    // images, which the spanning-tree initialization in
    // RotationEstimator::EstimateRotations requires).
    glomap_ra::KeepLargestConnectedComponents(view_graph, images);
    glomap_ra::RotationEstimator estimator(options);
    weights = estimator.EstimateRotations(view_graph, images);
  }

  py::dict images_out;
  for (auto& [iid, img] : images) {
    images_out[py::cast(iid)] = py::cast(img);
  }
  py::dict weights_out;
  for (const auto& [pid, w] : weights) {
    weights_out[py::cast(pid)] = py::cast(w);
  }
  return py::make_tuple(images_out, weights_out);
}

}  // namespace

void BindRotationAveragingGlomap(py::module& m) {
  m.def("run_rotation_averaging",
        &RunRotationAveragingGlomap,
        "view_graph"_a,
        "images"_a,
        "options"_a,
        "Glomap-style rotation averaging on a CorrespondenceGraph + images "
        "dict. Returns (images_dict, weights_dict). Uses the fork's "
        "extensions (precomputed weights, video-aware Ceres solver, "
        "risky-LC filtering) when the matching options on "
        "RotationEstimatorOptions are set.");
}
