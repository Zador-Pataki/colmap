#include "colmap/sfm/image_pair_inliers_glomap.h"

#include "colmap/scene/camera.h"
#include "colmap/scene/correspondence_graph.h"
#include "colmap/scene/image.h"
#include "colmap/util/types.h"

#include "pycolmap/helpers.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <unordered_map>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

namespace {

py::dict RunImagePairsInlierCount(
    CorrespondenceGraph& view_graph,
    py::dict cameras_py,
    py::dict images_py,
    const sfm_ext::InlierThresholdOptions& options,
    bool clean_inliers) {
  std::unordered_map<camera_t, Camera> cameras;
  cameras.reserve(cameras_py.size());
  for (auto item : cameras_py) {
    cameras.emplace(py::cast<camera_t>(item.first),
                    py::cast<Camera>(item.second));
  }
  std::unordered_map<image_t, Image> images;
  images.reserve(images_py.size());
  for (auto item : images_py) {
    images.emplace(py::cast<image_t>(item.first),
                   py::cast<Image>(item.second));
  }

  {
    py::gil_scoped_release release;
    sfm_ext::ImagePairsInlierCount(
        view_graph, cameras, images, options, clean_inliers);
  }

  py::dict cameras_out;
  for (auto& [cid, cam] : cameras) {
    cameras_out[py::cast(cid)] = py::cast(cam);
  }
  py::dict images_out;
  for (auto& [iid, img] : images) {
    images_out[py::cast(iid)] = py::cast(img);
  }
  py::dict output;
  output["view_graph"] = view_graph;
  output["cameras"] = cameras_out;
  output["images"] = images_out;
  return output;
}

}  // namespace

void BindImagePairInliersGlomap(py::module& m) {
  auto PyOpts =
      py::classh<sfm_ext::InlierThresholdOptions>(
          m, "InlierThresholdOptions")
          .def(py::init<>())
          .def_readwrite("max_angle_error",
                         &sfm_ext::InlierThresholdOptions::max_angle_error)
          .def_readwrite(
              "max_reprojection_error",
              &sfm_ext::InlierThresholdOptions::max_reprojection_error)
          .def_readwrite(
              "min_triangulation_angle",
              &sfm_ext::InlierThresholdOptions::min_triangulation_angle)
          .def_readwrite(
              "max_epipolar_error_E",
              &sfm_ext::InlierThresholdOptions::max_epipolar_error_E)
          .def_readwrite(
              "max_epipolar_error_F",
              &sfm_ext::InlierThresholdOptions::max_epipolar_error_F)
          .def_readwrite(
              "max_epipolar_error_H",
              &sfm_ext::InlierThresholdOptions::max_epipolar_error_H)
          .def_readwrite(
              "min_angle_from_epipole",
              &sfm_ext::InlierThresholdOptions::min_angle_from_epipole)
          .def_readwrite("min_inlier_num",
                         &sfm_ext::InlierThresholdOptions::min_inlier_num)
          .def_readwrite(
              "min_inlier_ratio",
              &sfm_ext::InlierThresholdOptions::min_inlier_ratio)
          .def_readwrite(
              "max_rotation_error",
              &sfm_ext::InlierThresholdOptions::max_rotation_error);
  MakeDataclass(PyOpts);

  m.def("image_pairs_inlier_count",
        &RunImagePairsInlierCount,
        "view_graph"_a,
        "cameras"_a,
        "images"_a,
        "options"_a,
        "clean_inliers"_a,
        "Per-pair inlier scoring (Sampson + cheirality + epipole checks). "
        "Updates ImagePair.inliers and is_valid in place. Reads "
        "Image.depth_prior_validity to apply depth-aware epipole thresholds "
        "(thres_epipole vs thres_epipole_nodepth).");
}
