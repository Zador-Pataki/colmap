#include "colmap/glomap/image_pair.h"

#include "pycolmap/glomap/types.h"
#include "pycolmap/helpers.h"

#include <memory>
#include <optional>
#include <sstream>

#include <colmap/geometry/rigid3.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;
using namespace pybind11::literals;

using colmap::Rigid3d;
using colmap::glomap::image_pair_t;
using colmap::glomap::image_t;
using colmap::glomap::ImagePair;
using colmap::glomap::ImagePairMap;
using colmap::glomap::PairType;

void BindGlomapImagePair(py::module& m) {
  py::enum_<PairType>(m, "PairType")
      .value("ADJACENT", PairType::ADJACENT)
      .value("NONADJACENT", PairType::NONADJACENT)
      .value("LOOP_CLOSURE", PairType::LOOP_CLOSURE)
      .export_values();

  py::classh<ImagePair> PyImagePair(m, "ImagePair");
  PyImagePair.def(py::init<>())
      .def(py::init<image_t, image_t, Rigid3d>(),
           "image_id1"_a = image_t{static_cast<image_t>(-1)},
           "image_id2"_a = image_t{static_cast<image_t>(-1)},
           "pose_rel"_a = Rigid3d())
      .def_property_readonly(
          "image_id1",
          [](const ImagePair& self) -> image_t { return self.image_id1; },
          "The unique identifier of image 1.")
      .def_property_readonly(
          "image_id2",
          [](const ImagePair& self) -> image_t { return self.image_id2; },
          "The unique identifier of image 2.")
      .def_property_readonly(
          "pair_id",
          [](const ImagePair& self) -> image_pair_t { return self.pair_id; },
          "The unique identifier of the image pair.")
      .def_readwrite("type", &ImagePair::type,
                     "PairType tag (ADJACENT / NONADJACENT / LOOP_CLOSURE).")
      .def_readwrite("is_valid", &ImagePair::is_valid,
                     "Whether the image pair is valid.")
      .def_readwrite("is_LC", &ImagePair::is_LC,
                     "Whether this is a loop closure (legacy; retained "
                     "alongside `type`).")
      .def_readwrite("weight", &ImagePair::weight, "The initial inlier rate.")
      .def_readwrite("rel_depth_scale", &ImagePair::rel_depth_scale,
                     "Relative depth scale: depth_2 = rel_depth_scale * "
                     "depth_1. -1.0 = not computed.")
      .def_readwrite("config", &ImagePair::config,
                     "Geometric configuration (colmap::TwoViewGeometry).")
      .def_readwrite("E", &ImagePair::E, "Essential matrix.")
      .def_readwrite("F", &ImagePair::F, "Fundamental matrix.")
      .def_readwrite("H", &ImagePair::H, "Homography matrix.")
      .def_property(
          "cam2_from_cam1",
          [](const ImagePair& self) -> Rigid3d { return self.cam2_from_cam1; },
          [](ImagePair& self, const Rigid3d& value) {
            self.cam2_from_cam1 = value;
          },
          "Relative pose from camera 1 to camera 2 (pycolmap.Rigid3d).")
      .def_readwrite("cov_t", &ImagePair::cov_t,
                     "3x3 covariance for relative translation (zero = not "
                     "computed).")
      .def_readwrite("matches", &ImagePair::matches,
                     "Matches between the two images (Nx2).")
      .def_readwrite("inliers", &ImagePair::inliers,
                     "Row indices of inlier matches.")
      .def_readwrite("are_lc", &ImagePair::are_lc,
                     "Whether each match is a loop-closure match.")
      .def("__repr__", CreateRepresentation<ImagePair>)
      // Batch update — reduces Python-C++ boundary crossings. Ported from
      // fork; Rigid3d now goes through colmap4's type caster.
      .def(
          "update",
          [](ImagePair& self,
             std::optional<bool> is_valid,
             std::optional<double> weight,
             std::optional<double> rel_depth_scale,
             std::optional<Rigid3d> cam2_from_cam1,
             std::optional<Eigen::Matrix3d> cov_t,
             std::optional<std::vector<int>> inliers) {
            if (is_valid) self.is_valid = *is_valid;
            if (weight) self.weight = *weight;
            if (rel_depth_scale) self.rel_depth_scale = *rel_depth_scale;
            if (cam2_from_cam1) self.cam2_from_cam1 = *cam2_from_cam1;
            if (cov_t) self.cov_t = *cov_t;
            if (inliers) self.inliers = *inliers;
          },
          py::arg("is_valid") = py::none(),
          py::arg("weight") = py::none(),
          py::arg("rel_depth_scale") = py::none(),
          py::arg("cam2_from_cam1") = py::none(),
          py::arg("cov_t") = py::none(),
          py::arg("inliers") = py::none(),
          "Batch update multiple attributes in a single call.")
      .def("summary", [](const ImagePair& self) {
        std::ostringstream ss;
        ss << "ImagePair: " << self.image_id1 << " - " << self.image_id2
           << "\n"
           << "  pair_id: " << self.pair_id << "\n"
           << "  type: " << static_cast<int>(self.type) << "\n"
           << "  is_valid: " << self.is_valid << "\n"
           << "  is_LC: " << self.is_LC << "\n"
           << "  weight: " << self.weight << "\n"
           << "  rel_depth_scale: " << self.rel_depth_scale << "\n"
           << "  cam2_from_cam1: " << self.cam2_from_cam1 << "\n"
           << "  num_matches: " << self.matches.rows() << "\n"
           << "  num_inliers: " << self.inliers.size() << "\n"
           << "  num_lc_matches: " << self.are_lc.size();
        return ss.str();
      });
  MakeDataclass(PyImagePair);

  py::bind_map<ImagePairMap>(m, "MapImagePairIdToImagePair");
}
