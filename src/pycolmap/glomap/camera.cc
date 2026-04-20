#include "colmap/glomap/camera.h"

#include "pycolmap/helpers.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using colmap::glomap::Camera;

void BindGlomapCamera(py::module& m) {
  py::classh<Camera> cls(m, "Camera");
  cls.def(py::init<>())
      .def(py::init<const colmap::Camera&>(), py::arg("camera"))
      .def_readwrite("camera", &Camera::camera,
                     "Underlying colmap.Camera (composition, not inheritance).")
      .def_readwrite("has_refined_focal_length", &Camera::has_refined_focal_length)
      .def("focal", &Camera::Focal, "Average (FocalLengthX + FocalLengthY) / 2.")
      .def("principal_point", &Camera::PrincipalPoint,
           "Principal point as Eigen::Vector2d (cx, cy).")
      .def("get_K", &Camera::GetK, "3x3 intrinsic matrix.");
  MakeDataclass(cls);
}
