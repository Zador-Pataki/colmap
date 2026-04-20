#include "colmap/glomap/gravity_info.h"

#include "pycolmap/helpers.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using colmap::glomap::GravityInfo;
using namespace pybind11::literals;

void BindGlomapGravityInfo(py::module& m) {
  py::classh<GravityInfo> cls(m, "GravityInfo");
  cls.def(py::init<>())
      .def_readwrite("has_gravity", &GravityInfo::has_gravity)
      .def("set_gravity",
           &GravityInfo::SetGravity,
           "g"_a,
           "Set gravity direction (camera frame, unit vector).")
      .def("get_gravity",
           &GravityInfo::GetGravity,
           "Get gravity direction (camera frame).")
      .def("get_r_align",
           &GravityInfo::GetRAlign,
           "Get alignment rotation (2nd column = gravity direction).");
  MakeDataclass(cls);
}
