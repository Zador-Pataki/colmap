#include "colmap/estimators/manifold.h"

#include "pycolmap/helpers.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindCustomizedManifold(py::module& m) {
#if CERES_VERSION_MAJOR >= 3 || \
    (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 1)
  // Only bind PositiveExponentialManifold if pyceres is available.
  // This avoids stubgen errors when pyceres is not installed.
  // The ceres::Manifold base type needs to be registered in Python for the binding to work.
  try {
    // Try to import ceres module to register the base type
    (void)py::module_::import("ceres");
    // If successful, bind the class
    py::class_<PositiveExponentialManifold<ceres::DYNAMIC>, ceres::Manifold>(
        m, "PositiveExponentialManifold")
        .def(py::init<int>());
  } catch (const py::error_already_set&) {
    // ceres module not available (e.g., during stub generation without pyceres)
    // Skip binding - this is fine since PositiveExponentialManifold is rarely used
  }
#endif
}

void BindManifold(py::module& m_parent) {
  py::module_ m = m_parent.def_submodule("manifold");
  if (IsPyceresAvailable()) {
    BindCustomizedManifold(m);
  }
}
