#include "colmap/scene/point3d.h"

#include "colmap/util/logging.h"
#include "colmap/util/misc.h"
#include "colmap/util/types.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"
#include "pycolmap/scene/types.h"

#include <memory>
#include <sstream>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;
namespace py = pybind11;

// Bind Point3D
void BindPoint3D(py::module& m) {
  py::class_ext_<Point3D, std::shared_ptr<Point3D>> PyPoint3D(m, "Point3D");
  PyPoint3D.def(py::init<>())
      .def_readwrite("xyz", &Point3D::xyz)
      .def_readwrite("color", &Point3D::color)
      .def_readwrite("error", &Point3D::error)
      .def_readwrite("track", &Point3D::track);
  MakeDataclass(PyPoint3D);

  py::bind_map<Point3DMap>(m, "MapPoint3DIdToPoint3D");
}

py::array GetPoint3DCoordinates(const Point3DMap& points3d_map, const std::vector<int>& point_ids) {
    std::vector<double> coordinates; // Store all xyz in a flat vector

    for (int id : point_ids) {
        auto it = points3d_map.find(id);
        if (it != points3d_map.end()) {
            const auto& xyz = it->second.xyz;
            coordinates.insert(coordinates.end(), xyz.begin(), xyz.end());
        }
    }

    // Create a NumPy array with shape (N, 3) where N is the number of points
    py::ssize_t num_points = coordinates.size() / 3;
    std::vector<py::ssize_t> shape = {num_points, 3};          // Shape of the array
    std::vector<py::ssize_t> strides = {3 * sizeof(double), sizeof(double)}; // Strides for the array

    return py::array(py::buffer_info(
        coordinates.data(),                // Pointer to the data
        sizeof(double),                    // Size of one element
        py::format_descriptor<double>::format(), // Data type descriptor
        2,                                 // Number of dimensions
        shape,                             // Shape of the array
        strides                            // Strides of the array
    ));
}
// Bind GetPoint3DCoordinates
void BindPoint3DFunctions(py::module& m) {
    m.def("get_point3d_coordinates", &GetPoint3DCoordinates, 
          py::arg("points3d_map"), py::arg("point_ids"),
          "Retrieve xyz coordinates of the given Point3D IDs as a NumPy array");
}