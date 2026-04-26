#include <pybind11/pybind11.h>

namespace py = pybind11;

void BindObservationManager(py::module& m);
void BindIncrementalTriangulator(py::module& m);
void BindIncrementalMapper(py::module& m);
void BindViewGraphManipulation(py::module& m);
void BindRelativePoseEstimation(py::module& m);
void BindRotationAveragingGlomap(py::module& m);

void BindSfm(py::module& m) {
  BindObservationManager(m);
  BindIncrementalTriangulator(m);
  BindIncrementalMapper(m);
  BindViewGraphManipulation(m);
  BindRelativePoseEstimation(m);
  BindRotationAveragingGlomap(m);
}
