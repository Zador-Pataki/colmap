#include <pybind11/pybind11.h>

namespace py = pybind11;

void BindObservationManager(py::module& m);
void BindIncrementalTriangulator(py::module& m);
void BindIncrementalMapper(py::module& m);
void BindViewGraphManipulation(py::module& m);
void BindTrackEstablishment(py::module& m);
void BindImagePairInliers(py::module& m);
void BindTrackFilter(py::module& m);

// GP + RA bindings live at top-level pycolmap as
// pycolmap.GlobalPositionerOptions / RotationEstimatorOptions /
// run_global_positioning / run_rotation_averaging. Track-filter,
// track-establishment, and image-pair-inliers also bind at top-level
// pycolmap.

void BindSfm(py::module& m) {
  BindObservationManager(m);
  BindIncrementalTriangulator(m);
  BindIncrementalMapper(m);
  BindViewGraphManipulation(m);
  BindTrackEstablishment(m);
  BindImagePairInliers(m);
  BindTrackFilter(m);
}
