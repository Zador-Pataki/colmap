#include <pybind11/pybind11.h>

namespace py = pybind11;

void BindObservationManager(py::module& m);
void BindIncrementalTriangulator(py::module& m);
void BindIncrementalMapper(py::module& m);
void BindViewGraphManipulation(py::module& m);
void BindRelativePoseEstimation(py::module& m);
void BindTrackEstablishmentGlomap(py::module& m);
void BindImagePairInliersGlomap(py::module& m);
void BindTrackFilterGlomap(py::module& m);

// M13: BindGlobalPositionerOptions, BindGlobalPositioningGlomap, and
// BindRotationAveragingGlomap removed — videosfm pipeline now drives
// GP + RA through native pycolmap.GlobalPositionerOptions /
// RotationEstimatorOptions + pycolmap.run_global_positioning /
// run_rotation_averaging (post M7 + M12). The pycolmap.sfm_ext
// submodule stays alive for track-filter / track-establishment /
// image-pair-inliers which still bind there.

void BindSfm(py::module& m) {
  BindObservationManager(m);
  BindIncrementalTriangulator(m);
  BindIncrementalMapper(m);
  BindViewGraphManipulation(m);
  BindRelativePoseEstimation(m);
  BindTrackEstablishmentGlomap(m);
  BindImagePairInliersGlomap(m);
  BindTrackFilterGlomap(m);
}
