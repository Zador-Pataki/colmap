#include <pybind11/pybind11.h>

namespace py = pybind11;

void BindGlomapGravityInfo(py::module& m);
void BindGlomapCamera(py::module& m);
void BindGlomapTrack(py::module& m);
void BindGlomapImagePair(py::module& m);
void BindGlomapImage(py::module& m);
void BindGlomapViewGraph(py::module& m);
void BindGlomapOptions(py::module& m);
void BindGlomapPipeline(py::module& m);
void BindGlomapCostFunctions(py::module& m);

void BindGlomapSceneTypes(py::module& m_outer) {
  // Fork-style scene types live under pycolmap.glomap.* to avoid name
  // collisions with existing pycolmap.Track / pycolmap.Image / pycolmap.Camera.
  py::module m = m_outer.def_submodule(
      "glomap",
      "Fork-style glomap scene layer (ViewGraph, Image, ImagePair, Track, "
      "Camera, GravityInfo) + pipeline estimators. Mirrors the old pyglomap "
      "fork package; imports rewrite from `pyglomap.X` to `pycolmap.glomap.X`.");

  // Order matters: types referenced by later MakeDataclass / field accessors
  // must already be registered.
  BindGlomapGravityInfo(m);  // no deps
  BindGlomapCamera(m);       // uses colmap::Camera (already bound)
  BindGlomapImagePair(m);    // defines PairType enum; used by ViewGraph
  BindGlomapTrack(m);        // uses Observation
  BindGlomapImage(m);        // uses GravityInfo
  BindGlomapViewGraph(m);    // uses ImagePair
  BindGlomapOptions(m);
  BindGlomapPipeline(m);         // 7 run_* free functions (§12)
  BindGlomapCostFunctions(m);    // §13 cost-functor factories (no-op)
}
