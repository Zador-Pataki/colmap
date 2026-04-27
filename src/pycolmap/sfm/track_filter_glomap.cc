// TrackFilter binding under the pycolmap.sfm_ext submodule. The tracks
// dict carries pycolmap.Point3D directly post-Track-collapse (no
// fork<->native round-trip).

#include "colmap/sfm/track_filter_glomap.h"

#include "colmap/scene/camera.h"
#include "colmap/scene/correspondence_graph.h"
#include "colmap/scene/image.h"
#include "colmap/scene/point3d.h"
#include "colmap/util/types.h"

#include "pycolmap/helpers.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <unordered_map>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

namespace {

py::dict RunFilterTracksByAngle(CorrespondenceGraph& view_graph,
                                py::dict cameras_py,
                                py::dict images_py,
                                py::dict tracks_py,
                                double max_angle_error) {
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
  std::unordered_map<point3D_t, Point3D> tracks;
  tracks.reserve(tracks_py.size());
  for (auto item : tracks_py) {
    tracks.emplace(py::cast<point3D_t>(item.first),
                   py::cast<Point3D>(item.second));
  }

  int counter;
  {
    py::gil_scoped_release release;
    counter = sfm_ext::TrackFilter::FilterTracksByAngle(
        view_graph, cameras, images, tracks, max_angle_error);
  }

  py::dict tracks_out;
  for (auto& [tid, p3d] : tracks) {
    tracks_out[py::cast(tid)] = py::cast(std::move(p3d));
  }
  py::dict output;
  output["tracks"] = tracks_out;
  output["counter"] = counter;
  return output;
}

py::dict RunFilterTrackTriangulationAngle(CorrespondenceGraph& view_graph,
                                          py::dict images_py,
                                          py::dict tracks_py,
                                          double min_angle) {
  std::unordered_map<image_t, Image> images;
  images.reserve(images_py.size());
  for (auto item : images_py) {
    images.emplace(py::cast<image_t>(item.first),
                   py::cast<Image>(item.second));
  }
  std::unordered_map<point3D_t, Point3D> tracks;
  tracks.reserve(tracks_py.size());
  for (auto item : tracks_py) {
    tracks.emplace(py::cast<point3D_t>(item.first),
                   py::cast<Point3D>(item.second));
  }

  int counter;
  {
    py::gil_scoped_release release;
    counter = sfm_ext::TrackFilter::FilterTrackTriangulationAngle(
        view_graph, images, tracks, min_angle);
  }

  py::dict tracks_out;
  for (auto& [tid, p3d] : tracks) {
    tracks_out[py::cast(tid)] = py::cast(std::move(p3d));
  }
  py::dict output;
  output["tracks"] = tracks_out;
  output["counter"] = counter;
  return output;
}

// Idempotent get-or-create for the `sfm_ext` submodule. Each binding TU
// has its own anonymous-namespace copy (internal linkage) — they all
// observe the same Python-level submodule object via py::hasattr().
py::module GetOrCreateSfmExtModule(py::module& m) {
  if (py::hasattr(m, "sfm_ext")) {
    return m.attr("sfm_ext").cast<py::module>();
  }
  return m.def_submodule("sfm_ext");
}

}  // namespace

void BindTrackFilterGlomap(py::module& m) {
  py::module m_sfm_ext = GetOrCreateSfmExtModule(m);
  m_sfm_ext.def("filter_tracks_by_angle",
                  &RunFilterTracksByAngle,
                  "view_graph"_a,
                  "cameras"_a,
                  "images"_a,
                  "tracks"_a,
                  "max_angle_error"_a = 1.,
                  "Filter tracks by angle error. Returns dict with keys "
                  "'tracks' (filtered subset) and 'counter' (count removed).");
  m_sfm_ext.def(
      "filter_track_triangulation_angle",
      &RunFilterTrackTriangulationAngle,
      "view_graph"_a,
      "images"_a,
      "tracks"_a,
      "min_angle"_a = 1.,
      "Filter tracks by triangulation angle. Returns dict with keys "
      "'tracks' (filtered subset) and 'counter' (count removed).");
}
