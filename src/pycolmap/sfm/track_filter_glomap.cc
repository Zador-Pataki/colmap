// Glomap-fork TrackFilter binding under pycolmap.glomap_ra submodule.
// Mirrors the pyglomap.filter_tracks_by_angle and
// pyglomap.filter_track_triangulation_angle Python signatures verbatim
// for migration drop-in compatibility. Round-trips dict[track_id,
// pycolmap.Point3D] through the internal glomap_ra::Track type so callers
// don't need to convert.

#include "colmap/sfm/track_filter_glomap.h"

#include "colmap/scene/camera.h"
#include "colmap/scene/correspondence_graph.h"
#include "colmap/scene/image.h"
#include "colmap/scene/point3d.h"
#include "colmap/scene/track.h"  // colmap::TrackElement (for Point3D round-trip)
#include "colmap/sfm/track_establishment_glomap.h"  // glomap_ra::Track + track_t
#include "colmap/util/types.h"

#include "pycolmap/helpers.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <unordered_map>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

namespace {

// pycolmap.Point3D <-> colmap::glomap_ra::Track conversion. Mirrors the
// pattern in track_establishment_glomap.cc / global_positioning_glomap.cc.
glomap_ra::Track ToGlomapRaTrack(glomap_ra::track_t tid, const Point3D& p3d) {
  glomap_ra::Track t;
  t.track_id = tid;
  t.xyz = p3d.xyz;
  t.color = p3d.color;
  t.is_initialized = p3d.is_initialized;
  t.observations.reserve(p3d.track.Length());
  for (const auto& el : p3d.track.Elements()) {
    t.observations.emplace_back(
        el.image_id, static_cast<glomap_ra::feature_t>(el.point2D_idx));
  }
  t.lc_observations.reserve(p3d.track.lc_elements.size());
  for (const auto& el : p3d.track.lc_elements) {
    t.lc_observations.emplace_back(
        el.image_id, static_cast<glomap_ra::feature_t>(el.point2D_idx));
  }
  return t;
}

Point3D FromGlomapRaTrack(const glomap_ra::Track& src) {
  Point3D p;
  p.xyz = src.xyz;
  p.color = src.color;
  p.is_initialized = src.is_initialized;
  std::vector<TrackElement> elements;
  elements.reserve(src.observations.size());
  for (const auto& [image_id, feature_id] : src.observations) {
    elements.emplace_back(image_id, static_cast<point2D_t>(feature_id));
  }
  p.track.SetElements(std::move(elements));
  p.track.lc_elements.clear();
  p.track.lc_elements.reserve(src.lc_observations.size());
  for (const auto& [image_id, feature_id] : src.lc_observations) {
    p.track.lc_elements.emplace_back(image_id,
                                     static_cast<point2D_t>(feature_id));
  }
  return p;
}

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
  std::unordered_map<glomap_ra::track_t, glomap_ra::Track> tracks;
  tracks.reserve(tracks_py.size());
  for (auto item : tracks_py) {
    glomap_ra::track_t tid = py::cast<glomap_ra::track_t>(item.first);
    Point3D p3d = py::cast<Point3D>(item.second);
    tracks.emplace(tid, ToGlomapRaTrack(tid, p3d));
  }

  int counter;
  {
    py::gil_scoped_release release;
    counter = glomap_ra::TrackFilter::FilterTracksByAngle(
        view_graph, cameras, images, tracks, max_angle_error);
  }

  py::dict tracks_out;
  for (auto& [tid, trk] : tracks) {
    Point3D p3d = FromGlomapRaTrack(trk);
    tracks_out[py::cast(tid)] = py::cast(p3d);
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
  std::unordered_map<glomap_ra::track_t, glomap_ra::Track> tracks;
  tracks.reserve(tracks_py.size());
  for (auto item : tracks_py) {
    glomap_ra::track_t tid = py::cast<glomap_ra::track_t>(item.first);
    Point3D p3d = py::cast<Point3D>(item.second);
    tracks.emplace(tid, ToGlomapRaTrack(tid, p3d));
  }

  int counter;
  {
    py::gil_scoped_release release;
    counter = glomap_ra::TrackFilter::FilterTrackTriangulationAngle(
        view_graph, images, tracks, min_angle);
  }

  py::dict tracks_out;
  for (auto& [tid, trk] : tracks) {
    Point3D p3d = FromGlomapRaTrack(trk);
    tracks_out[py::cast(tid)] = py::cast(p3d);
  }
  py::dict output;
  output["tracks"] = tracks_out;
  output["counter"] = counter;
  return output;
}

// Idempotent get-or-create for the `glomap_ra` submodule. Each binding TU
// has its own anonymous-namespace copy (internal linkage) — they all
// observe the same Python-level submodule object via py::hasattr().
py::module GetOrCreateGlomapRaModule(py::module& m) {
  if (py::hasattr(m, "glomap_ra")) {
    return m.attr("glomap_ra").cast<py::module>();
  }
  return m.def_submodule("glomap_ra");
}

}  // namespace

void BindTrackFilterGlomap(py::module& m) {
  py::module m_glomap_ra = GetOrCreateGlomapRaModule(m);
  m_glomap_ra.def("filter_tracks_by_angle",
                  &RunFilterTracksByAngle,
                  "view_graph"_a,
                  "cameras"_a,
                  "images"_a,
                  "tracks"_a,
                  "max_angle_error"_a = 1.,
                  "Filter tracks by angle error. Returns dict with keys "
                  "'tracks' (filtered subset) and 'counter' (count removed).");
  m_glomap_ra.def(
      "filter_track_triangulation_angle",
      &RunFilterTrackTriangulationAngle,
      "view_graph"_a,
      "images"_a,
      "tracks"_a,
      "min_angle"_a = 1.,
      "Filter tracks by triangulation angle. Returns dict with keys "
      "'tracks' (filtered subset) and 'counter' (count removed).");
}
