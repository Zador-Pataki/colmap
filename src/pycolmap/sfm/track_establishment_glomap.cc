#include "colmap/sfm/track_establishment_glomap.h"

#include "colmap/scene/correspondence_graph.h"
#include "colmap/scene/image.h"
#include "colmap/scene/point3d.h"
#include "colmap/scene/track.h"
#include "colmap/util/types.h"

#include "pycolmap/helpers.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <unordered_map>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

namespace {

// Convert a glomap_ra::Track (vendored algorithm output) to a
// pycolmap-side Point3D so the videosfm caller receives the same shape
// it gets from from_pyglomap_tracks today.
Point3D ToPoint3D(const glomap_ra::Track& src) {
  Point3D p;
  p.xyz = src.xyz;
  p.color = src.color;
  p.is_initialized = src.is_initialized;
  std::vector<TrackElement> elements;
  elements.reserve(src.observations.size());
  for (const auto& [image_id, feature_id] : src.observations) {
    elements.emplace_back(image_id,
                          static_cast<point2D_t>(feature_id));
  }
  p.track.SetElements(std::move(elements));
  // Glomap fork's lc_observations -> Track::lc_elements.
  p.track.lc_elements.clear();
  p.track.lc_elements.reserve(src.lc_observations.size());
  for (const auto& [image_id, feature_id] : src.lc_observations) {
    p.track.lc_elements.emplace_back(image_id,
                                      static_cast<point2D_t>(feature_id));
  }
  return p;
}

py::dict RunEstablishFullTracks(
    CorrespondenceGraph& view_graph,
    py::dict images_py,
    const glomap_ra::TrackEstablishmentOptions& options) {
  std::unordered_map<image_t, Image> images;
  images.reserve(images_py.size());
  for (auto item : images_py) {
    images.emplace(py::cast<image_t>(item.first),
                   py::cast<Image>(item.second));
  }

  std::unordered_map<glomap_ra::track_t, glomap_ra::Track> tracks_full;
  {
    py::gil_scoped_release release;
    glomap_ra::TrackEngine engine(view_graph, images, options);
    engine.EstablishFullTracks(tracks_full);
  }

  py::dict tracks_out;
  for (auto& [tid, track] : tracks_full) {
    Point3D p3d = ToPoint3D(track);
    tracks_out[py::cast(tid)] = py::cast(p3d);
  }
  return tracks_out;
}

py::dict RunFindTracksForProblem(
    CorrespondenceGraph& view_graph,
    py::dict images_py,
    py::dict tracks_full_py,
    const glomap_ra::TrackEstablishmentOptions& options) {
  std::unordered_map<image_t, Image> images;
  images.reserve(images_py.size());
  for (auto item : images_py) {
    images.emplace(py::cast<image_t>(item.first),
                   py::cast<Image>(item.second));
  }

  // Read tracks_full_py back into glomap_ra::Track form (fields are
  // round-tripped via the Point3D shape that ToPoint3D produced earlier).
  std::unordered_map<glomap_ra::track_t, glomap_ra::Track> tracks_full;
  tracks_full.reserve(tracks_full_py.size());
  for (auto item : tracks_full_py) {
    const glomap_ra::track_t tid = py::cast<glomap_ra::track_t>(item.first);
    Point3D p3d = py::cast<Point3D>(item.second);
    glomap_ra::Track& t = tracks_full[tid];
    t.track_id = tid;
    t.xyz = p3d.xyz;
    t.color = p3d.color;
    t.is_initialized = p3d.is_initialized;
    t.observations.reserve(p3d.track.Length());
    for (const auto& el : p3d.track.Elements()) {
      t.observations.emplace_back(el.image_id, el.point2D_idx);
    }
    t.lc_observations.reserve(p3d.track.lc_elements.size());
    for (const auto& el : p3d.track.lc_elements) {
      t.lc_observations.emplace_back(el.image_id, el.point2D_idx);
    }
  }

  std::unordered_map<glomap_ra::track_t, glomap_ra::Track> tracks_selected;
  {
    py::gil_scoped_release release;
    glomap_ra::TrackEngine engine(view_graph, images, options);
    engine.FindTracksForProblem(tracks_full, tracks_selected);
  }

  py::dict tracks_out;
  for (auto& [tid, track] : tracks_selected) {
    Point3D p3d = ToPoint3D(track);
    tracks_out[py::cast(tid)] = py::cast(p3d);
  }
  return tracks_out;
}

}  // namespace

void BindTrackEstablishmentGlomap(py::module& m) {
  auto PyOpts =
      py::classh<glomap_ra::TrackEstablishmentOptions>(
          m, "TrackEstablishmentOptions")
          .def(py::init<>())
          .def_readwrite("thres_inconsistency",
                         &glomap_ra::TrackEstablishmentOptions::thres_inconsistency)
          .def_readwrite(
              "min_num_tracks_per_view",
              &glomap_ra::TrackEstablishmentOptions::min_num_tracks_per_view)
          .def_readwrite(
              "min_num_view_per_track",
              &glomap_ra::TrackEstablishmentOptions::min_num_view_per_track)
          .def_readwrite(
              "max_num_view_per_track",
              &glomap_ra::TrackEstablishmentOptions::max_num_view_per_track)
          .def_readwrite("max_num_tracks",
                         &glomap_ra::TrackEstablishmentOptions::max_num_tracks);
  MakeDataclass(PyOpts);

  m.def("establish_full_tracks",
        &RunEstablishFullTracks,
        "view_graph"_a,
        "images"_a,
        "options"_a,
        "Glomap-style track establishment via TrackEngine. Returns a dict "
        "mapping track_id -> pycolmap.Point3D with .track.elements + the "
        "fork-added .track.lc_elements populated.");

  m.def("find_tracks_for_problem",
        &RunFindTracksForProblem,
        "view_graph"_a,
        "images"_a,
        "tracks_full"_a,
        "options"_a,
        "Subsample tracks for the global-positioning problem. Returns the "
        "selected subset as a fresh dict[int, pycolmap.Point3D].");
}
