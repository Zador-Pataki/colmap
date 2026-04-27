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

py::dict RunEstablishFullTracks(
    CorrespondenceGraph& view_graph,
    py::dict images_py,
    const sfm_ext::TrackEstablishmentOptions& options) {
  std::unordered_map<image_t, Image> images;
  images.reserve(images_py.size());
  for (auto item : images_py) {
    images.emplace(py::cast<image_t>(item.first),
                   py::cast<Image>(item.second));
  }

  std::unordered_map<sfm_ext::track_t, Point3D> tracks_full;
  {
    py::gil_scoped_release release;
    sfm_ext::TrackEngine engine(view_graph, images, options);
    engine.EstablishFullTracks(tracks_full);
  }

  py::dict tracks_out;
  for (auto& [tid, p3d] : tracks_full) {
    tracks_out[py::cast(tid)] = py::cast(std::move(p3d));
  }
  return tracks_out;
}

py::dict RunFindTracksForProblem(
    CorrespondenceGraph& view_graph,
    py::dict images_py,
    py::dict tracks_full_py,
    const sfm_ext::TrackEstablishmentOptions& options) {
  std::unordered_map<image_t, Image> images;
  images.reserve(images_py.size());
  for (auto item : images_py) {
    images.emplace(py::cast<image_t>(item.first),
                   py::cast<Image>(item.second));
  }

  std::unordered_map<sfm_ext::track_t, Point3D> tracks_full;
  tracks_full.reserve(tracks_full_py.size());
  for (auto item : tracks_full_py) {
    tracks_full.emplace(py::cast<sfm_ext::track_t>(item.first),
                        py::cast<Point3D>(item.second));
  }

  std::unordered_map<sfm_ext::track_t, Point3D> tracks_selected;
  {
    py::gil_scoped_release release;
    sfm_ext::TrackEngine engine(view_graph, images, options);
    engine.FindTracksForProblem(tracks_full, tracks_selected);
  }

  py::dict tracks_out;
  for (auto& [tid, p3d] : tracks_selected) {
    tracks_out[py::cast(tid)] = py::cast(std::move(p3d));
  }
  return tracks_out;
}

}  // namespace

void BindTrackEstablishmentGlomap(py::module& m) {
  auto PyOpts =
      py::classh<sfm_ext::TrackEstablishmentOptions>(
          m, "TrackEstablishmentOptions")
          .def(py::init<>())
          .def_readwrite("thres_inconsistency",
                         &sfm_ext::TrackEstablishmentOptions::thres_inconsistency)
          .def_readwrite(
              "min_num_tracks_per_view",
              &sfm_ext::TrackEstablishmentOptions::min_num_tracks_per_view)
          .def_readwrite(
              "min_num_view_per_track",
              &sfm_ext::TrackEstablishmentOptions::min_num_view_per_track)
          .def_readwrite(
              "max_num_view_per_track",
              &sfm_ext::TrackEstablishmentOptions::max_num_view_per_track)
          .def_readwrite("max_num_tracks",
                         &sfm_ext::TrackEstablishmentOptions::max_num_tracks);
  MakeDataclass(PyOpts);

  m.def("establish_full_tracks",
        &RunEstablishFullTracks,
        "view_graph"_a,
        "images"_a,
        "options"_a,
        "Track establishment via TrackEngine. Returns a dict mapping "
        "track_id -> pycolmap.Point3D with .track.elements + the fork-added "
        ".track.lc_elements populated.");

  m.def("find_tracks_for_problem",
        &RunFindTracksForProblem,
        "view_graph"_a,
        "images"_a,
        "tracks_full"_a,
        "options"_a,
        "Subsample tracks for the global-positioning problem. Returns the "
        "selected subset as a fresh dict[int, pycolmap.Point3D].");
}
