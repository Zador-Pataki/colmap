#include "colmap/glomap/track.h"

#include "pycolmap/glomap/types.h"
#include "pycolmap/helpers.h"

#include <sstream>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;
using colmap::glomap::Observation;
using colmap::glomap::Track;
using colmap::glomap::TrackMap;
using colmap::glomap::track_t;

void BindGlomapTrack(py::module& m) {
  py::classh<Track> PyTrack(m, "Track");
  PyTrack.def(py::init<>())
      .def_property_readonly(
          "track_id",
          [](const Track& self) -> track_t { return self.track_id; },
          "Unique track identifier.")
      .def_readwrite("xyz", &Track::xyz, "The 3D point.")
      .def_readwrite("color", &Track::color, "RGB color (currently unused).")
      .def_property(
          "observations",
          [](const Track& self) -> std::vector<Observation> {
            return self.observations;
          },
          [](Track& self, const std::vector<Observation>& value) {
            self.observations = value;
          },
          "Observations of the track.")
      .def_property(
          "lc_observations",
          [](const Track& self) -> std::vector<Observation> {
            return self.lc_observations;
          },
          [](Track& self, const std::vector<Observation>& value) {
            self.lc_observations = value;
          },
          "Loop-closure-specific observations of the track.")
      .def_readwrite("is_initialized",
                     &Track::is_initialized,
                     "Whether the point is initialized.")
      .def("__repr__", [](const Track& self) {
        std::ostringstream ss;
        ss << "Track(" << self.track_id
           << ", num_observations=" << self.observations.size() << ")";
        return ss.str();
      });
  MakeDataclass(PyTrack);

  py::bind_map<TrackMap>(m, "MapTrackIdToTrack");
}
