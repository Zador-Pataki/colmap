#include "colmap/sfm/relative_pose_estimation.h"

#include "colmap/scene/camera.h"
#include "colmap/scene/correspondence_graph.h"
#include "colmap/scene/image.h"
#include "colmap/util/types.h"

#include "pycolmap/helpers.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

namespace {

py::dict RunRelativePoseEstimation(CorrespondenceGraph& correspondence_graph,
                                   py::dict cameras_py,
                                   py::dict images_py) {
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

  RelativePoseEstimationOptions options;
  {
    py::gil_scoped_release release;
    EstimateRelativePoses(correspondence_graph, cameras, images, options);
  }

  py::dict cameras_out;
  for (auto& [cid, cam] : cameras) {
    cameras_out[py::cast(cid)] = py::cast(cam);
  }
  py::dict images_out;
  for (auto& [iid, img] : images) {
    images_out[py::cast(iid)] = py::cast(img);
  }
  py::dict output;
  output["correspondence_graph"] = correspondence_graph;
  output["cameras"] = cameras_out;
  output["images"] = images_out;
  return output;
}

}  // namespace

void BindRelativePoseEstimation(py::module& m) {
  m.def("run_relative_pose_estimation",
        &RunRelativePoseEstimation,
        "correspondence_graph"_a,
        "cameras"_a,
        "images"_a,
        "Estimate relative poses for every valid pair via "
        "poselib::estimate_relative_pose on the matched 2D correspondences. "
        "Writes image_pair.two_view_geometry.cam2_from_cam1; on per-pair "
        "PoseLib failure, marks is_valid=false. Returns dict with mutated "
        "correspondence_graph + cameras + images.");
}
