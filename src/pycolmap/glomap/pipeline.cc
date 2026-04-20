// Pipeline Python bindings for the glomap pipeline.
//
// Hand-written py::dict lambdas that match fork pyglomap's bindings
// byte-for-byte in argument order, kwargs, and return shape. mpsfm consumes
// these dicts directly — do NOT convert to bare-function bindings.
#include "colmap/estimators/glomap/bundle_adjustment.h"
#include "colmap/estimators/glomap/global_positioner.h"
#include "colmap/estimators/glomap/iteration_callback.h"
#include "colmap/estimators/glomap/relpose_estimation.h"
#include "colmap/estimators/glomap/rotation_averaging.h"
#include "colmap/estimators/glomap/track_establishment.h"
#include "colmap/estimators/glomap/track_filter.h"
#include "colmap/estimators/glomap/view_graph_calibration.h"
#include "colmap/glomap/camera.h"
#include "colmap/glomap/image.h"
#include "colmap/glomap/track.h"
#include "colmap/glomap/view_graph.h"
#include "colmap/glomap/processors/image_undistorter.h"
#include "colmap/glomap/processors/reconstruction_normalizer.h"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <map>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace py = pybind11;
using namespace colmap::glomap;
using namespace pybind11::literals;

namespace {

// ---------- rotation averaging ----------

std::tuple<bool,
           std::unordered_map<image_t, Image>,
           std::unordered_map<image_pair_t, double>>
RunRotationAveraging(ViewGraph& view_graph,
                     std::unordered_map<image_t, Image>& images,
                     RotationEstimatorOptions options) {
  view_graph.KeepLargestConnectedComponents(images);
  RotationEstimator ra_engine(options);
  auto estimation_result = ra_engine.EstimateRotations(view_graph, images);
  bool success = estimation_result.first;
  std::unordered_map<image_pair_t, double> final_weights =
      estimation_result.second;
  return std::make_tuple(success, images, final_weights);
}

// ---------- relative pose estimation ----------

py::dict RunEstimateRelativePoses(
    ViewGraph& view_graph,
    std::unordered_map<camera_t, Camera>& cameras,
    std::unordered_map<image_t, Image>& images) {
  RelativePoseEstimationOptions options;
  UndistortImages(cameras, images, /*clean_points=*/true);
  EstimateRelativePoses(view_graph, cameras, images, options);

  py::dict output;
  output["view_graph"] = view_graph;
  output["cameras"] = cameras;
  output["images"] = images;
  return output;
}

// ---------- view graph calibration ----------

py::dict RunViewGraphCalibration(
    ViewGraph& view_graph,
    std::unordered_map<camera_t, Camera>& cameras,
    std::unordered_map<image_t, Image>& images,
    const ViewGraphCalibratorOptions& options) {
  ViewGraphCalibrator vgcalib_engine(options);
  if (!vgcalib_engine.Solve(view_graph, cameras, images)) {
    throw std::runtime_error("Failed to solve view graph calibration.");
  }

  py::dict output;
  output["view_graph"] = view_graph;
  output["cameras"] = cameras;
  output["images"] = images;
  return output;
}

// ---------- global positioning ----------

py::dict RunGlobalPositioning(
    ViewGraph& view_graph,
    std::unordered_map<camera_t, Camera>& cameras,
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<track_t, Track>& tracks,
    GlobalPositionerOptions options,
    const std::vector<image_t>& image_ids_to_optimize,
    const std::map<image_t, double>& initial_dmap_scales,
    const std::vector<image_pair_t>& consecutive_pair_ids,
    py::object iteration_callback) {
  view_graph.KeepLargestConnectedComponents(images);
  UndistortImages(cameras, images, /*clean_points=*/false);

  GlobalPositioner gp_engine(options);

  IterationCallbackFn cpp_callback = nullptr;
  if (!iteration_callback.is_none()) {
    cpp_callback = [iteration_callback](const IterationData& data) {
      py::gil_scoped_acquire gil;
      py::dict py_data;
      py_data["iteration"] = data.iteration;
      py_data["cost"] = data.cost;
      py_data["cost_change"] = data.cost_change;
      py_data["gradient_norm"] = data.gradient_norm;
      py_data["images"] = data.images;
      py_data["tracks"] = data.tracks;
      py::list lc_edges;
      if (data.lc_edges != nullptr) {
        for (const auto& edge : *data.lc_edges) {
          py::dict e;
          e["image_id1"] = edge.image_id1;
          e["image_id2"] = edge.image_id2;
          e["shared_tracks"] = edge.shared_track_count;
          lc_edges.append(e);
        }
      }
      py_data["lc_edges"] = lc_edges;
      iteration_callback(py_data);
    };
  }

  std::unordered_set<image_t> image_ids_set(image_ids_to_optimize.begin(),
                                            image_ids_to_optimize.end());
  gp_engine.Solve(view_graph,
                  cameras,
                  images,
                  tracks,
                  image_ids_set,
                  initial_dmap_scales,
                  consecutive_pair_ids,
                  cpp_callback);

  py::dict output;
  output["images"] = images;
  output["tracks"] = tracks;
  output["scales"] = gp_engine.GetScales();
  output["dmap_scales"] = gp_engine.GetDepthMapScales();
  output["dmap_scale_map"] = gp_engine.GetDepthMapScaleMap();
  output["dmap_scale_map_nested"] = gp_engine.GetDepthMapScaleMapNested();
  output["geometry_only_constraints"] = gp_engine.GetGeometryOnlyConstraints();
  output["depth_constraints"] = gp_engine.GetDepthConstraints();
  return output;
}

// ---------- bundle adjustment ----------

py::dict RunBundleAdjustment(ViewGraph& view_graph,
                             std::unordered_map<camera_t, Camera>& cameras,
                             std::unordered_map<image_t, Image>& images,
                             std::unordered_map<track_t, Track>& tracks,
                             BundleAdjustmentOptions options,
                             py::object iteration_callback) {
  view_graph.KeepLargestConnectedComponents(images);

  IterationCallbackFn cpp_callback = nullptr;
  if (!iteration_callback.is_none()) {
    cpp_callback = [iteration_callback](const IterationData& data) {
      py::gil_scoped_acquire gil;
      py::dict py_data;
      py_data["iteration"] = data.iteration;
      py_data["cost"] = data.cost;
      py_data["cost_change"] = data.cost_change;
      py_data["gradient_norm"] = data.gradient_norm;
      py_data["images"] = data.images;
      py_data["tracks"] = data.tracks;
      py::list lc_edges;
      if (data.lc_edges != nullptr) {
        for (const auto& edge : *data.lc_edges) {
          py::dict e;
          e["image_id1"] = edge.image_id1;
          e["image_id2"] = edge.image_id2;
          e["shared_tracks"] = edge.shared_track_count;
          lc_edges.append(e);
        }
      }
      py_data["lc_edges"] = lc_edges;
      iteration_callback(py_data);
    };
  }

  BundleAdjuster ba_engine(options);
  ba_engine.Solve(view_graph, cameras, images, tracks, cpp_callback);

  NormalizeReconstruction(cameras, images, tracks);
  UndistortImages(cameras, images, /*clean_points=*/true);

  py::dict output;
  output["cameras"] = cameras;
  output["images"] = images;
  output["tracks"] = tracks;
  return output;
}

// ---------- track filter (by reprojection — complements angle filters) ----------

py::dict RunFilterTracksByReprojection(
    ViewGraph& view_graph,
    std::unordered_map<camera_t, Camera>& cameras,
    std::unordered_map<image_t, Image>& images,
    std::unordered_map<track_t, Track>& tracks,
    double max_reprojection_error,
    bool in_normalized_image) {
  int counter = TrackFilter::FilterTracksByReprojection(
      view_graph, cameras, images, tracks, max_reprojection_error,
      in_normalized_image);
  py::dict output;
  output["tracks"] = tracks;
  output["counter"] = counter;
  return output;
}

}  // namespace

void BindGlomapPipeline(py::module& m) {
  m.def("run_rotation_averaging",
        &RunRotationAveraging,
        "view_graph"_a, "images"_a,
        py::arg_v("options", RotationEstimatorOptions(),
                  "RotationEstimatorOptions()"),
        py::call_guard<py::gil_scoped_release>(),
        "Estimates global rotations. Returns (success, images, weights).");

  m.def("run_relative_pose_estimation",
        &RunEstimateRelativePoses,
        "view_graph"_a, "cameras"_a, "images"_a,
        "Run relative pose estimation with default options. "
        "Returns dict with 'view_graph', 'cameras', 'images'.");

  m.def("run_view_graph_calibration",
        &RunViewGraphCalibration,
        "view_graph"_a, "cameras"_a, "images"_a,
        py::arg_v("options", ViewGraphCalibratorOptions(),
                  "ViewGraphCalibratorOptions()"),
        "Run view graph calibration. "
        "Returns dict with 'view_graph', 'cameras', 'images'.");

  m.def("run_global_positioning",
        &RunGlobalPositioning,
        "view_graph"_a, "cameras"_a, "images"_a, "tracks"_a,
        py::arg_v("options", GlobalPositionerOptions(),
                  "GlobalPositionerOptions()"),
        py::arg_v("image_ids_to_optimize", std::vector<image_t>(), ""),
        py::arg_v("initial_dmap_scales", std::map<image_t, double>(), ""),
        py::arg_v("consecutive_pair_ids", std::vector<image_pair_t>(), ""),
        py::arg_v("iteration_callback", py::none(), ""),
        "Run global positioning. Returns dict with 'images', 'tracks', "
        "'scales', 'dmap_scales', 'dmap_scale_map', 'dmap_scale_map_nested', "
        "'geometry_only_constraints', 'depth_constraints'.");

  m.def("run_bundle_adjustment",
        &RunBundleAdjustment,
        "view_graph"_a, "cameras"_a, "images"_a, "tracks"_a,
        py::arg_v("options", BundleAdjustmentOptions(),
                  "BundleAdjustmentOptions()"),
        py::arg_v("iteration_callback", py::none(), ""),
        "Run bundle adjustment. Returns dict with 'cameras', 'images', 'tracks'.");

  m.def("run_track_filter",
        &RunFilterTracksByReprojection,
        "view_graph"_a, "cameras"_a, "images"_a, "tracks"_a,
        "max_reprojection_error"_a = 1e-2,
        "in_normalized_image"_a = true,
        "Filter tracks by reprojection error. "
        "Returns dict with 'tracks', 'counter'.");

  // run_track_establishment / run_track_retriangulation not bound:
  // - establish_full_tracks / find_tracks_for_problem are module-level (§14).
  // - retriangulation deferred (§11).
}
