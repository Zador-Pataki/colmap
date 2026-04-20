#include "colmap/estimators/glomap/inlier_threshold_options.h"
#include "colmap/estimators/glomap/rotation_estimator_options.h"
#include "colmap/estimators/glomap/track_establishment.h"
#include "colmap/estimators/glomap/track_establishment_options.h"
#include "colmap/estimators/glomap/track_filter.h"
#include "colmap/glomap/camera.h"
#include "colmap/glomap/image.h"
#include "colmap/glomap/processors/image_pair_inliers.h"
#include "colmap/glomap/processors/image_undistorter.h"
#include "colmap/glomap/processors/relpose_filter.h"
#include "colmap/glomap/processors/view_graph_manipulation.h"
#include "colmap/glomap/view_graph.h"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace colmap::glomap;

namespace {

py::dict RunUpdateImagePairsConfig(
    ViewGraph& view_graph,
    const std::unordered_map<camera_t, Camera>& cameras,
    const std::unordered_map<image_t, Image>& images) {
  ViewGraphManipulater::UpdateImagePairsConfig(view_graph, cameras, images);
  py::dict output;
  output["view_graph"] = view_graph;
  output["cameras"] = cameras;
  output["images"] = images;
  return output;
}

py::dict RunDecomposeRelPose(ViewGraph& view_graph,
                             std::unordered_map<camera_t, Camera>& cameras,
                             std::unordered_map<image_t, Image>& images) {
  ViewGraphManipulater::DecomposeRelPose(view_graph, cameras, images);
  py::dict output;
  output["view_graph"] = view_graph;
  output["cameras"] = cameras;
  output["images"] = images;
  return output;
}

std::unordered_map<track_t, Track> EstablishFullTracks(
    const ViewGraph& view_graph,
    const std::unordered_map<image_t, Image>& images,
    TrackEstablishmentOptions options,
    const std::vector<image_t>& image_ids_to_process) {
  std::unordered_map<track_t, Track> tracks;
  TrackEngine track_engine(view_graph, images, options);
  std::unordered_set<image_t> image_ids_set(image_ids_to_process.begin(),
                                            image_ids_to_process.end());
  track_engine.SetImageIdsToProcess(image_ids_set);
  track_engine.EstablishFullTracks(tracks);
  return tracks;
}

std::unordered_map<track_t, Track> FindTracksForProblem(
    const ViewGraph& view_graph,
    const std::unordered_map<image_t, Image>& images,
    const std::unordered_map<track_t, Track>& tracks_full,
    TrackEstablishmentOptions options) {
  std::unordered_map<track_t, Track> tracks_selected;
  TrackEngine track_engine(view_graph, images, options);
  track_engine.FindTracksForProblem(tracks_full, tracks_selected);
  return tracks_selected;
}

py::dict RunFilterTracksByAngle(
    const ViewGraph& view_graph,
    const std::unordered_map<camera_t, Camera>& cameras,
    const std::unordered_map<image_t, Image>& images,
    std::unordered_map<track_t, Track>& tracks,
    double max_angle_error) {
  int counter = TrackFilter::FilterTracksByAngle(
      view_graph, cameras, images, tracks, max_angle_error);
  py::dict output;
  output["tracks"] = tracks;
  output["counter"] = counter;
  return output;
}

py::dict RunFilterTrackTriangulationAngle(
    const ViewGraph& view_graph,
    const std::unordered_map<image_t, Image>& images,
    std::unordered_map<track_t, Track>& tracks,
    double min_angle) {
  int counter = TrackFilter::FilterTrackTriangulationAngle(
      view_graph, images, tracks, min_angle);
  py::dict output;
  output["tracks"] = tracks;
  output["counter"] = counter;
  return output;
}

py::dict ExtractAllImageData(
    const std::unordered_map<image_t, Image>& images) {
  py::dict result;
  for (const auto& [imid, im] : images) {
    const size_t n_feat = im.features.size();

    py::array_t<double> features({(py::ssize_t)n_feat, (py::ssize_t)2});
    auto feat_mut = features.mutable_unchecked<2>();
    for (size_t i = 0; i < n_feat; i++) {
      feat_mut(i, 0) = im.features[i][0];
      feat_mut(i, 1) = im.features[i][1];
    }

    py::array_t<double> depth_priors(im.depth_priors.size());
    auto dp_mut = depth_priors.mutable_unchecked<1>();
    for (size_t i = 0; i < im.depth_priors.size(); i++) {
      dp_mut(i) = im.depth_priors[i];
    }

    py::array_t<bool> dpv(im.depth_prior_validity.size());
    auto dpv_mut = dpv.mutable_unchecked<1>();
    for (size_t i = 0; i < im.depth_prior_validity.size(); i++) {
      dpv_mut(i) = im.depth_prior_validity[i];
    }

    py::dict img_dict;
    img_dict["features"] = std::move(features);
    img_dict["depth_priors"] = std::move(depth_priors);
    img_dict["depth_prior_validity"] = std::move(dpv);
    img_dict["camera_id"] = im.camera_id;
    img_dict["file_name"] = py::str(im.file_name);

    result[py::int_(imid)] = std::move(img_dict);
  }
  return result;
}

}  // namespace

void BindGlomapFreeFunctions(py::module& m) {
  // ----- Top-level aliases for nested enums (fork-style usage) -----
  // Re-expose GlobalPositionerOptions::ConstraintType / PointConstraintType at
  // module scope so `pycolmap.glomap.ConstraintType` (fork idiom) works in
  // addition to the nested `GlobalPositionerOptions.ConstraintType`.
  if (py::hasattr(m, "GlobalPositionerOptions")) {
    py::object PyGP = m.attr("GlobalPositionerOptions");
    m.attr("ConstraintType") = PyGP.attr("ConstraintType");
    m.attr("PointConstraintType") = PyGP.attr("PointConstraintType");
  }
  if (py::hasattr(m, "RotationEstimatorOptions")) {
    m.attr("RotationWeightType") =
        m.attr("RotationEstimatorOptions").attr("WeightType");
  }

  // ----- Free functions -----
  m.def("extract_all_image_data",
        &ExtractAllImageData,
        py::arg("images"),
        "Batch extract image data (features, depth_priors, "
        "depth_prior_validity, camera_id, file_name) for all images. "
        "Returns dict[image_id -> dict].");

  m.def("image_pairs_inlier_count",
        &ImagePairsInlierCount,
        py::arg("view_graph"),
        py::arg("cameras"),
        py::arg("images"),
        py::arg("options"),
        py::arg("clean_inliers") = true,
        "Computes the number of inliers for each image pair in the view graph, "
        "modifying the ViewGraph object in place.");

  m.def("filter_inlier_num",
        &RelPoseFilter::FilterInlierNum,
        py::arg("view_graph"),
        py::arg("min_inlier_num") = 30,
        "Filter relative pose based on number of inliers.");

  m.def("filter_inlier_ratio",
        &RelPoseFilter::FilterInlierRatio,
        py::arg("view_graph"),
        py::arg("min_inlier_ratio") = 0.25,
        "Filter relative pose based on rate of inliers.");

  m.def("update_image_pairs_config",
        &RunUpdateImagePairsConfig,
        py::arg("view_graph"),
        py::arg("cameras"),
        py::arg("images"),
        "Update image pairs config.");

  m.def("decompose_rel_pose",
        &RunDecomposeRelPose,
        py::arg("view_graph"),
        py::arg("cameras"),
        py::arg("images"),
        "Decompose relative pose.");

  m.def("establish_full_tracks",
        &EstablishFullTracks,
        py::arg("view_graph"),
        py::arg("images"),
        py::arg("options") = TrackEstablishmentOptions(),
        py::arg("image_ids_to_process") = std::vector<image_t>(),
        "Establish full tracks from the view graph.");

  m.def("find_tracks_for_problem",
        &FindTracksForProblem,
        py::arg("view_graph"),
        py::arg("images"),
        py::arg("tracks_full"),
        py::arg("options") = TrackEstablishmentOptions(),
        "Find tracks for the problem.");

  m.def("filter_tracks_by_angle",
        &RunFilterTracksByAngle,
        py::arg("view_graph"),
        py::arg("cameras"),
        py::arg("images"),
        py::arg("tracks"),
        py::arg("max_angle_error") = 1.0,
        "Filter tracks by angle error.");

  m.def("filter_track_triangulation_angle",
        &RunFilterTrackTriangulationAngle,
        py::arg("view_graph"),
        py::arg("images"),
        py::arg("tracks"),
        py::arg("min_angle") = 1.0,
        "Filter tracks by triangulation angle.");

  m.def(
      "undistort_images",
      [](std::unordered_map<camera_t, Camera> cameras,
         std::unordered_map<image_t, Image> images,
         bool clean_points) {
        py::gil_scoped_release release;
        UndistortImages(cameras, images, clean_points);
        return images;
      },
      py::arg("cameras"),
      py::arg("images"),
      py::arg("clean_points") = true,
      "Undistorts feature points for all images using camera parameters, "
      "and returns the modified images dictionary.");

  // write_glomap_reconstruction: deferred. Fork's WriteGlomapReconstruction
  // depends on colmap_io.cc/colmap_converter.cc, which hit multiple colmap4
  // API divergences (PosePrior::IsValid, Database::PairIdToImagePair,
  // Image::SetCamFromWorld, std::optional<Rigid3d> on TwoViewGeometry,
  // refine_extrinsics → refine_extra_params). Bind a stub that raises
  // NotImplementedError so callers see a clean error instead of AttributeError.
  m.def(
      "write_glomap_reconstruction",
      [](const std::string&,
         const std::unordered_map<camera_t, Camera>&,
         const std::unordered_map<image_t, Image>&,
         const std::unordered_map<track_t, Track>&,
         const std::string&,
         const std::string&) {
        throw std::runtime_error(
            "write_glomap_reconstruction is not yet ported to colmap4. "
            "colmap_converter.cc has multiple API divergences that must be "
            "reconciled first (see §11 deferral note in "
            "src/colmap/estimators/glomap/CMakeLists.txt).");
      },
      py::arg("reconstruction_path"),
      py::arg("cameras"),
      py::arg("images"),
      py::arg("tracks"),
      py::arg("output_format") = "bin",
      py::arg("image_path") = "",
      "DEFERRED (§14) — not yet ported to colmap4. See §11 deferral note.");
}
