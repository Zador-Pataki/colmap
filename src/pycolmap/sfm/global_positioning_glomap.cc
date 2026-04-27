#include "colmap/sfm/global_positioning_glomap.h"

#include "colmap/scene/camera.h"
#include "colmap/scene/correspondence_graph.h"
#include "colmap/scene/image.h"
#include "colmap/scene/point3d.h"
#include "colmap/scene/track.h"  // colmap::TrackElement (for Point3D round-trip)
#include "colmap/util/types.h"
// glomap_ra::Track / glomap_ra::track_t / glomap_ra::feature_t / glomap_ra::Observation come from
// colmap/sfm/track_establishment_glomap.h (transitively via the GP header).

#include "pycolmap/helpers.h"

#include <ceres/loss_function.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Core>

#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

namespace {

// pycolmap.Point3D <-> colmap::glomap_ra::Track conversion. Mirrors the
// pattern used by track_establishment_glomap.cc::ToPoint3D so videosfm
// can pass its dict[track_id, pycolmap.Point3D] into pycolmap.glomap_ra
// entrypoints unchanged.
glomap_ra::Track ToGlomapRaTrack(glomap_ra::track_t tid, const Point3D& p3d) {
  glomap_ra::Track t;
  t.track_id = tid;
  t.xyz = p3d.xyz;
  t.color = p3d.color;
  t.is_initialized = p3d.is_initialized;
  t.observations.reserve(p3d.track.Length());
  for (const auto& el : p3d.track.Elements()) {
    t.observations.emplace_back(el.image_id,
                                static_cast<glomap_ra::feature_t>(el.point2D_idx));
  }
  t.lc_observations.reserve(p3d.track.lc_elements.size());
  for (const auto& el : p3d.track.lc_elements) {
    t.lc_observations.emplace_back(el.image_id,
                                   static_cast<glomap_ra::feature_t>(el.point2D_idx));
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

py::dict RunGlobalPositioning(
    CorrespondenceGraph& view_graph,
    py::dict cameras_py,
    py::dict images_py,
    py::dict tracks_py,
    glomap_ra::GlobalPositionerOptions options,
    const std::vector<image_t>& image_ids_to_optimize,
    const std::map<image_t, double>& initial_dmap_scales,
    const std::vector<image_pair_t>& consecutive_pair_ids) {
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

  std::unordered_set<image_t> image_ids_set(image_ids_to_optimize.begin(),
                                            image_ids_to_optimize.end());

  glomap_ra::GlobalPositioner positioner(options);
  {
    py::gil_scoped_release release;
    positioner.Solve(view_graph,
                     cameras,
                     images,
                     tracks,
                     image_ids_set,
                     initial_dmap_scales,
                     consecutive_pair_ids);
  }

  py::dict images_out;
  for (auto& [iid, img] : images) {
    images_out[py::cast(iid)] = py::cast(img);
  }
  py::dict tracks_out;
  for (auto& [tid, trk] : tracks) {
    Point3D p3d = FromGlomapRaTrack(trk);
    tracks_out[py::cast(tid)] = py::cast(p3d);
  }

  py::dict output;
  output["images"] = images_out;
  output["tracks"] = tracks_out;
  output["scales"] = positioner.GetScales();
  output["dmap_scales"] = positioner.GetDepthMapScales();
  output["dmap_scale_map"] = positioner.GetDepthMapScaleMap();
  output["dmap_scale_map_nested"] = positioner.GetDepthMapScaleMapNested();
  return output;
}

}  // namespace

// Get-or-create the `glomap_ra` submodule. Both BindGlobalPositionerOptions
// and BindGlobalPositioningGlomap call this so the submodule is shared
// regardless of registration order.
py::module GetOrCreateGlomapRaModule(py::module& m) {
  if (py::hasattr(m, "glomap_ra")) {
    return m.attr("glomap_ra").cast<py::module>();
  }
  return m.def_submodule(
      "glomap_ra",
      "Glomap-fork ports under colmap::glomap_ra namespace. Mirrors the "
      "pyglomap public API for migration; will be folded into colmap-native "
      "bindings once the port stabilises.");
}

void BindGlobalPositionerOptions(py::module& m) {
  py::module m_glomap_ra = GetOrCreateGlomapRaModule(m);

  py::enum_<glomap_ra::GlobalPositionerOptions::ConstraintType>(m_glomap_ra,
                                                     "ConstraintType")
      .value("ONLY_POINTS",
             glomap_ra::GlobalPositionerOptions::ConstraintType::ONLY_POINTS)
      .export_values();

  py::enum_<glomap_ra::GlobalPositionerOptions::PointConstraintType>(m_glomap_ra,
                                                          "PointConstraintType")
      .value("GEOMETRY_ONLY",
             glomap_ra::GlobalPositionerOptions::PointConstraintType::GEOMETRY_ONLY)
      .value("SPLIT_METRIC_DEPTH",
             glomap_ra::GlobalPositionerOptions::PointConstraintType::SPLIT_METRIC_DEPTH)
      .export_values();

  py::classh<glomap_ra::GlobalPositionerOptions> PyGlobalPositionerOptions(
      m_glomap_ra, "GlobalPositionerOptions");

  PyGlobalPositionerOptions.def(py::init<>())
      .def_readwrite("thres_loss_function",
                     &glomap_ra::GlobalPositionerOptions::thres_loss_function,
                     "The threshold for the loss function.")
      .def_readwrite(
          "generate_scales",
          &glomap_ra::GlobalPositionerOptions::generate_scales,
          "Whether to initialize per-observation scales to 1.0 (true) "
          "or compute from current camera/track positions (false). "
          "Set to false when using use_init=true for warm-starting.")
      .def_readwrite(
          "random_init_scale",
          &glomap_ra::GlobalPositionerOptions::random_init_scale,
          "Scale for random initialization (in meters if working in metric "
          "scale). "
          "Random positions/points will be initialized in a cube from "
          "[-random_init_scale, +random_init_scale] in each dimension. "
          "Default: 100.0 meters (200m x 200m x 200m cube).")
      .def_readwrite("optimize_depth_map_scales",
                     &glomap_ra::GlobalPositionerOptions::optimize_depth_map_scales,
                     "Whether to optimize depth map scale parameters.")
      .def_readwrite("min_num_view_per_track",
                     &glomap_ra::GlobalPositionerOptions::min_num_view_per_track,
                     "Minimum number of views required for a track to be used.")
      .def_readwrite(
          "use_init",
          &glomap_ra::GlobalPositionerOptions::use_init,
          "If true, use existing camera positions and track xyz values "
          "for initialization instead of random initialization.")
      .def_readwrite("constraint_type",
                     &glomap_ra::GlobalPositionerOptions::constraint_type,
                     "Overall constraint balancing strategy.")
      .def_readwrite("point_constraint_type",
                     &glomap_ra::GlobalPositionerOptions::point_constraint_type,
                     "How point constraints incorporate depth priors.")
      .def_readwrite(
          "scale_regularization_sigma",
          &glomap_ra::GlobalPositionerOptions::scale_regularization_sigma,
          "Standard deviation (uncertainty) for scale prior. "
          "Smaller values = stronger regularization toward metric depth. "
          "Used with SCALED_METRIC_DEPTH.")
      .def_readwrite(
          "scale_prior_stddev",
          &glomap_ra::GlobalPositionerOptions::scale_prior_stddev,
          "Standard deviation for depth map scale prior regularization. "
          "Pulls per-observation depth map scales toward 1.0. "
          "Smaller values = stronger regularization (more trust in initial "
          "scale=1.0).")
      .def_readwrite(
          "use_log_scale_for_depth_map_scales",
          &glomap_ra::GlobalPositionerOptions::use_log_scale_for_depth_map_scales,
          "If true, use log-space parameterization for depth map scales "
          "(optimizes log_scale where scale = exp(log_scale)). This makes "
          "regularization less aggressive for large scale errors. Default: "
          "true (recommended).")
      .def_readwrite(
          "use_log_residual_for_depth",
          &glomap_ra::GlobalPositionerOptions::use_log_residual_for_depth,
          "If true, use log-space residual for depth errors when points are "
          "in front of camera (z_est > 0). Points behind camera always use "
          "linear residual. This treats relative depth errors equally "
          "regardless of absolute depth. Default: false (backward "
          "compatible).")
      .def_readwrite(
          "zero_residual_behind_camera",
          &glomap_ra::GlobalPositionerOptions::zero_residual_behind_camera,
          "If true, set depth residuals to zero for points behind camera "
          "(z_est <= 0). This effectively ignores depth constraints for "
          "points behind the camera, which can help during initialization "
          "when many points start behind cameras. Default: false (backward "
          "compatible).")
      .def_readwrite(
          "smooth_log_linear_transition",
          &glomap_ra::GlobalPositionerOptions::smooth_log_linear_transition,
          "If true, use smooth transition from log to linear residual at "
          "log_linear_threshold. For z > threshold: log-space residual. "
          "For z <= threshold: linear residual with gradient=1, continuous "
          "at threshold. This avoids gradient discontinuity at z=0 when "
          "use_log_residual_for_depth=true. Default: false (backward "
          "compatible).")
      .def_readwrite(
          "log_linear_threshold",
          &glomap_ra::GlobalPositionerOptions::log_linear_threshold,
          "Threshold for smooth log-linear transition (in depth units, "
          "e.g., meters). Only used if smooth_log_linear_transition=true. "
          "Default: 0.1")
      .def_readwrite(
          "filter_depth_outliers",
          &glomap_ra::GlobalPositionerOptions::filter_depth_outliers,
          "If true, pre-compute depth outliers before optimization and "
          "disable depth constraints for them. Outliers are detected using "
          "log-space check: |log(z_est) - log(depth_prior)| < 3 * log(1 + "
          "metric_std/depth_prior) where metric_std = relative_stddev * "
          "depth_prior. Outliers will only use geometry constraints. "
          "Default: false (backward compatible).")
      .def_readwrite(
          "consecutive_trivial",
          &glomap_ra::GlobalPositionerOptions::consecutive_trivial,
          "If true, use trivial loss functions for consecutive constraints. "
          "Otherwise use the same loss function as regular constraints.")
      .def_property(
          "num_threads",
          [](const glomap_ra::GlobalPositionerOptions& self) -> int {
            return self.solver_options.num_threads;
          },
          [](glomap_ra::GlobalPositionerOptions& self, const int value) {
            self.solver_options.num_threads = value;
          },
          "Number of threads for the Ceres solver. Set to 1 for "
          "byte-deterministic GP results (parallel residual evaluation is "
          "not deterministic at floating-point precision).")
      .def_property(
          "max_num_iterations",
          [](const glomap_ra::GlobalPositionerOptions& self) -> int {
            return self.solver_options.max_num_iterations;
          },
          [](glomap_ra::GlobalPositionerOptions& self, const int value) {
            self.solver_options.max_num_iterations = value;
          },
          "Maximum number of Ceres solver iterations.")
      .def_property(
          "function_tolerance",
          [](const glomap_ra::GlobalPositionerOptions& self) {
            return self.solver_options.function_tolerance;
          },
          [](glomap_ra::GlobalPositionerOptions& self, double value) {
            self.solver_options.function_tolerance = value;
          },
          "Convergence tolerance based on cost function change. "
          "Solver terminates if cost change is less than this value.")
      .def_property(
          "gradient_tolerance",
          [](const glomap_ra::GlobalPositionerOptions& self) {
            return self.solver_options.gradient_tolerance;
          },
          [](glomap_ra::GlobalPositionerOptions& self, double value) {
            self.solver_options.gradient_tolerance = value;
          },
          "Convergence tolerance based on gradient norm. "
          "Solver terminates if gradient norm is less than this value.")
      .def_property(
          "parameter_tolerance",
          [](const glomap_ra::GlobalPositionerOptions& self) {
            return self.solver_options.parameter_tolerance;
          },
          [](glomap_ra::GlobalPositionerOptions& self, double value) {
            self.solver_options.parameter_tolerance = value;
          },
          "Convergence tolerance based on parameter change. "
          "Solver terminates if parameter change is less than this value.")
      .def_property(
          "loss_normal_geometry",
          [](const glomap_ra::GlobalPositionerOptions& self) -> py::dict {
            py::dict d;
            d["name"] = self.loss_normal_geometry.name;
            d["scale"] = self.loss_normal_geometry.scale;
            d["weight"] = self.loss_normal_geometry.weight;
            return d;
          },
          [](glomap_ra::GlobalPositionerOptions& self, const py::dict& d) {
            self.loss_normal_geometry.name =
                d.contains("name") ? py::cast<std::string>(d["name"]) : "huber";
            self.loss_normal_geometry.scale =
                d.contains("scale") ? py::cast<double>(d["scale"]) : 1.0;
            self.loss_normal_geometry.weight =
                d.contains("weight") ? py::cast<double>(d["weight"]) : 1.0;
          },
          "Loss function config for normal geometry residuals (dict with "
          "'name', 'scale', 'weight')")
      .def_property(
          "loss_normal_depth",
          [](const glomap_ra::GlobalPositionerOptions& self) -> py::dict {
            py::dict d;
            d["name"] = self.loss_normal_depth.name;
            d["scale"] = self.loss_normal_depth.scale;
            d["weight"] = self.loss_normal_depth.weight;
            return d;
          },
          [](glomap_ra::GlobalPositionerOptions& self, const py::dict& d) {
            self.loss_normal_depth.name =
                d.contains("name") ? py::cast<std::string>(d["name"]) : "huber";
            self.loss_normal_depth.scale =
                d.contains("scale") ? py::cast<double>(d["scale"]) : 1.0;
            self.loss_normal_depth.weight =
                d.contains("weight") ? py::cast<double>(d["weight"]) : 1.0;
          },
          "Loss function config for normal depth residuals (dict with "
          "'name', 'scale', 'weight')")
      .def_property(
          "loss_lc_geometry",
          [](const glomap_ra::GlobalPositionerOptions& self) -> py::dict {
            py::dict d;
            d["name"] = self.loss_lc_geometry.name;
            d["scale"] = self.loss_lc_geometry.scale;
            d["weight"] = self.loss_lc_geometry.weight;
            return d;
          },
          [](glomap_ra::GlobalPositionerOptions& self, const py::dict& d) {
            self.loss_lc_geometry.name =
                d.contains("name") ? py::cast<std::string>(d["name"]) : "huber";
            self.loss_lc_geometry.scale =
                d.contains("scale") ? py::cast<double>(d["scale"]) : 1.0;
            self.loss_lc_geometry.weight =
                d.contains("weight") ? py::cast<double>(d["weight"]) : 1.0;
          },
          "Loss function config for LC geometry residuals (dict with 'name', "
          "'scale', 'weight')")
      .def_property(
          "loss_lc_depth",
          [](const glomap_ra::GlobalPositionerOptions& self) -> py::dict {
            py::dict d;
            d["name"] = self.loss_lc_depth.name;
            d["scale"] = self.loss_lc_depth.scale;
            d["weight"] = self.loss_lc_depth.weight;
            return d;
          },
          [](glomap_ra::GlobalPositionerOptions& self, const py::dict& d) {
            self.loss_lc_depth.name =
                d.contains("name") ? py::cast<std::string>(d["name"]) : "huber";
            self.loss_lc_depth.scale =
                d.contains("scale") ? py::cast<double>(d["scale"]) : 1.0;
            self.loss_lc_depth.weight =
                d.contains("weight") ? py::cast<double>(d["weight"]) : 1.0;
          },
          "Loss function config for LC depth residuals (dict with 'name', "
          "'scale', 'weight')")
      .def_property(
          "loss_scale_prior",
          [](const glomap_ra::GlobalPositionerOptions& self) -> py::dict {
            py::dict d;
            d["name"] = self.loss_scale_prior.name;
            d["scale"] = self.loss_scale_prior.scale;
            d["weight"] = self.loss_scale_prior.weight;
            return d;
          },
          [](glomap_ra::GlobalPositionerOptions& self, const py::dict& d) {
            self.loss_scale_prior.name =
                d.contains("name") ? py::cast<std::string>(d["name"]) : "huber";
            self.loss_scale_prior.scale =
                d.contains("scale") ? py::cast<double>(d["scale"]) : 1.0;
            self.loss_scale_prior.weight =
                d.contains("weight") ? py::cast<double>(d["weight"]) : 1.0;
          },
          "Loss function config for depth map scale prior regularization "
          "(dict with 'name', 'scale', 'weight')")
      .def_property(
          "loss_normal_geometry_inlier",
          [](const glomap_ra::GlobalPositionerOptions& self) -> py::dict {
            py::dict d;
            d["name"] = self.loss_normal_geometry_inlier.name;
            d["scale"] = self.loss_normal_geometry_inlier.scale;
            d["weight"] = self.loss_normal_geometry_inlier.weight;
            return d;
          },
          [](glomap_ra::GlobalPositionerOptions& self, const py::dict& d) {
            self.loss_normal_geometry_inlier.name =
                d.contains("name") ? py::cast<std::string>(d["name"])
                                   : "trivial";
            self.loss_normal_geometry_inlier.scale =
                d.contains("scale") ? py::cast<double>(d["scale"]) : 1.0;
            self.loss_normal_geometry_inlier.weight =
                d.contains("weight") ? py::cast<double>(d["weight"]) : 1.0;
          },
          "Loss function config for inlier non-LC geometry residuals (dict "
          "with 'name', 'scale', 'weight'). Default: trivial.")
      .def_property(
          "loss_normal_depth_inlier",
          [](const glomap_ra::GlobalPositionerOptions& self) -> py::dict {
            py::dict d;
            d["name"] = self.loss_normal_depth_inlier.name;
            d["scale"] = self.loss_normal_depth_inlier.scale;
            d["weight"] = self.loss_normal_depth_inlier.weight;
            return d;
          },
          [](glomap_ra::GlobalPositionerOptions& self, const py::dict& d) {
            self.loss_normal_depth_inlier.name =
                d.contains("name") ? py::cast<std::string>(d["name"])
                                   : "trivial";
            self.loss_normal_depth_inlier.scale =
                d.contains("scale") ? py::cast<double>(d["scale"]) : 1.0;
            self.loss_normal_depth_inlier.weight =
                d.contains("weight") ? py::cast<double>(d["weight"]) : 1.0;
          },
          "Loss function config for inlier non-LC depth residuals (dict with "
          "'name', 'scale', 'weight'). Default: trivial.")
      .def_property(
          "loss_normal_depth_outlier",
          [](const glomap_ra::GlobalPositionerOptions& self) -> py::dict {
            py::dict d;
            d["name"] = self.loss_normal_depth_outlier.name;
            d["scale"] = self.loss_normal_depth_outlier.scale;
            d["weight"] = self.loss_normal_depth_outlier.weight;
            return d;
          },
          [](glomap_ra::GlobalPositionerOptions& self, const py::dict& d) {
            self.loss_normal_depth_outlier.name =
                d.contains("name") ? py::cast<std::string>(d["name"]) : "huber";
            self.loss_normal_depth_outlier.scale =
                d.contains("scale") ? py::cast<double>(d["scale"]) : 1.0;
            self.loss_normal_depth_outlier.weight =
                d.contains("weight") ? py::cast<double>(d["weight"]) : 1.0;
          },
          "Loss function config for MDRP depth outlier observations (dict "
          "with 'name', 'scale', 'weight'). Default: huber scale=1.")
      .def_property(
          "loss_normal_geometry_trackstart",
          [](const glomap_ra::GlobalPositionerOptions& self) -> py::dict {
            py::dict d;
            d["name"] = self.loss_normal_geometry_trackstart.name;
            d["scale"] = self.loss_normal_geometry_trackstart.scale;
            d["weight"] = self.loss_normal_geometry_trackstart.weight;
            return d;
          },
          [](glomap_ra::GlobalPositionerOptions& self, const py::dict& d) {
            self.loss_normal_geometry_trackstart.name =
                d.contains("name") ? py::cast<std::string>(d["name"])
                                   : "trivial";
            self.loss_normal_geometry_trackstart.scale =
                d.contains("scale") ? py::cast<double>(d["scale"]) : 1.0;
            self.loss_normal_geometry_trackstart.weight =
                d.contains("weight") ? py::cast<double>(d["weight"]) : 1.0;
          },
          "Loss function config for track anchor geometry observations (dict "
          "with 'name', 'scale', 'weight'). Default: trivial (L2) to trust "
          "anchors fully.")
      .def_property(
          "loss_normal_depth_trackstart",
          [](const glomap_ra::GlobalPositionerOptions& self) -> py::dict {
            py::dict d;
            d["name"] = self.loss_normal_depth_trackstart.name;
            d["scale"] = self.loss_normal_depth_trackstart.scale;
            d["weight"] = self.loss_normal_depth_trackstart.weight;
            return d;
          },
          [](glomap_ra::GlobalPositionerOptions& self, const py::dict& d) {
            self.loss_normal_depth_trackstart.name =
                d.contains("name") ? py::cast<std::string>(d["name"]) : "huber";
            self.loss_normal_depth_trackstart.scale =
                d.contains("scale") ? py::cast<double>(d["scale"]) : 1.0;
            self.loss_normal_depth_trackstart.weight =
                d.contains("weight") ? py::cast<double>(d["weight"]) : 1.0;
          },
          "Loss function config for track anchor depth observations (dict "
          "with 'name', 'scale', 'weight'). Default: huber to be robust for "
          "depth while trusting geometry.")
      .def_readwrite(
          "use_relative_pose_constraints",
          &glomap_ra::GlobalPositionerOptions::use_relative_pose_constraints,
          "If true, add relative pose constraints for consecutive pairs "
          "using MDRP-derived relative translations with covariance "
          "weighting. Default: false.")
      .def_readwrite(
          "relative_pose_default_stddev",
          &glomap_ra::GlobalPositionerOptions::relative_pose_default_stddev,
          "Standard deviation for relative pose constraints if cov_t is "
          "zero. Smaller values = stronger regularization. Default: 0.1")
      .def_property(
          "loss_relative_pose",
          [](const glomap_ra::GlobalPositionerOptions& self) -> py::dict {
            py::dict d;
            d["name"] = self.loss_relative_pose.name;
            d["scale"] = self.loss_relative_pose.scale;
            d["weight"] = self.loss_relative_pose.weight;
            return d;
          },
          [](glomap_ra::GlobalPositionerOptions& self, const py::dict& d) {
            self.loss_relative_pose.name =
                d.contains("name") ? py::cast<std::string>(d["name"]) : "huber";
            self.loss_relative_pose.scale =
                d.contains("scale") ? py::cast<double>(d["scale"]) : 0.1;
            self.loss_relative_pose.weight =
                d.contains("weight") ? py::cast<double>(d["weight"]) : 1.0;
          },
          "Loss function config for relative pose constraints (dict with "
          "'name', 'scale', 'weight'). Default: huber scale=0.1.")
      .def_readwrite(
          "debug_only_relative_pose",
          &glomap_ra::GlobalPositionerOptions::debug_only_relative_pose,
          "Debug mode: if true, skip all point-to-camera constraints and "
          "only use relative pose constraints. Default: false.");

  MakeDataclass(PyGlobalPositionerOptions);
}

void BindGlobalPositioningGlomap(py::module& m) {
  py::module m_glomap_ra = GetOrCreateGlomapRaModule(m);
  m_glomap_ra.def("run_global_positioning",
        &RunGlobalPositioning,
        "view_graph"_a,
        "cameras"_a,
        "images"_a,
        "tracks"_a,
        py::arg_v(
            "options", glomap_ra::GlobalPositionerOptions(), "GlobalPositionerOptions()"),
        py::arg_v("image_ids_to_optimize", std::vector<image_t>(), "[]"),
        py::arg_v("initial_dmap_scales", std::map<image_t, double>(), "{}"),
        py::arg_v(
            "consecutive_pair_ids", std::vector<image_pair_t>(), "[]"),
        "Glomap-style global positioning on a CorrespondenceGraph + cameras/"
        "images/tracks dicts. Returns a dict with keys 'images', 'tracks', "
        "'scales', 'dmap_scales', 'dmap_scale_map', 'dmap_scale_map_nested'.");
}
