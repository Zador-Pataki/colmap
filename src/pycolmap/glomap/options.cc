#include "colmap/estimators/glomap/bundle_adjustment_options.h"
#include "colmap/estimators/glomap/global_mapper_options.h"
#include "colmap/estimators/glomap/global_positioner_options.h"
#include "colmap/estimators/glomap/inlier_threshold_options.h"
#include "colmap/estimators/glomap/optimization_base_options.h"
#include "colmap/estimators/glomap/rotation_estimator_options.h"
#include "colmap/estimators/glomap/track_establishment_options.h"
#include "colmap/estimators/glomap/triangulator_options.h"
#include "colmap/estimators/glomap/view_graph_calibrator_options.h"

#include "pycolmap/helpers.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace colmap::glomap;

void BindGlomapOptions(py::module& m) {
  // ----- OptimizationBaseOptions -----
  py::classh<OptimizationBaseOptions> PyBase(m, "OptimizationBaseOptions");
  PyBase.def(py::init<>())
      .def_readwrite("thres_loss_function",
                     &OptimizationBaseOptions::thres_loss_function);
  // Skip binding `solver_options` (ceres::Solver::Options) — that type is
  // registered `py::module_local()` in ceres_bindings.cc so cross-TU
  // `def_readwrite` triggers an AttributeError at module init. Callers that
  // need to tweak ceres solver options can construct their own
  // `pycolmap.SolverOptions()` and assign later via a separate setter if
  // needed. TODO(§09/§10): expose read-only getter if any reader requires it.
  MakeDataclass(PyBase);

  // ----- GlobalPositionerOptions (nested LossFunctionConfig + 2 enums) -----
  py::classh<GlobalPositionerOptions> PyGP(m, "GlobalPositionerOptions");

  py::classh<GlobalPositionerOptions::LossFunctionConfig> PyLoss(
      PyGP, "LossFunctionConfig");
  PyLoss.def(py::init<>())
      .def(py::init<const std::string&, double, double>(), py::arg("name"),
           py::arg("scale"), py::arg("weight"))
      .def_readwrite("name", &GlobalPositionerOptions::LossFunctionConfig::name)
      .def_readwrite("scale",
                     &GlobalPositionerOptions::LossFunctionConfig::scale)
      .def_readwrite("weight",
                     &GlobalPositionerOptions::LossFunctionConfig::weight);
  MakeDataclass(PyLoss);

  py::enum_<GlobalPositionerOptions::ConstraintType>(PyGP, "ConstraintType")
      .value("ONLY_POINTS", GlobalPositionerOptions::ConstraintType::ONLY_POINTS)
      ;

  py::enum_<GlobalPositionerOptions::PointConstraintType>(PyGP,
                                                           "PointConstraintType")
      .value("GEOMETRY_ONLY",
             GlobalPositionerOptions::PointConstraintType::GEOMETRY_ONLY)
      .value("SPLIT_METRIC_DEPTH",
             GlobalPositionerOptions::PointConstraintType::SPLIT_METRIC_DEPTH)
      ;

  PyGP.def(py::init<>())
      .def_readwrite("solver_base", &GlobalPositionerOptions::solver_base)
      .def_readwrite("generate_random_positions",
                     &GlobalPositionerOptions::generate_random_positions)
      .def_readwrite("generate_random_points",
                     &GlobalPositionerOptions::generate_random_points)
      .def_readwrite("generate_scales", &GlobalPositionerOptions::generate_scales)
      .def_readwrite("random_init_scale",
                     &GlobalPositionerOptions::random_init_scale)
      .def_readwrite("use_init", &GlobalPositionerOptions::use_init)
      .def_readwrite("optimize_positions",
                     &GlobalPositionerOptions::optimize_positions)
      .def_readwrite("optimize_points",
                     &GlobalPositionerOptions::optimize_points)
      .def_readwrite("optimize_scales",
                     &GlobalPositionerOptions::optimize_scales)
      .def_readwrite("optimize_depth_map_scales",
                     &GlobalPositionerOptions::optimize_depth_map_scales)
      .def_readwrite("optimize_rotations",
                     &GlobalPositionerOptions::optimize_rotations)
      .def_readwrite("regularize_rotations",
                     &GlobalPositionerOptions::regularize_rotations)
      .def_readwrite("rotation_prior_sigma",
                     &GlobalPositionerOptions::rotation_prior_sigma)
      .def_readwrite("loss_rotation_prior",
                     &GlobalPositionerOptions::loss_rotation_prior)
      .def_readwrite("min_num_view_per_track",
                     &GlobalPositionerOptions::min_num_view_per_track)
      .def_readwrite("seed", &GlobalPositionerOptions::seed)
      .def_readwrite("constraint_type",
                     &GlobalPositionerOptions::constraint_type)
      .def_readwrite("point_constraint_type",
                     &GlobalPositionerOptions::point_constraint_type)
      .def_readwrite("scale_regularization_sigma",
                     &GlobalPositionerOptions::scale_regularization_sigma)
      .def_readwrite("scale_prior_stddev",
                     &GlobalPositionerOptions::scale_prior_stddev)
      .def_readwrite("regularize_depth_map_scales",
                     &GlobalPositionerOptions::regularize_depth_map_scales)
      .def_readwrite("use_log_scale_for_depth_map_scales",
                     &GlobalPositionerOptions::use_log_scale_for_depth_map_scales)
      .def_readwrite("use_log_residual_for_depth",
                     &GlobalPositionerOptions::use_log_residual_for_depth)
      .def_readwrite("zero_residual_behind_camera",
                     &GlobalPositionerOptions::zero_residual_behind_camera)
      .def_readwrite("smooth_log_linear_transition",
                     &GlobalPositionerOptions::smooth_log_linear_transition)
      .def_readwrite("log_linear_threshold",
                     &GlobalPositionerOptions::log_linear_threshold)
      .def_readwrite("filter_depth_outliers",
                     &GlobalPositionerOptions::filter_depth_outliers)
      .def_readwrite("consecutive_trivial",
                     &GlobalPositionerOptions::consecutive_trivial)
      .def_readwrite("use_relative_pose_constraints",
                     &GlobalPositionerOptions::use_relative_pose_constraints)
      .def_readwrite("relative_pose_default_stddev",
                     &GlobalPositionerOptions::relative_pose_default_stddev)
      .def_readwrite("debug_only_relative_pose",
                     &GlobalPositionerOptions::debug_only_relative_pose)
      .def_readwrite("thres_loss_function",
                     &GlobalPositionerOptions::thres_loss_function)
      .def_readwrite("loss_normal_geometry",
                     &GlobalPositionerOptions::loss_normal_geometry)
      .def_readwrite("loss_normal_depth",
                     &GlobalPositionerOptions::loss_normal_depth)
      .def_readwrite("loss_lc_geometry",
                     &GlobalPositionerOptions::loss_lc_geometry)
      .def_readwrite("loss_lc_depth", &GlobalPositionerOptions::loss_lc_depth)
      .def_readwrite("loss_scale_prior",
                     &GlobalPositionerOptions::loss_scale_prior)
      .def_readwrite("loss_normal_geometry_inlier",
                     &GlobalPositionerOptions::loss_normal_geometry_inlier)
      .def_readwrite("loss_normal_depth_inlier",
                     &GlobalPositionerOptions::loss_normal_depth_inlier)
      .def_readwrite("loss_normal_depth_outlier",
                     &GlobalPositionerOptions::loss_normal_depth_outlier)
      .def_readwrite("loss_normal_geometry_trackstart",
                     &GlobalPositionerOptions::loss_normal_geometry_trackstart)
      .def_readwrite("loss_normal_depth_trackstart",
                     &GlobalPositionerOptions::loss_normal_depth_trackstart)
      .def_readwrite("loss_relative_pose",
                     &GlobalPositionerOptions::loss_relative_pose);
  MakeDataclass(PyGP);

  // ----- RotationEstimatorOptions (nested SolverType + WeightType enums) -----
  py::classh<RotationEstimatorOptions> PyRA(m, "RotationEstimatorOptions");

  py::enum_<RotationEstimatorOptions::SolverType>(PyRA, "SolverType")
      .value("L1_IRLS", RotationEstimatorOptions::SolverType::L1_IRLS)
      .value("CERES_GRAVITY",
             RotationEstimatorOptions::SolverType::CERES_GRAVITY)
      ;

  py::enum_<RotationEstimatorOptions::WeightType>(PyRA, "WeightType")
      .value("GEMAN_MCCLURE",
             RotationEstimatorOptions::WeightType::GEMAN_MCCLURE)
      .value("HALF_NORM", RotationEstimatorOptions::WeightType::HALF_NORM)
      ;

  PyRA.def(py::init<>())
      .def_readwrite("solver_base", &RotationEstimatorOptions::solver_base)
      .def_readwrite("max_num_l1_iterations",
                     &RotationEstimatorOptions::max_num_l1_iterations)
      .def_readwrite("l1_step_convergence_threshold",
                     &RotationEstimatorOptions::l1_step_convergence_threshold)
      .def_readwrite("max_num_irls_iterations",
                     &RotationEstimatorOptions::max_num_irls_iterations)
      .def_readwrite("irls_step_convergence_threshold",
                     &RotationEstimatorOptions::irls_step_convergence_threshold)
      .def_readwrite("irls_loss_parameter_sigma",
                     &RotationEstimatorOptions::irls_loss_parameter_sigma)
      .def_readwrite("axis", &RotationEstimatorOptions::axis)
      .def_readwrite("weight_type", &RotationEstimatorOptions::weight_type)
      .def_readwrite("skip_initialization",
                     &RotationEstimatorOptions::skip_initialization)
      .def_readwrite("use_precomputed_weights",
                     &RotationEstimatorOptions::use_precomputed_weights)
      .def_readwrite("use_gravity", &RotationEstimatorOptions::use_gravity)
      .def_readwrite("fix_non_lc_weights",
                     &RotationEstimatorOptions::fix_non_lc_weights)
      .def_readwrite("fixed_non_lc_weight",
                     &RotationEstimatorOptions::fixed_non_lc_weight)
      .def_readwrite("skip_risky_LC_pairs",
                     &RotationEstimatorOptions::skip_risky_LC_pairs)
      .def_readwrite("solver_type", &RotationEstimatorOptions::solver_type)
      .def_readwrite("use_video_constraints",
                     &RotationEstimatorOptions::use_video_constraints)
      .def_readwrite("video_tracking_huber_scale",
                     &RotationEstimatorOptions::video_tracking_huber_scale)
      .def_readwrite("video_lc_cauchy_scale",
                     &RotationEstimatorOptions::video_lc_cauchy_scale)
      .def_readwrite("use_gravity_prior",
                     &RotationEstimatorOptions::use_gravity_prior)
      .def_readwrite("gravity_world", &RotationEstimatorOptions::gravity_world)
      .def_readwrite("default_gravity_sigma",
                     &RotationEstimatorOptions::default_gravity_sigma)
      .def_readwrite("gravity_loss_scale",
                     &RotationEstimatorOptions::gravity_loss_scale)
      .def_readwrite("gravity_loss_type",
                     &RotationEstimatorOptions::gravity_loss_type)
      .def_readwrite("video_adjacent_frames",
                     &RotationEstimatorOptions::video_adjacent_frames)
      .def_readwrite("video_prior_sigma",
                     &RotationEstimatorOptions::video_prior_sigma)
      .def_readwrite("gravity_prior_weight",
                     &RotationEstimatorOptions::gravity_prior_weight)
      .def_readwrite("loop_closure_weight",
                     &RotationEstimatorOptions::loop_closure_weight)
      .def_readwrite("max_num_ceres_iterations",
                     &RotationEstimatorOptions::max_num_ceres_iterations)
      .def_readwrite("random_seed", &RotationEstimatorOptions::random_seed)
      .def_readwrite("num_threads", &RotationEstimatorOptions::num_threads)
      .def_readwrite("regularize_to_initial",
                     &RotationEstimatorOptions::regularize_to_initial)
      .def_readwrite("regularize_sigma",
                     &RotationEstimatorOptions::regularize_sigma)
      .def_readwrite("update_rel_rotations",
                     &RotationEstimatorOptions::update_rel_rotations)
      .def_readwrite("min_weight", &RotationEstimatorOptions::min_weight);
  MakeDataclass(PyRA);

  // ----- BundleAdjustmentOptions -----
  py::classh<BundleAdjustmentOptions> PyBA(m, "BundleAdjustmentOptions");
  PyBA.def(py::init<>())
      .def_readwrite("solver_base", &BundleAdjustmentOptions::solver_base)
      .def_readwrite("optimize_rotations",
                     &BundleAdjustmentOptions::optimize_rotations)
      .def_readwrite("optimize_translation",
                     &BundleAdjustmentOptions::optimize_translation)
      .def_readwrite("optimize_intrinsics",
                     &BundleAdjustmentOptions::optimize_intrinsics)
      .def_readwrite("optimize_points",
                     &BundleAdjustmentOptions::optimize_points)
      .def_readwrite("min_num_view_per_track",
                     &BundleAdjustmentOptions::min_num_view_per_track)
      .def_readwrite("thres_loss_function",
                     &BundleAdjustmentOptions::thres_loss_function);
  MakeDataclass(PyBA);

  // ----- ViewGraphCalibratorOptions -----
  py::classh<ViewGraphCalibratorOptions> PyVGC(m, "ViewGraphCalibratorOptions");
  PyVGC.def(py::init<>())
      .def_readwrite("solver_base", &ViewGraphCalibratorOptions::solver_base)
      .def_readwrite("thres_lower_ratio",
                     &ViewGraphCalibratorOptions::thres_lower_ratio)
      .def_readwrite("thres_higher_ratio",
                     &ViewGraphCalibratorOptions::thres_higher_ratio)
      .def_readwrite("thres_two_view_error",
                     &ViewGraphCalibratorOptions::thres_two_view_error);
  MakeDataclass(PyVGC);

  // ----- TriangulatorOptions -----
  py::classh<TriangulatorOptions> PyTri(m, "TriangulatorOptions");
  PyTri.def(py::init<>())
      .def_readwrite("tri_complete_max_reproj_error",
                     &TriangulatorOptions::tri_complete_max_reproj_error)
      .def_readwrite("tri_merge_max_reproj_error",
                     &TriangulatorOptions::tri_merge_max_reproj_error)
      .def_readwrite("tri_min_angle", &TriangulatorOptions::tri_min_angle)
      .def_readwrite("min_num_matches", &TriangulatorOptions::min_num_matches);
  MakeDataclass(PyTri);

  // ----- TrackEstablishmentOptions -----
  py::classh<TrackEstablishmentOptions> PyTE(m, "TrackEstablishmentOptions");
  PyTE.def(py::init<>())
      .def_readwrite("thres_inconsistency",
                     &TrackEstablishmentOptions::thres_inconsistency)
      .def_readwrite("min_num_tracks_per_view",
                     &TrackEstablishmentOptions::min_num_tracks_per_view)
      .def_readwrite("min_num_view_per_track",
                     &TrackEstablishmentOptions::min_num_view_per_track)
      .def_readwrite("max_num_view_per_track",
                     &TrackEstablishmentOptions::max_num_view_per_track)
      .def_readwrite("max_num_tracks",
                     &TrackEstablishmentOptions::max_num_tracks);
  MakeDataclass(PyTE);

  // ----- InlierThresholdOptions -----
  py::classh<InlierThresholdOptions> PyITO(m, "InlierThresholdOptions");
  PyITO.def(py::init<>())
      .def_readwrite("max_angle_error", &InlierThresholdOptions::max_angle_error)
      .def_readwrite("max_reprojection_error",
                     &InlierThresholdOptions::max_reprojection_error)
      .def_readwrite("min_triangulation_angle",
                     &InlierThresholdOptions::min_triangulation_angle)
      .def_readwrite("max_epipolar_error_E",
                     &InlierThresholdOptions::max_epipolar_error_E)
      .def_readwrite("max_epipolar_error_F",
                     &InlierThresholdOptions::max_epipolar_error_F)
      .def_readwrite("max_epipolar_error_H",
                     &InlierThresholdOptions::max_epipolar_error_H)
      .def_readwrite("min_angle_from_epipole",
                     &InlierThresholdOptions::min_angle_from_epipole)
      .def_readwrite("min_inlier_num", &InlierThresholdOptions::min_inlier_num)
      .def_readwrite("min_inlier_ratio",
                     &InlierThresholdOptions::min_inlier_ratio)
      .def_readwrite("max_rotation_error",
                     &InlierThresholdOptions::max_rotation_error);
  MakeDataclass(PyITO);

  // ----- GlobalMapperOptions (composes all the sub-options structs) -----
  py::classh<GlobalMapperOptions> PyGM(m, "GlobalMapperOptions");
  PyGM.def(py::init<>())
      .def_readwrite("opt_vgcalib", &GlobalMapperOptions::opt_vgcalib)
      .def_readwrite("opt_ra", &GlobalMapperOptions::opt_ra)
      .def_readwrite("opt_track", &GlobalMapperOptions::opt_track)
      .def_readwrite("opt_gp", &GlobalMapperOptions::opt_gp)
      .def_readwrite("opt_ba", &GlobalMapperOptions::opt_ba)
      .def_readwrite("opt_triangulator", &GlobalMapperOptions::opt_triangulator)
      .def_readwrite("inlier_thresholds",
                     &GlobalMapperOptions::inlier_thresholds)
      .def_readwrite("num_iteration_bundle_adjustment",
                     &GlobalMapperOptions::num_iteration_bundle_adjustment)
      .def_readwrite("num_iteration_retriangulation",
                     &GlobalMapperOptions::num_iteration_retriangulation)
      .def_readwrite("skip_preprocessing",
                     &GlobalMapperOptions::skip_preprocessing)
      .def_readwrite("skip_view_graph_calibration",
                     &GlobalMapperOptions::skip_view_graph_calibration)
      .def_readwrite("skip_relative_pose_estimation",
                     &GlobalMapperOptions::skip_relative_pose_estimation)
      .def_readwrite("skip_rotation_averaging",
                     &GlobalMapperOptions::skip_rotation_averaging)
      .def_readwrite("skip_track_establishment",
                     &GlobalMapperOptions::skip_track_establishment)
      .def_readwrite("skip_global_positioning",
                     &GlobalMapperOptions::skip_global_positioning)
      .def_readwrite("skip_bundle_adjustment",
                     &GlobalMapperOptions::skip_bundle_adjustment)
      .def_readwrite("skip_retriangulation",
                     &GlobalMapperOptions::skip_retriangulation)
      .def_readwrite("skip_pruning", &GlobalMapperOptions::skip_pruning);
  MakeDataclass(PyGM);
}
