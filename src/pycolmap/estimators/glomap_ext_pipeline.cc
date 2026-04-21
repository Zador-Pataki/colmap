#include "colmap/controllers/global_pipeline.h"
#include "colmap/util/file.h"
#include "colmap/estimators/global_positioning.h"
#include "colmap/estimators/rotation_averaging.h"
#include "colmap/estimators/iteration_callback.h"
#include "colmap/scene/database.h"
#include "colmap/scene/database_cache.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/sfm/global_mapper.h"

#include "colmap/estimators/cost_functions/bata_pairwise_direction_error.h"
#include "colmap/estimators/cost_functions/bata_pairwise_direction_error_with_rotation.h"
#include "colmap/estimators/cost_functions/direct_scale_regularization_error.h"
#include "colmap/estimators/cost_functions/fetzer_focal_length_cost.h"
#include "colmap/estimators/cost_functions/fetzer_focal_length_same_camera_cost.h"
#include "colmap/estimators/cost_functions/focal_length_random_walk_error.h"
#include "colmap/estimators/cost_functions/grav_error.h"
#include "colmap/estimators/cost_functions/log_scale_prior_error.h"
#include "colmap/estimators/cost_functions/mahalanobis_bata_directional_error.h"
#include "colmap/estimators/cost_functions/mahalanobis_bata_directional_error_with_rotation.h"
#include "colmap/estimators/cost_functions/metric_depth_error.h"
#include "colmap/estimators/cost_functions/metric_depth_error_with_rotation.h"
#include "colmap/estimators/cost_functions/point_camera_anisotropic_range_error.h"
#include "colmap/estimators/cost_functions/point_camera_direct_scaled_range_error.h"
#include "colmap/estimators/cost_functions/point_camera_range_error.h"
#include "colmap/estimators/cost_functions/point_camera_scaled_metric_range_error.h"
#include "colmap/estimators/cost_functions/point_camera_scaled_range_error.h"
#include "colmap/estimators/cost_functions/relative_translation_error.h"
#include "colmap/estimators/cost_functions/rotation_prior_error.h"
#include "colmap/estimators/cost_functions/scale_prior_error.h"
#include "colmap/estimators/cost_functions/scale_regularization_error.h"
#include "colmap/estimators/cost_functions/weighted_bata_directional_error.h"
#include "colmap/estimators/cost_functions/weighted_bata_directional_error_with_rotation.h"
#include "colmap/estimators/cost_functions/weighted_point_camera_scaled_metric_range_error.h"

#include <filesystem>

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

namespace {

py::dict IterationDataToPy(const glomap_ext::IterationData& data) {
  py::dict d;
  d["iteration"] = data.iteration;
  d["cost"] = data.cost;
  d["cost_change"] = data.cost_change;
  d["gradient_norm"] = data.gradient_norm;
  if (data.lc_edges) {
    py::list lc;
    for (const auto& e : *data.lc_edges) {
      lc.append(py::make_tuple(e.image_id1, e.image_id2, e.shared_track_count));
    }
    d["lc_edges"] = lc;
  }
  return d;
}

glomap_ext::IterationCallbackFn WrapPyCallback(py::object py_cb) {
  if (py_cb.is_none()) return nullptr;
  auto cb = std::make_shared<py::object>(std::move(py_cb));
  return [cb](const glomap_ext::IterationData& data) {
    py::gil_scoped_acquire gil;
    (*cb)(IterationDataToPy(data));
  };
}

}  // namespace

void BindGlomapExtCostFunctions(py::module& m) {
  // --- BATA variants ---
  m.def("BATAPairwiseDirectionErrorCost",
        &BATAPairwiseDirectionError::Create,
        "translation_obs"_a,
        "BATA pairwise direction error (3 residuals).");
  m.def("BATAPairwiseDirectionErrorWithRotationCost",
        &BATAPairwiseDirectionErrorWithRotation::Create,
        "bearing_cam"_a,
        "BATA pairwise direction error with optimized rotation (3 residuals).");
  m.def("WeightedBATADirectionalErrorCost",
        &WeightedBATADirectionalError::Create,
        "translation_obs"_a,
        "rotation"_a,
        "sigma_x"_a,
        "sigma_y"_a,
        "sigma_z"_a,
        "Anisotropically weighted BATA directional error (3 residuals).");
  m.def("WeightedBATADirectionalErrorWithRotationCost",
        &WeightedBATADirectionalErrorWithRotation::Create,
        "bearing_cam"_a,
        "sigma_x"_a,
        "sigma_y"_a,
        "sigma_z"_a,
        "Anisotropically weighted BATA directional error with optimized "
        "rotation (3 residuals).");
  m.def("MahalanobisBATADirectionalErrorCost",
        &MahalanobisBATADirectionalError::Create,
        "translation_obs"_a,
        "rotation"_a,
        "L00"_a,
        "L10"_a,
        "L11"_a,
        "sigma_z"_a,
        "Mahalanobis BATA directional error with Cholesky XY weighting (3 "
        "residuals).");
  m.def("MahalanobisBATADirectionalErrorWithRotationCost",
        &MahalanobisBATADirectionalErrorWithRotation::Create,
        "bearing_cam"_a,
        "L00"_a,
        "L10"_a,
        "L11"_a,
        "sigma_z"_a,
        "Mahalanobis BATA directional error with optimized rotation (3 "
        "residuals).");

  // --- Range errors ---
  m.def("PointCameraRangeErrorCost",
        &PointCameraRangeError::Create,
        "v_ik"_a,
        "depth_prior"_a,
        "feature_undist_z"_a,
        "Point-to-camera depth range error (1 residual).");
  m.def("PointCameraScaledRangeErrorCost",
        &PointCameraScaledRangeError::Create,
        "v_ik"_a,
        "depth_prior"_a,
        "feature_undist_z"_a,
        "depth_stddev"_a,
        "Scaled point-to-camera depth range error (1 residual).");
  m.def("PointCameraDirectScaledRangeErrorCost",
        &PointCameraDirectScaledRangeError::Create,
        "v_ik"_a,
        "depth_prior"_a,
        "feature_undist_z"_a,
        "Direct-scaled point-to-camera depth range error (1 residual).");
  m.def("PointCameraScaledMetricRangeErrorCost",
        &PointCameraScaledMetricRangeError::Create,
        "v_ik"_a,
        "depth_prior"_a,
        "feature_undist_z"_a,
        "Scaled metric depth range error (1 residual).");
  m.def(
      "WeightedPointCameraScaledMetricRangeErrorCost",
      &WeightedPointCameraScaledMetricRangeError::Create,
      "v_ik"_a,
      "depth_prior"_a,
      "feature_undist_z"_a,
      "rotation"_a,
      "sigma_x"_a,
      "sigma_y"_a,
      "sigma_z"_a,
      "Anisotropically weighted scaled metric depth range error (3 residuals).");
  m.def(
      "PointCameraAnisotropicRangeErrorCost",
      &PointCameraAnisotropicRangeError::Create,
      "v_ik"_a,
      "depth_prior"_a,
      "feature_undist_z"_a,
      "rotation"_a,
      "sigma_x"_a,
      "sigma_y"_a,
      "sigma_z"_a,
      "Anisotropic point-to-camera range error with constant rotation (3 "
      "residuals).");

  // --- Depth / metric ---
  m.def(
      "MetricDepthErrorCost",
      [](double depth_prior,
         double sigma_depth,
         bool use_log_scale,
         bool use_log_residual,
         bool zero_residual_behind) {
        return MetricDepthErrorWithRotation::Create(
            depth_prior, sigma_depth, use_log_scale, use_log_residual,
            zero_residual_behind);
      },
      "depth_prior"_a,
      "sigma_depth"_a,
      "use_log_scale"_a = false,
      "use_log_residual"_a = false,
      "zero_residual_behind"_a = false,
      "Metric depth prior error — rotation is a Ceres parameter block (1 residual).");
  m.def(
      "MetricDepthErrorWithRotationCost",
      [](const Eigen::Quaterniond& rotation,
         double depth_prior,
         double sigma_depth,
         bool use_log_scale,
         bool use_log_residual) {
        return MetricDepthError::Create(
            rotation, depth_prior, sigma_depth, use_log_scale, use_log_residual);
      },
      "rotation"_a,
      "depth_prior"_a,
      "sigma_depth"_a,
      "use_log_scale"_a = false,
      "use_log_residual"_a = false,
      "Metric depth prior error with fixed rotation (1 residual).");

  // --- Priors & regularization ---
  m.def("RelativeTranslationErrorCost",
        &RelativeTranslationError::Create,
        "R_w1"_a,
        "t_12_metric"_a,
        "cov_t"_a,
        "Relative translation error with Mahalanobis weighting (3 residuals).");
  m.def("RotationPriorErrorCost",
        &RotationPriorError::Create,
        "prior_rotation"_a,
        "sigma_rotation"_a,
        "Rotation prior error (3 residuals).");
  m.def("ScalePriorErrorCost",
        &ScalePriorError::Create,
        "prior"_a,
        "stddev"_a,
        "Scale prior error (1 residual).");
  m.def("LogScalePriorErrorCost",
        &LogScalePriorError::Create,
        "sigma_log"_a,
        "Log-scale prior error (1 residual).");
  m.def("ScaleRegularizationErrorCost",
        &ScaleRegularizationError::Create,
        "weight"_a,
        "Scale regularization error on log_scale (1 residual).");
  m.def("DirectScaleRegularizationErrorCost",
        &DirectScaleRegularizationError::Create,
        "sigma"_a,
        "Direct scale regularization error (1 residual).");
  m.def("GravErrorCost",
        &GravError::CreateCost,
        "grav_obs"_a,
        "Gravity direction error (3 residuals).");

  // --- Focal length ---
  m.def("FetzerFocalLengthCost",
        &FetzerFocalLengthCost::Create,
        "F"_a,
        "principal_point0"_a,
        "principal_point1"_a,
        "Fetzer focal length cost from fundamental matrix (2 residuals).");
  m.def("FetzerFocalLengthSameCameraCost",
        &FetzerFocalLengthSameCameraCost::Create,
        "F"_a,
        "principal_point"_a,
        "Fetzer focal length cost for same camera (2 residuals).");
  m.def("FocalLengthRandomWalkErrorCost",
        &FocalLengthRandomWalkError::Create,
        "variance_per_frame"_a,
        "frame_gap"_a,
        "Focal length random-walk regularization error (1 residual).");
}

void BindGlomapExtPipeline(py::module& m) {
  py::module_ m_cf = m.def_submodule("cost_functions");
  BindGlomapExtCostFunctions(m_cf);

  m.def(
      "run_rotation_averaging_extended",
      [](const RotationEstimatorOptions& options,
         PoseGraph& pose_graph,
         Reconstruction& reconstruction,
         const std::vector<PosePrior>& pose_priors) {
        py::gil_scoped_release release;
        auto [success, weights] = RunRotationAveragingWithWeights(
            options, pose_graph, reconstruction, pose_priors);
        py::gil_scoped_acquire acquire;
        return py::make_tuple(
            success,
            py::cast(&reconstruction.Images(),
                     py::return_value_policy::reference_internal),
            weights);
      },
      "options"_a,
      "pose_graph"_a,
      "reconstruction"_a,
      "pose_priors"_a = std::vector<PosePrior>{},
      "Extended rotation averaging. Returns (success, images, weights_dict) "
      "where weights_dict maps pair_id to per-edge weight.");

  m.def(
      "run_global_positioning_extended",
      [](const PoseGraph& pose_graph,
         Reconstruction& reconstruction,
         const GlobalPositionerOptions& options,
         const std::unordered_set<image_t>& image_ids_to_optimize,
         const std::map<image_t, double>& initial_dmap_scales,
         const std::vector<image_pair_t>& consecutive_pair_ids,
         py::object iteration_callback) {
        // Use upstream colmap::GlobalPositioner with extended options populated.
        GlobalPositionerOptions ext_opts = options;
        ext_opts.image_ids_to_optimize = image_ids_to_optimize;
        ext_opts.initial_dmap_scales = initial_dmap_scales;
        ext_opts.relative_pose_pair_ids = consecutive_pair_ids;
        if (!iteration_callback.is_none()) {
          ext_opts.iteration_callback =
              WrapPyCallback(std::move(iteration_callback));
        }
        GlobalPositioner gp(ext_opts);
        {
          py::gil_scoped_release release;
          gp.Solve(pose_graph, reconstruction);
        }
        py::dict out;
        out["images"] =
            py::cast(&reconstruction.Images(),
                     py::return_value_policy::reference_internal);
        out["points3D"] =
            py::cast(&reconstruction.Points3D(),
                     py::return_value_policy::reference_internal);
        out["scales"] = gp.GetScales();
        out["dmap_scales"] = gp.GetDepthMapScales();
        out["dmap_scale_map"] = gp.GetDepthMapScaleMap();
        out["dmap_scale_map_nested"] = gp.GetDepthMapScaleMapNested();
        out["geometry_only_constraints"] = gp.GetGeometryOnlyConstraints();
        out["depth_constraints"] = gp.GetDepthConstraints();
        return out;
      },
      "pose_graph"_a,
      "reconstruction"_a,
      "options"_a,
      "image_ids_to_optimize"_a = std::unordered_set<image_t>{},
      "initial_dmap_scales"_a = std::map<image_t, double>{},
      "consecutive_pair_ids"_a = std::vector<image_pair_t>{},
      "iteration_callback"_a = py::none(),
      "Extended global positioning using upstream colmap::GlobalPositioner. "
      "Returns dict with images, points3D, scales, dmap_scales, "
      "dmap_scale_map, dmap_scale_map_nested, geometry_only_constraints, "
      "depth_constraints.");

  // run_track_retriangulation
  m.def(
      "run_track_retriangulation",
      [](const std::shared_ptr<Reconstruction>& reconstruction,
         const std::filesystem::path& database_path,
         const IncrementalTriangulator::Options& options,
         const BundleAdjustmentOptions& ba_options,
         double max_normalized_reproj_error,
         double min_tri_angle_deg,
         int min_num_matches) {
        THROW_CHECK_FILE_EXISTS(database_path);
        auto database = Database::Open(database_path);
        DatabaseCache::Options cache_opts;
        cache_opts.min_num_matches = static_cast<size_t>(min_num_matches);
        auto database_cache = DatabaseCache::Create(*database, cache_opts);
        GlobalMapper mapper(database_cache);
        mapper.BeginReconstruction(reconstruction);
        py::gil_scoped_release release;
        return mapper.IterativeRetriangulateAndRefine(
            options, ba_options, max_normalized_reproj_error, min_tri_angle_deg);
      },
      "reconstruction"_a,
      "database_path"_a,
      py::arg_v("options",
                IncrementalTriangulator::Options(),
                "IncrementalTriangulatorOptions()"),
      py::arg_v("ba_options",
                BundleAdjustmentOptions(),
                "BundleAdjustmentOptions()"),
      "max_normalized_reproj_error"_a = 1e-2,
      "min_tri_angle_deg"_a = 1.0,
      "min_num_matches"_a = 15,
      "Retriangulate tracks and run iterative refinement using colmap incremental mapper.");
}
