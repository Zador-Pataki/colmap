#include "colmap/scene/pose_graph.h"
#include "colmap/estimators/pose_graph_processors.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"
#include "pycolmap/scene/types.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

void BindPoseGraph(py::module& m) {
  py::class_<PoseGraph::MDRPResult>(m, "MDRPResult")
      .def(py::init<>())
      .def_readwrite("is_valid", &PoseGraph::MDRPResult::is_valid)
      .def_readwrite("cam2_from_cam1", &PoseGraph::MDRPResult::cam2_from_cam1)
      .def_readwrite("weight", &PoseGraph::MDRPResult::weight)
      .def_readwrite("rel_depth_scale",
                     &PoseGraph::MDRPResult::rel_depth_scale)
      .def_readwrite("inliers", &PoseGraph::MDRPResult::inliers)
      .def_readwrite("cov_t", &PoseGraph::MDRPResult::cov_t);

  py::class_<PoseGraph::ValidPairData>(m, "ValidPairData")
      .def(py::init<>())
      .def_readwrite("pair_ids", &PoseGraph::ValidPairData::pair_ids)
      .def_readwrite("image_ids1", &PoseGraph::ValidPairData::image_ids1)
      .def_readwrite("image_ids2", &PoseGraph::ValidPairData::image_ids2)
      .def_readwrite("inlier_counts",
                     &PoseGraph::ValidPairData::inlier_counts)
      .def_readwrite("are_lc", &PoseGraph::ValidPairData::are_lc);

  auto PyPoseGraphEdge =
      py::classh<PoseGraph::Edge>(m, "PoseGraphEdge")
          .def(py::init<>())
          .def(py::init<const Rigid3d&>(), "cam2_from_cam1"_a)
          .def_readwrite("cam2_from_cam1",
                         &PoseGraph::Edge::cam2_from_cam1,
                         "Relative pose from image 1 to image 2.")
          .def_readwrite("num_matches",
                         &PoseGraph::Edge::num_matches,
                         "Number of two-view matches used to compute the "
                         "relative pose.")
          .def_readwrite("total_matches",
                         &PoseGraph::Edge::total_matches,
                         "Total number of two-view matches (before inlier "
                         "filtering).")
          .def_readwrite("valid",
                         &PoseGraph::Edge::valid,
                         "Whether this edge is valid for reconstruction.")
          .def("invert",
               &PoseGraph::Edge::Invert,
               "Invert the geometry to match swapped image order.")
          .def_readwrite("rel_depth_scale",
                         &PoseGraph::Edge::rel_depth_scale,
                         "Relative metric depth scale (MDRP result).")
          .def_readwrite("cov_t",
                         &PoseGraph::Edge::cov_t,
                         "3x3 covariance of relative translation.")
          .def_readwrite("are_lc",
                         &PoseGraph::Edge::are_lc,
                         "Per-inlier loop-closure flag.")
          .def_readwrite("is_LC",
                         &PoseGraph::Edge::is_LC,
                         "Whether this edge was added as a loop closure.")
          .def_readwrite("weight",
                         &PoseGraph::Edge::weight,
                         "Edge weight for rotation averaging.")
          .def_readwrite(
              "F",
              &PoseGraph::Edge::F,
              "Fundamental matrix (optional; set by DecomposeRelPose).")
          .def_readwrite(
              "H",
              &PoseGraph::Edge::H,
              "Homography matrix (optional; set by DecomposeRelPose).")
          .def_readwrite("config",
                         &PoseGraph::Edge::config,
                         "TwoViewGeometry::ConfigurationType value.")
          .def_readwrite(
              "matches",
              &PoseGraph::Edge::matches,
              "Feature correspondences: matches[i] = (feat_idx_1, feat_idx_2).")
          .def_readwrite("inliers",
                         &PoseGraph::Edge::inliers,
                         "Inlier indices into matches.")
          .def_readwrite("image_id1",
                         &PoseGraph::Edge::image_id1,
                         "ID of the first image (set by AddEdge).")
          .def_readwrite("image_id2",
                         &PoseGraph::Edge::image_id2,
                         "ID of the second image (set by AddEdge).")
          .def_property(
              "is_valid",
              [](const PoseGraph::Edge& e) { return e.valid; },
              [](PoseGraph::Edge& e, bool v) { e.valid = v; },
              "Alias for valid. Whether this edge is valid for reconstruction.");
  MakeDataclass(PyPoseGraphEdge);

  py::bind_map<PoseGraphEdgeMap>(m, "PoseGraphEdgeMap");

  py::classh<PoseGraph>(m, "PoseGraph")
      .def(py::init<>())
      .def_property_readonly("edges",
                             py::overload_cast<>(&PoseGraph::Edges),
                             py::return_value_policy::reference_internal,
                             "Access to all edges in the pose graph.")
      .def_property_readonly("num_edges",
                             &PoseGraph::NumEdges,
                             "Number of edges in the pose graph.")
      .def_property_readonly(
          "empty", &PoseGraph::Empty, "Whether the pose graph has no edges.")
      .def("clear", &PoseGraph::Clear, "Remove all edges.")
      .def("load",
           &PoseGraph::Load,
           "corr_graph"_a,
           "Load edges from a correspondence graph.")
      .def("add_edge",
           &PoseGraph::AddEdge,
           "image_id1"_a,
           "image_id2"_a,
           "edge"_a,
           py::return_value_policy::reference_internal,
           "Add a new edge between two images. Throws if edge already exists.")
      .def("has_edge",
           &PoseGraph::HasEdge,
           "image_id1"_a,
           "image_id2"_a,
           "Check if an edge exists between two images.")
      .def("get_edge",
           &PoseGraph::GetEdge,
           "image_id1"_a,
           "image_id2"_a,
           "Get a copy of the edge between two images. Automatically handles "
           "geometric inversion if image order was swapped.")
      .def("delete_edge",
           &PoseGraph::DeleteEdge,
           "image_id1"_a,
           "image_id2"_a,
           "Delete the edge between two images. Returns True if deleted.")
      .def("update_edge",
           &PoseGraph::UpdateEdge,
           "image_id1"_a,
           "image_id2"_a,
           "edge"_a,
           "Update an existing edge. Throws if edge does not exist.")
      .def("is_valid",
           &PoseGraph::IsValid,
           "pair_id"_a,
           "Check if an edge is marked as valid.")
      .def("set_valid_edge",
           &PoseGraph::SetValidEdge,
           "pair_id"_a,
           "Mark an edge as valid.")
      .def("set_invalid_edge",
           &PoseGraph::SetInvalidEdge,
           "pair_id"_a,
           "Mark an edge as invalid.")
      .def("compute_largest_connected_frame_component",
           &PoseGraph::ComputeLargestConnectedFrameComponent,
           "reconstruction"_a,
           "filter_unregistered"_a = true,
           "Compute the largest connected component of frames. If "
           "filter_unregistered is True, only considers frames with poses.")
      .def("invalidate_pairs_outside_active_image_ids",
           &PoseGraph::InvalidatePairsOutsideActiveImageIds,
           "active_image_ids"_a,
           "Mark image pairs as invalid if either image is not in the active "
           "set.")
      .def(
          "mark_connected_components",
          [](const PoseGraph& self,
             const Reconstruction& reconstruction,
             int min_num_images) {
            std::unordered_map<frame_t, int> cluster_ids;
            int num_components = self.MarkConnectedComponents(
                reconstruction, cluster_ids, min_num_images);
            return py::dict("num_components"_a = num_components,
                            "cluster_ids"_a = cluster_ids);
          },
          "reconstruction"_a,
          "min_num_images"_a = -1,
          "Mark connected clusters of images. Returns dict with num_components "
          "and cluster_ids mapping frame IDs to cluster IDs.")
      .def("assign_mdrp_results",
           &PoseGraph::AssignMdrpResults,
           "results"_a,
           "metric_scale_stddev"_a,
           "Assign MDRP results to matching edges. Returns (num_valid, "
           "num_invalid) counts.")
      .def("extract_valid_pair_data",
           &PoseGraph::ExtractValidPairData,
           "Return per-edge data arrays for all currently valid edges.")
      .def("filter_inlier_num",
           &PoseGraph::FilterInlierNum,
           "min_inliers"_a,
           "Invalidate edges with inlier count below threshold.")
      .def("filter_inlier_ratio",
           &PoseGraph::FilterInlierRatio,
           "min_ratio"_a,
           "Invalidate edges with inlier ratio below threshold.")
      .def("keep_largest_connected_components",
           &PoseGraph::KeepLargestConnectedComponents,
           "k"_a,
           "min_component_size"_a,
           "Keep the top-k connected components; invalidate edges in smaller "
           "components.")
      .def_property_readonly(
          "image_pairs",
          py::overload_cast<>(&PoseGraph::Edges),
          py::return_value_policy::reference_internal,
          "Alias for edges. Access to all edges in the pose graph.")
      .def("add_pair",
           &PoseGraph::AddEdge,
           "image_id1"_a,
           "image_id2"_a,
           "edge"_a,
           py::return_value_policy::reference_internal,
           "Alias for add_edge. Add a new edge between two images.")
      .def(
          "update_image_pairs_config",
          [](PoseGraph& self,
             const std::unordered_map<camera_t, Camera>& cameras,
             const std::unordered_map<image_t, Image>& images) {
            py::gil_scoped_release release;
            UpdateImagePairsConfig(self, cameras, images);
          },
          "cameras"_a,
          "images"_a,
          "Re-derive per-pair TwoViewGeometry config after intrinsics change.")
      .def(
          "decompose_rel_pose",
          [](PoseGraph& self,
             const std::unordered_map<camera_t, Camera>& cameras,
             const std::unordered_map<image_t, Image>& images) {
            py::gil_scoped_release release;
            DecomposeRelPose(self, cameras, images);
          },
          "cameras"_a,
          "images"_a,
          "Decompose E/F/H into cam2_from_cam1 for calibrated edges.");

  m.def(
      "undistort_images",
      [](const std::unordered_map<camera_t, Camera>& cameras,
         std::unordered_map<image_t, Image>& images,
         bool clean_points) {
        py::gil_scoped_release release;
        UndistortImages(cameras, images, clean_points);
      },
      "cameras"_a,
      "images"_a,
      "clean_points"_a = false,
      "Populate Image::features_undist for each image (CamFromImg per point).");
}
