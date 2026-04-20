#include "colmap/glomap/view_graph.h"

#include "pycolmap/glomap/types.h"
#include "pycolmap/helpers.h"

#include <memory>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <colmap/geometry/rigid3.h>

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;
using namespace pybind11::literals;

using colmap::Rigid3d;
using colmap::glomap::image_pair_t;
using colmap::glomap::image_t;
using colmap::glomap::ImagePair;
using colmap::glomap::ViewGraph;

void BindGlomapViewGraph(py::module& m) {
  py::classh<ViewGraph> PyViewGraph(m, "ViewGraph");
  PyViewGraph.def(py::init<>())

      // Read-only view of the pair map (uses reference_internal so mpsfm can
      // mutate pairs in place via ViewGraph.image_pairs[pair_id].<field>).
      .def_property_readonly(
          "image_pairs",
          [](const ViewGraph& self)
              -> const std::unordered_map<image_pair_t, ImagePair>& {
            return self.image_pairs;
          },
          py::return_value_policy::reference_internal,
          "The (read-only) map of image pairs.")
      .def_readwrite("num_images", &ViewGraph::num_images)
      .def_readwrite("num_pairs", &ViewGraph::num_pairs)

      // Real C++ methods from §04.
      .def("keep_largest_connected_components",
           &ViewGraph::KeepLargestConnectedComponents,
           "images"_a,
           "Mark only images in the largest connected component as registered. "
           "Returns the size of that component.")
      .def("mark_connected_components",
           &ViewGraph::MarkConnectedComponents,
           "images"_a,
           "min_num_img"_a = -1,
           "Assign cluster_id per image (sorted by component size).")
      .def("establish_adjacency_list",
           &ViewGraph::EstablishAdjacencyList,
           "Rebuild the adjacency list from current image_pairs.")
      .def("get_adjacency_list",
           &ViewGraph::GetAdjacencyList,
           py::return_value_policy::reference_internal)
      .def("remove_invalid_pair",
           &ViewGraph::RemoveInvalidPair,
           "pair_id"_a,
           "Mark a specific pair invalid.")

      // -------------------------------------------------------------------
      // Batch methods ported from fork pyglomap/scene/view_graph.cc.
      // -------------------------------------------------------------------

      .def(
          "add_pair",
          [](ViewGraph& self, const ImagePair& pair) {
            self.image_pairs.emplace(pair.pair_id, pair);
          },
          "pair"_a,
          "Add an ImagePair to the view graph.")

      // Batch assign MDRP results. Uses colmap4's Rigid3d type caster — fork's
      // manual quat unpack (quat[0/1/2/3]) is replaced with a direct
      // `cast<Rigid3d>()`.
      .def(
          "assign_mdrp_results",
          [](ViewGraph& self,
             const py::dict& results,
             double metric_scale_std) {
            const double metric_scale_std_sq =
                metric_scale_std * metric_scale_std;
            int num_valid = 0;
            int num_invalid = 0;

            for (const auto& item : results) {
              const image_pair_t pair_id = item.first.cast<image_pair_t>();
              const py::dict result = item.second.cast<py::dict>();

              auto it = self.image_pairs.find(pair_id);
              if (it == self.image_pairs.end()) continue;

              ImagePair& image_pair = it->second;

              const bool is_valid = result["is_valid"].cast<bool>();
              if (!is_valid) {
                image_pair.is_valid = false;
                ++num_invalid;
                continue;
              }

              image_pair.is_valid = true;

              // Rigid3d caster handles pycolmap.Rigid3d -> colmap::Rigid3d.
              image_pair.cam2_from_cam1 =
                  result["cam2_from_cam1"].cast<Rigid3d>();
              image_pair.weight = result["weight"].cast<double>();
              image_pair.rel_depth_scale =
                  result["rel_depth_scale"].cast<double>();
              image_pair.inliers = result["inliers"].cast<std::vector<int>>();

              // Compute cov_t with scale adjustment
              Eigen::Matrix3d cov_t =
                  result["cov_t"].cast<Eigen::Matrix3d>();
              const Eigen::Vector3d t =
                  image_pair.cam2_from_cam1.translation();
              cov_t += t * t.transpose() * metric_scale_std_sq;
              image_pair.cov_t = cov_t;

              ++num_valid;
            }

            return std::make_pair(num_valid, num_invalid);
          },
          py::arg("results"),
          py::arg("metric_scale_std") = 0.3,
          "Batch assign MDRP results to image pairs. Returns (num_valid, "
          "num_invalid).")

      // Batch extract pair data for all valid pairs with inliers.
      .def(
          "extract_valid_pair_data",
          [](const ViewGraph& self) {
            py::dict result;
            for (const auto& [pair_id, ip] : self.image_pairs) {
              if (!ip.is_valid) continue;
              if (ip.inliers.empty()) continue;

              const size_t n_are_lc = ip.are_lc.size();
              py::array_t<bool> are_lc_arr(n_are_lc);
              auto are_lc_mut = are_lc_arr.mutable_unchecked<1>();
              for (size_t i = 0; i < n_are_lc; ++i) {
                are_lc_mut(i) = ip.are_lc[i];
              }

              py::array_t<int> inliers_arr(ip.inliers.size());
              auto inliers_mut = inliers_arr.mutable_unchecked<1>();
              for (size_t i = 0; i < ip.inliers.size(); ++i) {
                inliers_mut(i) = ip.inliers[i];
              }

              Eigen::MatrixXi matches_copy = ip.matches;

              py::dict pair_dict;
              pair_dict["id1"] = ip.image_id1;
              pair_dict["id2"] = ip.image_id2;
              pair_dict["matches"] = std::move(matches_copy);
              pair_dict["inliers"] = std::move(inliers_arr);
              pair_dict["are_lc"] = std::move(are_lc_arr);
              result[py::int_(pair_id)] = std::move(pair_dict);
            }
            return result;
          },
          "Batch extract data for all valid pairs with inliers. Returns "
          "dict[pair_id -> {id1, id2, matches, inliers, are_lc}].")

      // Batch extract pair pose data for given pair IDs. Uses .rotation() /
      // .translation() method form (Rigid3d sweep).
      .def(
          "extract_pair_poses",
          [](const ViewGraph& self,
             const std::vector<image_pair_t>& pair_ids) {
            py::dict result;
            for (const auto pair_id : pair_ids) {
              auto it = self.image_pairs.find(pair_id);
              if (it == self.image_pairs.end()) continue;
              const ImagePair& ip = it->second;

              Eigen::Matrix3d R =
                  ip.cam2_from_cam1.rotation().toRotationMatrix();
              Eigen::Vector3d t = ip.cam2_from_cam1.translation();
              Eigen::MatrixXi matches_copy = ip.matches;

              py::dict pair_dict;
              pair_dict["id1"] = ip.image_id1;
              pair_dict["id2"] = ip.image_id2;
              pair_dict["R"] = std::move(R);
              pair_dict["t"] = std::move(t);
              pair_dict["matches"] = std::move(matches_copy);
              pair_dict["rel_depth_scale"] = ip.rel_depth_scale;

              result[py::int_(pair_id)] = std::move(pair_dict);
            }
            return result;
          },
          py::arg("pair_ids"),
          "Batch extract pair pose data (id1, id2, R, t, matches, "
          "rel_depth_scale) for given pair IDs.")

      // Batch mark match indices as LC across multiple pairs.
      .def(
          "batch_mark_matches_as_lc",
          [](ViewGraph& self, const py::dict& marks) {
            int total_marked = 0;
            for (const auto& item : marks) {
              const image_pair_t pair_id = item.first.cast<image_pair_t>();
              auto it = self.image_pairs.find(pair_id);
              if (it == self.image_pairs.end()) continue;

              ImagePair& ip = it->second;
              const auto indices = item.second.cast<std::vector<int>>();
              for (const int idx : indices) {
                if (idx >= 0 &&
                    idx < static_cast<int>(ip.are_lc.size())) {
                  if (!ip.are_lc[idx]) {
                    ip.are_lc[idx] = true;
                    ++total_marked;
                  }
                }
              }
            }
            return total_marked;
          },
          py::arg("marks"),
          "Batch mark match indices as LC. Takes dict[pair_id -> list[int]]. "
          "Returns total newly marked.")

      // Lightweight extract of (image_id1, image_id2) for given pair IDs.
      .def(
          "extract_pair_image_ids",
          [](const ViewGraph& self,
             const std::vector<image_pair_t>& pair_ids) {
            py::dict result;
            for (const auto pair_id : pair_ids) {
              auto it = self.image_pairs.find(pair_id);
              if (it == self.image_pairs.end()) continue;
              const ImagePair& ip = it->second;
              result[py::int_(pair_id)] =
                  py::make_tuple(ip.image_id1, ip.image_id2);
            }
            return result;
          },
          py::arg("pair_ids"),
          "Batch extract (image_id1, image_id2) tuples for given pair IDs.")

      // Initialize all pair poses to identity.
      .def(
          "initialize_all_poses_identity",
          [](ViewGraph& self) {
            const Rigid3d identity;
            for (auto& [pair_id, ip] : self.image_pairs) {
              ip.cam2_from_cam1 = identity;
            }
            return self.image_pairs.size();
          },
          "Set all pair poses to identity. Returns number of pairs.")

      // (id1, id2) for ALL pairs.
      .def(
          "extract_all_pair_image_ids",
          [](const ViewGraph& self) {
            py::dict result;
            for (const auto& [pair_id, ip] : self.image_pairs) {
              result[py::int_(pair_id)] =
                  py::make_tuple(ip.image_id1, ip.image_id2);
            }
            return result;
          },
          "Batch extract (image_id1, image_id2) tuples for ALL pairs.")

      // Extract (matches, are_lc) for given pair IDs.
      .def(
          "extract_pair_matches_and_lc",
          [](const ViewGraph& self,
             const std::vector<image_pair_t>& pair_ids) {
            py::dict result;
            for (const auto pid : pair_ids) {
              auto it = self.image_pairs.find(pid);
              if (it == self.image_pairs.end()) continue;
              const ImagePair& ip = it->second;

              Eigen::MatrixXi matches_copy = ip.matches;

              const size_t n = ip.are_lc.size();
              py::array_t<bool> are_lc_arr(n);
              auto mut = are_lc_arr.mutable_unchecked<1>();
              for (size_t i = 0; i < n; ++i) mut(i) = ip.are_lc[i];

              py::dict pd;
              pd["id1"] = ip.image_id1;
              pd["id2"] = ip.image_id2;
              pd["matches"] = std::move(matches_copy);
              pd["are_lc"] = std::move(are_lc_arr);
              result[py::int_(pid)] = std::move(pd);
            }
            return result;
          },
          py::arg("pair_ids"),
          "Batch extract (id1, id2, matches, are_lc) for given pair IDs.")

      // Batch set pair config (VGC two-stage).
      .def(
          "batch_set_pair_configs",
          [](ViewGraph& self,
             const std::vector<image_pair_t>& pair_ids,
             int config) {
            int count = 0;
            for (const auto pid : pair_ids) {
              auto it = self.image_pairs.find(pid);
              if (it != self.image_pairs.end()) {
                it->second.config = config;
                ++count;
              }
            }
            return count;
          },
          py::arg("pair_ids"),
          py::arg("config"),
          "Set config for given pair IDs. Returns count set.")

      // Extract pair configs (for saving before VGC).
      .def(
          "extract_pair_configs",
          [](const ViewGraph& self,
             const std::vector<image_pair_t>& pair_ids) {
            py::dict result;
            for (const auto pid : pair_ids) {
              auto it = self.image_pairs.find(pid);
              if (it != self.image_pairs.end()) {
                result[py::int_(pid)] = it->second.config;
              }
            }
            return result;
          },
          py::arg("pair_ids"),
          "Extract config values for given pair IDs.")

      // Restore pair configs from dict.
      .def(
          "restore_pair_configs",
          [](ViewGraph& self, const py::dict& configs) {
            int count = 0;
            for (const auto& item : configs) {
              const image_pair_t pid = item.first.cast<image_pair_t>();
              const int config = item.second.cast<int>();
              auto it = self.image_pairs.find(pid);
              if (it != self.image_pairs.end()) {
                it->second.config = config;
                ++count;
              }
            }
            return count;
          },
          py::arg("configs"),
          "Restore config values from dict[pair_id -> config]. Returns count.")

      // Extract is_valid for given pair IDs.
      .def(
          "extract_pair_validity",
          [](const ViewGraph& self,
             const std::vector<image_pair_t>& pair_ids) {
            py::dict result;
            for (const auto pid : pair_ids) {
              auto it = self.image_pairs.find(pid);
              if (it != self.image_pairs.end()) {
                result[py::int_(pid)] = it->second.is_valid;
              }
            }
            return result;
          },
          py::arg("pair_ids"),
          "Extract is_valid for given pair IDs.")

      // Restore validity: set is_valid=true where was true before but false now.
      .def(
          "restore_validity_where_invalidated",
          [](ViewGraph& self, const py::dict& validity_before) {
            int count = 0;
            for (const auto& item : validity_before) {
              const image_pair_t pid = item.first.cast<image_pair_t>();
              const bool was_valid = item.second.cast<bool>();
              if (!was_valid) continue;
              auto it = self.image_pairs.find(pid);
              if (it != self.image_pairs.end() && !it->second.is_valid) {
                it->second.is_valid = true;
                ++count;
              }
            }
            return count;
          },
          py::arg("validity_before"),
          "Restore is_valid=true where was true before but false now. Returns "
          "count restored.")

      // Full iterative LC forward propagation in C++.
      .def(
          "propagate_lc_forward",
          [](ViewGraph& self, const py::dict& initial_tainted_dict) {
            std::unordered_map<image_t, std::unordered_set<int>> tainted;
            for (const auto& item : initial_tainted_dict) {
              const image_t image_id = item.first.cast<image_t>();
              const auto indices = item.second.cast<std::vector<int>>();
              tainted[image_id].insert(indices.begin(), indices.end());
            }

            int total_new_lc = 0;
            int iteration = 0;

            while (true) {
              ++iteration;
              int new_lc_count = 0;
              std::unordered_map<image_t, std::unordered_set<int>>
                  newly_tainted;

              for (auto& [pair_id, ip] : self.image_pairs) {
                if (!ip.is_valid) continue;

                for (const int idx : ip.inliers) {
                  if (idx >= static_cast<int>(ip.are_lc.size())) continue;
                  if (ip.are_lc[idx]) continue;

                  const int kp1_idx = ip.matches(idx, 0);
                  auto it = tainted.find(ip.image_id1);
                  if (it != tainted.end() && it->second.count(kp1_idx)) {
                    ip.are_lc[idx] = true;
                    ++new_lc_count;
                    const int kp2_idx = ip.matches(idx, 1);
                    newly_tainted[ip.image_id2].insert(kp2_idx);
                  }
                }
              }

              for (auto& [img_id, kps] : newly_tainted) {
                tainted[img_id].insert(kps.begin(), kps.end());
              }
              total_new_lc += new_lc_count;

              if (new_lc_count == 0) break;
            }

            size_t total_tainted = 0;
            for (const auto& [img_id, kps] : tainted) {
              total_tainted += kps.size();
            }

            return py::make_tuple(total_new_lc, total_tainted, iteration);
          },
          py::arg("initial_tainted"),
          "Propagate LC forward from tainted keypoints until convergence. "
          "Takes dict[image_id -> list[kp_idx]]. "
          "Returns (total_lc_marked, total_tainted, iterations).")

      // (id1, id2, num_matches) for all pairs — avoids copying match matrices.
      .def(
          "extract_all_pair_summary",
          [](const ViewGraph& self) {
            py::dict result;
            for (const auto& [pair_id, ip] : self.image_pairs) {
              result[py::int_(pair_id)] =
                  py::make_tuple(ip.image_id1,
                                 ip.image_id2,
                                 static_cast<int>(ip.matches.rows()));
            }
            return result;
          },
          "Extract (id1, id2, num_matches) for all pairs.")

      // Initialize are_lc to all-false for every pair.
      .def(
          "initialize_all_are_lc_false",
          [](ViewGraph& self) {
            size_t total = 0;
            for (auto& [pair_id, ip] : self.image_pairs) {
              const size_t n = ip.matches.rows();
              ip.are_lc.assign(n, false);
              total += n;
            }
            return total;
          },
          "Initialize are_lc to all-false for every pair. Returns total "
          "matches.")

      // Batch set are_lc from dict[pair_id -> list[bool]].
      .def(
          "batch_set_are_lc",
          [](ViewGraph& self, const py::dict& lc_dict) {
            int count = 0;
            for (const auto& item : lc_dict) {
              const image_pair_t pid = item.first.cast<image_pair_t>();
              auto it = self.image_pairs.find(pid);
              if (it == self.image_pairs.end()) continue;
              it->second.are_lc = item.second.cast<std::vector<bool>>();
              ++count;
            }
            return count;
          },
          py::arg("lc_dict"),
          "Batch set are_lc from dict[pair_id -> list[bool]]. Returns count.")

      // Filter pairs by image set.
      .def(
          "filter_pairs_by_image_set",
          [](ViewGraph& self,
             const std::unordered_set<image_t>& keep_ids) {
            int removed = 0;
            for (auto it = self.image_pairs.begin();
                 it != self.image_pairs.end();) {
              const auto& ip = it->second;
              if (keep_ids.count(ip.image_id1) == 0 ||
                  keep_ids.count(ip.image_id2) == 0) {
                it = self.image_pairs.erase(it);
                ++removed;
              } else {
                ++it;
              }
            }
            return removed;
          },
          py::arg("keep_ids"),
          "Remove all pairs where either image is not in keep_ids. Returns "
          "count removed.")

      .def("__repr__", [](const ViewGraph& self) {
        std::ostringstream ss;
        ss << "ViewGraph(" << self.num_images << ", " << self.num_pairs << ")";
        return ss.str();
      })
      .def(
          "clone",
          [](const ViewGraph& self) {
            auto vg = std::make_shared<ViewGraph>();
            vg->image_pairs = self.image_pairs;
            vg->num_images = self.num_images;
            vg->num_pairs = self.num_pairs;
            return vg;
          },
          "Create a deep copy of this view graph.");
  MakeDataclass(PyViewGraph);
}
