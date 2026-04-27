#include "colmap/sfm/rotation_averaging_glomap.h"

#include "colmap/geometry/rigid3.h"          // Inverse(Rigid3d)
#include "colmap/math/math.h"                // DegToRad
#include "colmap/scene/correspondence_graph.h"  // ImagePairToPairId
#include "colmap/sfm/l1_solver_glomap.h"
#include "colmap/sfm/rigid3d_glomap.h"
#include "colmap/sfm/tree_glomap.h"
#include "colmap/util/logging.h"

#include <iostream>
#include <queue>
#include <unordered_set>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace colmap {
namespace glomap_ra {
namespace {

// Constants from Eigen / glomap.
#ifndef TWO_PI
constexpr double TWO_PI = 2.0 * EIGEN_PI;
#endif
constexpr double EPS = 1e-12;

// --- Gravity stubs ---
// Glomap's RA reads image.gravity_info inside `if (options_.use_gravity)`
// branches. Our forked colmap::Image has no gravity field; the videosfm
// pipeline never enables use_gravity. Stub here so the dead branches still
// compile. If/when gravity support is needed, plumb gravity_info through
// colmap::Image and replace these stubs.
struct GravityInfoStub {
  bool has_gravity = false;
  Eigen::Matrix3d GetRAlign() const { return Eigen::Matrix3d::Identity(); }
};
inline GravityInfoStub gravity_info_for(const Image&) { return {}; }

inline Eigen::Matrix3d AngleToRotUp(double /*angle*/) {
  return Eigen::Matrix3d::Identity();
}
inline double RotUpToAngle(const Eigen::Matrix3d& /*R*/) { return 0.0; }
// --- end gravity stubs ---

}  // namespace

int KeepLargestConnectedComponents(
    ViewGraph& view_graph, std::unordered_map<image_t, Image>& images) {
  // Build adjacency list over valid pairs.
  std::unordered_map<image_t, std::vector<image_t>> adj;
  adj.reserve(images.size());
  for (const auto& [pair_id, image_pair] : view_graph.MutableImagePairs()) {
    if (!image_pair.is_valid) continue;
    adj[image_pair.image_id1].push_back(image_pair.image_id2);
    adj[image_pair.image_id2].push_back(image_pair.image_id1);
  }

  // BFS each unvisited image to enumerate connected components.
  std::unordered_set<image_t> visited;
  visited.reserve(images.size());
  std::vector<std::unordered_set<image_t>> components;
  for (const auto& [image_id, _] : images) {
    if (visited.count(image_id)) continue;
    std::unordered_set<image_t> comp;
    std::queue<image_t> q;
    q.push(image_id);
    visited.insert(image_id);
    while (!q.empty()) {
      const image_t curr = q.front();
      q.pop();
      comp.insert(curr);
      auto it = adj.find(curr);
      if (it == adj.end()) continue;
      for (const image_t nbr : it->second) {
        if (visited.count(nbr)) continue;
        visited.insert(nbr);
        q.push(nbr);
      }
    }
    if (!comp.empty()) components.push_back(std::move(comp));
  }

  // Pick the largest.
  size_t max_size = 0;
  int max_idx = -1;
  for (size_t i = 0; i < components.size(); ++i) {
    if (components[i].size() > max_size) {
      max_size = components[i].size();
      max_idx = static_cast<int>(i);
    }
  }
  if (max_idx < 0) return 0;
  const auto& largest = components[max_idx];

  // Reset registration; mark only images in the largest CC.
  for (auto& [_, image] : images) image.is_registered = false;
  for (const image_t image_id : largest) images[image_id].is_registered = true;

  // Invalidate pairs that cross the CC boundary.
  for (auto& [_, image_pair] : view_graph.MutableImagePairs()) {
    if (!image_pair.is_valid) continue;
    if (!images[image_pair.image_id1].is_registered ||
        !images[image_pair.image_id2].is_registered) {
      image_pair.is_valid = false;
    }
  }

  return static_cast<int>(largest.size());
}

namespace {
double RelAngleError(double angle_12, double angle_1, double angle_2) {
  double est = (angle_2 - angle_1) - angle_12;

  while (est >= EIGEN_PI) est -= TWO_PI;

  while (est < -EIGEN_PI) est += TWO_PI;

  return est;
}

// Ceres cost functor for relative rotation error
struct RelativeRotationError {
  RelativeRotationError(const Eigen::Vector3d& rel_rot_aa)
      : rel_rot_aa_(rel_rot_aa) {}

  template <typename T>
  bool operator()(const T* const r1_ptr,
                  const T* const r2_ptr,
                  T* residuals) const {
    // Map inputs to Eigen for safe matrix math (handles ColMajor automatically)
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> r1_aa(r1_ptr);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> r2_aa(r2_ptr);
    Eigen::Map<Eigen::Matrix<T, 3, 1>> res(residuals);

    Eigen::Matrix<T, 3, 3> R1, R2, R_rel;
    ceres::AngleAxisToRotationMatrix(r1_ptr, R1.data());
    ceres::AngleAxisToRotationMatrix(r2_ptr, R2.data());

    // Cast pre-computed relative rotation to T
    Eigen::Matrix<T, 3, 1> rel_aa_t = rel_rot_aa_.cast<T>();
    ceres::AngleAxisToRotationMatrix(rel_aa_t.data(), R_rel.data());

    // Form relation: R2 = R_rel * R1  =>  I = R2^T * R_rel * R1
    // Using Eigen ensures correct multiplication order and memory layout
    Eigen::Matrix<T, 3, 3> R_err = R2.transpose() * R_rel * R1;

    ceres::RotationMatrixToAngleAxis(R_err.data(), residuals);
    return true;
  }

  static ceres::CostFunction* Create(const Eigen::Vector3d& rel_rot_aa) {
    return (new ceres::AutoDiffCostFunction<RelativeRotationError, 3, 3, 3>(
        new RelativeRotationError(rel_rot_aa)));
  }

  const Eigen::Vector3d rel_rot_aa_;
};

}  // namespace

std::unordered_map<image_pair_t, double>
RotationEstimator::EstimateRotations(
    ViewGraph& view_graph, std::unordered_map<image_t, Image>& images) {
  bool use_video_constraints = options_.use_video_constraints;
  if (use_video_constraints && options_.use_gravity) {
    LOG(WARNING) << "use_video_constraints is incompatible with use_gravity. "
                    "Disabling video constraints.";
    use_video_constraints = false;
  }

  if (!options_.skip_initialization && !options_.use_gravity) {
    InitializeFromMaximumSpanningTree(view_graph, images);
  }

  SetupLinearSystem(view_graph, images);

  std::unordered_map<image_pair_t, double> final_weights;

  if (use_video_constraints) {
    if (!SolveCeres(view_graph, images)) {
      LOG(WARNING) << "Video-Aware Ceres solver failed.";
      return {};
    }
  } else {
    if (options_.max_num_l1_iterations > 0) {
      if (!SolveL1Regression(view_graph, images)) {
        LOG(WARNING) << "L1 Regression failed.";
        return {};
      }
    }

    if (options_.max_num_irls_iterations > 0) {
      auto irls_result = SolveIRLS(view_graph, images);
      final_weights = irls_result.second;
      if (!irls_result.first) {
        LOG(WARNING) << "IRLS failed.";
        return final_weights;
      }
    }
  }

  for (auto& [image_id, image] : images) {
    if (!image.is_registered) continue;

    if (options_.use_gravity && gravity_info_for(image).has_gravity) {
      image.cam_from_world.rotation() = Eigen::Quaterniond(
          gravity_info_for(image).GetRAlign() *
          AngleToRotUp(rotation_estimated_[image_id_to_idx_[image_id]]));
    } else {
      image.cam_from_world.rotation() = Eigen::Quaterniond(AngleAxisToRotation(
          rotation_estimated_.segment(image_id_to_idx_[image_id], 3)));
    }
  }

  return final_weights;
}

void RotationEstimator::InitializeFromMaximumSpanningTree(
    ViewGraph& view_graph, std::unordered_map<image_t, Image>& images) {
  // Here, we assume that largest connected component is already retrieved, so
  // we do not need to do that again compute maximum spanning tree.
  std::unordered_map<image_t, image_t> parents;
  // Note: prioritize_tracking is only used if video constraints are enabled
  // but we need to check options_ here since we don't have access to
  // use_video_constraints
  image_t root = MaximumSpanningTree(
      view_graph, images, parents, INLIER_NUM, options_.use_video_constraints);

  // Iterate through the tree to initialize the rotation
  // Establish child info
  std::unordered_map<image_t, std::vector<image_t>> children;
  for (const auto& [image_id, image] : images) {
    if (!image.is_registered) continue;
    children.insert(std::make_pair(image_id, std::vector<image_t>()));
  }
  for (auto& [child, parent] : parents) {
    if (root == child) continue;
    children[parent].emplace_back(child);
  }

  std::queue<image_t> indexes;
  indexes.push(root);

  while (!indexes.empty()) {
    image_t curr = indexes.front();
    indexes.pop();

    // Add all children into the tree
    for (auto& child : children[curr]) indexes.push(child);
    // If it is root, then fix it to be the original estimation
    if (curr == root) continue;

    // Directly use the relative pose for estimation rotation
    const ImagePair& image_pair = view_graph.MutableImagePairs().at(
        colmap::ImagePairToPairId(curr, parents[curr]));
    if (image_pair.image_id1 == curr) {
      // 1_R_w = 2_R_1^T * 2_R_w
      images[curr].cam_from_world.rotation() =
          (Inverse((*image_pair.two_view_geometry.cam2_from_cam1)) *
           images[parents[curr]].cam_from_world)
              .rotation();
    } else {
      // 2_R_w = 2_R_1 * 1_R_w
      images[curr].cam_from_world.rotation() =
          ((*image_pair.two_view_geometry.cam2_from_cam1) * images[parents[curr]].cam_from_world)
              .rotation();
    }
  }
}

void RotationEstimator::SetupLinearSystem(
    ViewGraph& view_graph, std::unordered_map<image_t, Image>& images) {
  // Clear all the structures
  sparse_matrix_.resize(0, 0);
  tangent_space_step_.resize(0);
  tangent_space_residual_.resize(0);
  rotation_estimated_.resize(0);
  image_id_to_idx_.clear();
  rel_temp_info_.clear();

  // Initialize the structures for estimated rotation
  image_id_to_idx_.reserve(images.size());
  rotation_estimated_.resize(
      3 * images.size());  // allocate more memory than needed
  image_t num_dof = 0;
  for (auto& [image_id, image] : images) {
    if (!image.is_registered) continue;
    image_id_to_idx_[image_id] = num_dof;
    if (options_.use_gravity && gravity_info_for(image).has_gravity) {
      rotation_estimated_[num_dof] =
          RotUpToAngle(gravity_info_for(image).GetRAlign().transpose() *
                       image.cam_from_world.rotation().toRotationMatrix());
      num_dof++;

      if (fixed_camera_id_ == -1) {
        fixed_camera_rotation_ =
            Eigen::Vector3d(0, rotation_estimated_[num_dof - 1], 0);
        fixed_camera_id_ = image_id;
      }
    } else {
      rotation_estimated_.segment(num_dof, 3) =
          Rigid3dToAngleAxis(image.cam_from_world);
      num_dof += 3;
    }
  }

  // If no cameras are set to be fixed, then take the first camera
  if (fixed_camera_id_ == -1) {
    for (auto& [image_id, image] : images) {
      if (!image.is_registered) continue;
      fixed_camera_id_ = image_id;
      fixed_camera_rotation_ = Rigid3dToAngleAxis(image.cam_from_world);
      break;
    }
  }

  rotation_estimated_.conservativeResize(num_dof);

  // Prepare the relative information
  int counter = 0;

  for (auto& [pair_id, image_pair] : view_graph.MutableImagePairs()) {
    if (!image_pair.is_valid) continue;

    // Skip risky pairs if enabled (LC inliers > normal inliers)
    if (options_.skip_risky_LC_pairs && !image_pair.inliers.empty() &&
        !image_pair.are_lc.empty()) {
      int lc_inliers_count = 0;
      for (int inlier_idx : image_pair.inliers) {
        if (inlier_idx >= 0 &&
            inlier_idx < static_cast<int>(image_pair.are_lc.size()) &&
            image_pair.are_lc[inlier_idx]) {
          lc_inliers_count++;
        }
      }
      int normal_inliers_count =
          static_cast<int>(image_pair.inliers.size()) - lc_inliers_count;
      if (lc_inliers_count > normal_inliers_count) continue;
    }

    int image_id1 = image_pair.image_id1;
    int image_id2 = image_pair.image_id2;

    if (images.find(image_id1) == images.end() ||
        images.find(image_id2) == images.end() ||
        !images[image_id1].is_registered || !images[image_id2].is_registered) {
      continue;
    }

    rel_temp_info_[pair_id].R_rel =
        (*image_pair.two_view_geometry.cam2_from_cam1).rotation().toRotationMatrix();

    // Align the relative rotation to the gravity
    if (options_.use_gravity) {
      if (gravity_info_for(images[image_id1]).has_gravity) {
        rel_temp_info_[pair_id].R_rel =
            rel_temp_info_[pair_id].R_rel *
            gravity_info_for(images[image_id1]).GetRAlign();
      }

      if (gravity_info_for(images[image_id2]).has_gravity) {
        rel_temp_info_[pair_id].R_rel =
            gravity_info_for(images[image_id2]).GetRAlign().transpose() *
            rel_temp_info_[pair_id].R_rel;
      }
    }

    if (options_.use_gravity && gravity_info_for(images[image_id1]).has_gravity &&
        gravity_info_for(images[image_id2]).has_gravity) {
      counter++;
      Eigen::Vector3d aa = RotationToAngleAxis(rel_temp_info_[pair_id].R_rel);
      double error = aa[0] * aa[0] + aa[2] * aa[2];

      // Keep track of the error for x and z axis for gravity-aligned relative
      // pose
      rel_temp_info_[pair_id].xz_error = error;
      rel_temp_info_[pair_id].has_gravity = true;
      rel_temp_info_[pair_id].angle_rel = aa[1];
    } else {
      rel_temp_info_[pair_id].has_gravity = false;
    }
  }

  VLOG(2) << counter << " image pairs are gravity aligned" << std::endl;

  std::vector<Eigen::Triplet<double>> coeffs;
  coeffs.reserve(rel_temp_info_.size() * 6 + 3);

  // Establish linear systems
  size_t curr_pos = 0;
  for (const auto& [pair_id, image_pair] : view_graph.MutableImagePairs()) {
    if (!image_pair.is_valid) continue;

    // Skip risky pairs if enabled (LC inliers > normal inliers)
    if (options_.skip_risky_LC_pairs && !image_pair.inliers.empty() &&
        !image_pair.are_lc.empty()) {
      int lc_inliers_count = 0;
      for (int inlier_idx : image_pair.inliers) {
        if (inlier_idx >= 0 &&
            inlier_idx < static_cast<int>(image_pair.are_lc.size()) &&
            image_pair.are_lc[inlier_idx]) {
          lc_inliers_count++;
        }
      }
      int normal_inliers_count =
          static_cast<int>(image_pair.inliers.size()) - lc_inliers_count;
      if (lc_inliers_count > normal_inliers_count) {
        continue;
      }
    }

    int image_id1 = image_pair.image_id1;
    int image_id2 = image_pair.image_id2;

    // Check if images are registered (should already be filtered, but
    // double-check)
    if (images.find(image_id1) == images.end() ||
        images.find(image_id2) == images.end() ||
        !images[image_id1].is_registered || !images[image_id2].is_registered) {
      continue;
    }

    int vector_idx1 = image_id_to_idx_[image_id1];
    int vector_idx2 = image_id_to_idx_[image_id2];

    rel_temp_info_[pair_id].index = curr_pos;

    if (rel_temp_info_[pair_id].has_gravity) {
      coeffs.emplace_back(Eigen::Triplet<double>(curr_pos, vector_idx1, -1));
      coeffs.emplace_back(Eigen::Triplet<double>(curr_pos, vector_idx2, 1));
      curr_pos++;
    } else {
      // If it is not gravity aligned, then we need to consider 3 dof
      if (!options_.use_gravity ||
          !gravity_info_for(images[image_id1]).has_gravity) {
        for (int i = 0; i < 3; i++) {
          coeffs.emplace_back(
              Eigen::Triplet<double>(curr_pos + i, vector_idx1 + i, -1));
        }
      } else
        // else, other components are zero, and can be safely ignored
        coeffs.emplace_back(
            Eigen::Triplet<double>(curr_pos + 1, vector_idx1, -1));

      // Similarly for the second componenet
      if (!options_.use_gravity ||
          !gravity_info_for(images[image_id2]).has_gravity) {
        for (int i = 0; i < 3; i++) {
          coeffs.emplace_back(
              Eigen::Triplet<double>(curr_pos + i, vector_idx2 + i, 1));
        }
      } else
        coeffs.emplace_back(
            Eigen::Triplet<double>(curr_pos + 1, vector_idx2, 1));

      curr_pos += 3;
    }
  }

  // Set some cameras to be fixed
  // if some cameras have gravity, then add a single term constraint
  // Else, change to 3 constriants
  if (options_.use_gravity &&
      gravity_info_for(images[fixed_camera_id_]).has_gravity) {
    coeffs.emplace_back(Eigen::Triplet<double>(
        curr_pos, image_id_to_idx_[fixed_camera_id_], 1));
    curr_pos++;
  } else {
    for (int i = 0; i < 3; i++) {
      coeffs.emplace_back(Eigen::Triplet<double>(
          curr_pos + i, image_id_to_idx_[fixed_camera_id_] + i, 1));
    }
    curr_pos += 3;
  }

  sparse_matrix_.resize(curr_pos, num_dof);
  sparse_matrix_.setFromTriplets(coeffs.begin(), coeffs.end());

  // Initialize x and b
  tangent_space_step_.resize(num_dof);
  tangent_space_residual_.resize(curr_pos);
}

bool RotationEstimator::SolveL1Regression(
    ViewGraph& view_graph, std::unordered_map<image_t, Image>& images) {
  L1SolverOptions opt_l1_solver;
  opt_l1_solver.max_num_iterations = 10;

  L1Solver<Eigen::SparseMatrix<double>> l1_solver(opt_l1_solver,
                                                  sparse_matrix_);
  double last_norm = 0;
  double curr_norm = 0;

  ComputeResiduals(view_graph, images);
  VLOG(2) << "ComputeResiduals done";

  int iteration = 0;
  for (iteration = 0; iteration < options_.max_num_l1_iterations; iteration++) {
    VLOG(2) << "L1 ADMM iteration: " << iteration;

    last_norm = curr_norm;
    // use the current residual as b (Ax - b)

    tangent_space_step_.setZero();
    l1_solver.Solve(tangent_space_residual_, &tangent_space_step_);
    if (tangent_space_step_.array().isNaN().any()) {
      LOG(ERROR) << "nan error";
      iteration++;
      return false;
    }

    if (VLOG_IS_ON(2))
      LOG(INFO) << "residual:"
                << (sparse_matrix_ * tangent_space_step_ -
                    tangent_space_residual_)
                       .array()
                       .abs()
                       .sum();

    curr_norm = tangent_space_step_.norm();
    UpdateGlobalRotations(view_graph, images);
    ComputeResiduals(view_graph, images);

    // Check the residual. If it is small, stop
    // TODO: strange bug for the L1 solver: update norm state constant
    if (ComputeAverageStepSize(images) <
            options_.l1_step_convergence_threshold ||
        std::abs(last_norm - curr_norm) < EPS) {
      if (std::abs(last_norm - curr_norm) < EPS)
        LOG(INFO) << "std::abs(last_norm - curr_norm) < EPS";
      iteration++;
      break;
    }
    opt_l1_solver.max_num_iterations =
        std::min(opt_l1_solver.max_num_iterations * 2, 100);
  }
  VLOG(2) << "L1 ADMM total iteration: " << iteration;
  return true;
}

std::pair<bool, std::unordered_map<image_pair_t, double>>
RotationEstimator::SolveIRLS(ViewGraph& view_graph,
                             std::unordered_map<image_t, Image>& images) {
  std::unordered_map<image_pair_t, double> final_weights;

  // TODO: Determine what is the best solver for this part
  Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>> llt;

  // weight_matrix.setIdentity();
  // sparse_matrix_ = A_ori;

  llt.analyzePattern(sparse_matrix_.transpose() * sparse_matrix_);

  const double sigma = DegToRad(options_.irls_loss_parameter_sigma);
  VLOG(2) << "sigma: " << options_.irls_loss_parameter_sigma;

  Eigen::ArrayXd weights_irls(sparse_matrix_.rows());
  Eigen::SparseMatrix<double> at_weight;

  if (options_.use_gravity && gravity_info_for(images[fixed_camera_id_]).has_gravity)
    weights_irls[sparse_matrix_.rows() - 1] = 1;
  else
    weights_irls.segment(sparse_matrix_.rows() - 3, 3).setConstant(1);

  ComputeResiduals(view_graph, images);
  int iteration = 0;
  bool success = true;
  for (iteration = 0; iteration < options_.max_num_irls_iterations;
       iteration++) {
    VLOG(2) << "IRLS iteration: " << iteration;

    // Compute the weights for IRLS
    for (auto& [pair_id, pair_info] : rel_temp_info_) {
      image_pair_t image_pair_pos = pair_info.index;
      double err_squared = 0;
      double w = 0;
      // If both cameras have gravity, then we only consider the y-axis
      if (pair_info.has_gravity)
        err_squared = std::pow(tangent_space_residual_[image_pair_pos], 2) +
                      pair_info.xz_error;
      // Otherwise, we consider all 3 dof
      else
        err_squared =
            tangent_space_residual_.segment<3>(image_pair_pos).squaredNorm();

      // Compute the weight

      if (options_.weight_type == RotationEstimatorOptions::GEMAN_MCCLURE) {
        double tmp = err_squared + sigma * sigma;
        w = sigma * sigma / (tmp * tmp);
      } else if (options_.weight_type == RotationEstimatorOptions::HALF_NORM) {
        w = std::pow(err_squared, (0.5 - 2) / 2);
      }

      if (std::isnan(w)) {
        LOG(ERROR) << "nan weight!";
        success = false;
        break;
      }

      // If both cameras have gravity, then only 1 equation
      if (pair_info.has_gravity) weights_irls[image_pair_pos] = w;
      // Otherwise, 3 equations
      else
        weights_irls.segment<3>(image_pair_pos).setConstant(w);
    }

    if (!success) {
      break;
    }

    final_weights.clear();
    for (const auto& [pair_id, pair_info] : rel_temp_info_) {
      image_pair_t weight_index = pair_info.index;
      final_weights[pair_id] = weights_irls[weight_index];
    }

    // Update the factorization for the weighted values.
    at_weight = sparse_matrix_.transpose() * weights_irls.matrix().asDiagonal();

    llt.factorize(at_weight * sparse_matrix_);

    // Solve the least squares problem..
    tangent_space_step_.setZero();
    tangent_space_step_ = llt.solve(at_weight * tangent_space_residual_);

    if (tangent_space_step_.array().isNaN().any()) {  // Example check
      LOG(ERROR) << "NaN step detected!";
      success = false;
      break;
    }

    UpdateGlobalRotations(view_graph, images);
    ComputeResiduals(view_graph, images);

    // Check the residual. If it is small, stop
    if (ComputeAverageStepSize(images) <
        options_.irls_step_convergence_threshold) {
      iteration++;
      break;
    }
  }
  VLOG(2) << "IRLS total iteration: " << iteration;

  return {success, final_weights};
}

std::pair<bool, std::unordered_map<image_pair_t, double>>
RotationEstimator::SolveWeightedLS(ViewGraph& view_graph,
                                   std::unordered_map<image_t, Image>& images) {
  std::unordered_map<image_pair_t, double> final_weights;

  // Setup solver
  Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>> llt;
  llt.analyzePattern(sparse_matrix_.transpose() * sparse_matrix_);

  // Create weight array from image_pair weights
  Eigen::ArrayXd weights_precomputed(sparse_matrix_.rows());

  // Set weights for relative pose constraints from image_pairs
  for (const auto& [pair_id, pair_info] : rel_temp_info_) {
    if (view_graph.MutableImagePairs().find(pair_id) == view_graph.MutableImagePairs().end()) {
      LOG(WARNING) << "Image pair " << pair_id << " not found in view_graph";
      continue;
    }

    const ImagePair& image_pair = view_graph.MutableImagePairs().at(pair_id);
    double weight = image_pair.weight;

    // Store weights for return
    final_weights[pair_id] = weight;

    image_pair_t weight_index = pair_info.index;

    // If both cameras have gravity, only 1 equation
    if (pair_info.has_gravity) {
      weights_precomputed[weight_index] = weight;
    }
    // Otherwise, 3 equations (all get the same weight)
    else {
      weights_precomputed.segment<3>(weight_index).setConstant(weight);
    }
  }

  // Set weight for fixed camera constraint to 1.0
  if (options_.use_gravity &&
      gravity_info_for(images[fixed_camera_id_]).has_gravity) {
    weights_precomputed[sparse_matrix_.rows() - 1] = 1.0;
  } else {
    weights_precomputed.segment(sparse_matrix_.rows() - 3, 3).setConstant(1.0);
  }

  ComputeResiduals(view_graph, images);

  // Perform iterative weighted least squares (similar to IRLS but fixed
  // weights)
  int iteration = 0;
  bool success = true;
  int max_iterations = options_.max_num_irls_iterations > 0
                           ? options_.max_num_irls_iterations
                           : 100;  // Default to 100 if not set

  for (iteration = 0; iteration < max_iterations; iteration++) {
    VLOG(2) << "Weighted LS iteration: " << iteration;

    // Update the factorization with fixed weights
    Eigen::SparseMatrix<double> at_weight =
        sparse_matrix_.transpose() * weights_precomputed.matrix().asDiagonal();

    llt.factorize(at_weight * sparse_matrix_);

    // Solve the least squares problem
    tangent_space_step_.setZero();
    tangent_space_step_ = llt.solve(at_weight * tangent_space_residual_);

    if (tangent_space_step_.array().isNaN().any()) {
      LOG(ERROR) << "NaN step detected in weighted LS!";
      success = false;
      break;
    }

    UpdateGlobalRotations(view_graph, images);
    ComputeResiduals(view_graph, images);

    // Check convergence
    double convergence_threshold =
        options_.irls_step_convergence_threshold > 0
            ? options_.irls_step_convergence_threshold
            : 0.001;
    if (ComputeAverageStepSize(images) < convergence_threshold) {
      iteration++;
      break;
    }
  }

  VLOG(2) << "Weighted LS total iterations: " << iteration;

  return {success, final_weights};
}

void RotationEstimator::UpdateGlobalRotations(
    ViewGraph& view_graph, std::unordered_map<image_t, Image>& images) {
  for (const auto& [image_id, image] : images) {
    if (!image.is_registered) continue;

    image_t vector_idx = image_id_to_idx_[image_id];
    if (!(options_.use_gravity && gravity_info_for(image).has_gravity)) {
      Eigen::Matrix3d R_ori =
          AngleAxisToRotation(rotation_estimated_.segment(vector_idx, 3));

      rotation_estimated_.segment(vector_idx, 3) = RotationToAngleAxis(
          R_ori *
          AngleAxisToRotation(-tangent_space_step_.segment(vector_idx, 3)));
    } else {
      rotation_estimated_[vector_idx] -= tangent_space_step_[vector_idx];
    }
  }
}

void RotationEstimator::ComputeResiduals(
    ViewGraph& view_graph, std::unordered_map<image_t, Image>& images) {
  int curr_pos = 0;
  for (auto& [pair_id, pair_info] : rel_temp_info_) {
    image_t image_id1 = view_graph.MutableImagePairs().at(pair_id).image_id1;
    image_t image_id2 = view_graph.MutableImagePairs().at(pair_id).image_id2;

    image_t idx1 = image_id_to_idx_[image_id1];
    image_t idx2 = image_id_to_idx_[image_id2];

    if (pair_info.has_gravity) {
      tangent_space_residual_[pair_info.index] =
          (RelAngleError(pair_info.angle_rel,
                         rotation_estimated_[image_id_to_idx_[image_id1]],
                         rotation_estimated_[image_id_to_idx_[image_id2]]));
    } else {
      Eigen::Matrix3d R_1, R_2;
      if (options_.use_gravity && gravity_info_for(images[image_id1]).has_gravity) {
        R_1 = AngleToRotUp(rotation_estimated_[image_id_to_idx_[image_id1]]);
      } else {
        R_1 = AngleAxisToRotation(
            rotation_estimated_.segment(image_id_to_idx_[image_id1], 3));
      }

      if (options_.use_gravity && gravity_info_for(images[image_id2]).has_gravity) {
        R_2 = AngleToRotUp(rotation_estimated_[image_id_to_idx_[image_id2]]);
      } else {
        R_2 = AngleAxisToRotation(
            rotation_estimated_.segment(image_id_to_idx_[image_id2], 3));
      }

      tangent_space_residual_.segment(pair_info.index, 3) =
          -RotationToAngleAxis(R_2.transpose() * pair_info.R_rel * R_1);
    }
  }

  if (options_.use_gravity && gravity_info_for(images[fixed_camera_id_]).has_gravity)
    tangent_space_residual_[tangent_space_residual_.size() - 1] =
        rotation_estimated_[image_id_to_idx_[fixed_camera_id_]] -
        fixed_camera_rotation_[1];
  else
    tangent_space_residual_.segment(tangent_space_residual_.size() - 3, 3) =
        RotationToAngleAxis(
            AngleAxisToRotation(fixed_camera_rotation_).transpose() *
            AngleAxisToRotation(rotation_estimated_.segment(
                image_id_to_idx_[fixed_camera_id_], 3)));
}

double RotationEstimator::ComputeAverageStepSize(
    const std::unordered_map<image_t, Image>& images) {
  double total_update = 0;
  for (const auto& [image_id, image] : images) {
    if (!image.is_registered) continue;

    if (options_.use_gravity && gravity_info_for(image).has_gravity) {
      total_update += std::abs(tangent_space_step_[image_id_to_idx_[image_id]]);
    } else {
      total_update +=
          tangent_space_step_.segment(image_id_to_idx_[image_id], 3).norm();
    }
  }
  return total_update / image_id_to_idx_.size();
}

bool RotationEstimator::IsTrackingPair(const ImagePair& image_pair) {
  if (image_pair.inliers.empty()) return false;
  int lc_count = 0;
  for (int idx : image_pair.inliers) {
    if (idx >= 0 && idx < static_cast<int>(image_pair.are_lc.size()) &&
        image_pair.are_lc[idx]) {
      lc_count++;
    }
  }
  return (static_cast<int>(image_pair.inliers.size()) - lc_count) >= lc_count;
}

bool RotationEstimator::SolveCeres(ViewGraph& view_graph,
                                   std::unordered_map<image_t, Image>& images) {
  ceres::Problem problem;

  // Add parameter blocks for all registered images
  for (auto& [image_id, image] : images) {
    if (!image.is_registered) continue;

    double* param = rotation_estimated_.data() + image_id_to_idx_[image_id];
    problem.AddParameterBlock(param, 3);
    if (image_id == fixed_camera_id_) {
      problem.SetParameterBlockConstant(param);
    }
  }

  // Add residual blocks for all valid image pairs
  for (const auto& [pair_id, image_pair] : view_graph.MutableImagePairs()) {
    if (!image_pair.is_valid) continue;

    // Ensure both images are registered before adding residual
    if (images.find(image_pair.image_id1) == images.end() ||
        images.find(image_pair.image_id2) == images.end() ||
        !images.at(image_pair.image_id1).is_registered ||
        !images.at(image_pair.image_id2).is_registered) {
      continue;
    }

    Eigen::Vector3d rel_aa = RotationToAngleAxis(
        (*image_pair.two_view_geometry.cam2_from_cam1).rotation().toRotationMatrix());
    ceres::CostFunction* cost = RelativeRotationError::Create(rel_aa);

    ceres::LossFunction* loss =
        IsTrackingPair(image_pair)
            ? static_cast<ceres::LossFunction*>(
                  new ceres::HuberLoss(options_.video_tracking_huber_scale))
            : static_cast<ceres::LossFunction*>(
                  new ceres::CauchyLoss(options_.video_lc_cauchy_scale));

    problem.AddResidualBlock(
        cost,
        loss,
        rotation_estimated_.data() + image_id_to_idx_[image_pair.image_id1],
        rotation_estimated_.data() + image_id_to_idx_[image_pair.image_id2]);
  }

  ceres::Solver::Options solver_options;
  solver_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  solver_options.max_num_iterations = 100;
  ceres::Solver::Summary summary;
  ceres::Solve(solver_options, &problem, &summary);
  return summary.IsSolutionUsable();
}

}  // namespace glomap_ra
}  // namespace colmap