#pragma once

#include "colmap/estimators/glomap/rotation_estimator_options.h"
#include "colmap/glomap/image.h"
#include "colmap/glomap/image_pair.h"
#include "colmap/glomap/math/l1_solver.h"
#include "colmap/glomap/track.h"
#include "colmap/glomap/view_graph.h"

#include <Eigen/SparseCore>


#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// Code is adapted from Theia's RobustRotationEstimator
// (http://www.theia-sfm.org/). For gravity aligned rotation averaging, refere
// to the paper "Gravity Aligned Rotation Averaging"
namespace colmap::glomap {

// The struct to store the temporary information for each image pair
struct ImagePairTempInfo {
  // The index of relative pose in the residual vector
  image_pair_t index = -1;

  // Whether the relative rotation is gravity aligned
  double has_gravity = false;

  // The relative rotation between the two images (x, z component)
  double xz_error = 0;

  // R_rel is gravity aligned if gravity prior is available, otherwise it is the
  // relative rotation between the two images
  Eigen::Matrix3d R_rel = Eigen::Matrix3d::Identity();

  // angle_rel is the converted angle if gravity prior is available for both
  // images
  double angle_rel = 0;
};

// RotationEstimatorOptions is defined in rotation_estimator_options.h (§07).

// TODO: Implement the stratified camera rotation estimation
// TODO: Implement the HALF_NORM loss for IRLS
// TODO: Implement the weighted version for rotation averaging
// TODO: Implement the gravity as prior for rotation averaging
class RotationEstimator {
 public:
  explicit RotationEstimator(const RotationEstimatorOptions& options)
      : options_(options) {}

  // Estimates the global orientations of all views based on an initial
  // guess. Returns true on successful estimation and false otherwise.
  std::pair<bool, std::unordered_map<image_pair_t, double>> EstimateRotations(
      const ViewGraph& view_graph, std::unordered_map<image_t, Image>& images);

 protected:
  // Initialize the rotation from the maximum spanning tree
  // Number of inliers serve as weights
  void InitializeFromMaximumSpanningTree(
      const ViewGraph& view_graph, std::unordered_map<image_t, Image>& images);

  // Sets up the sparse linear system such that dR_ij = dR_j - dR_i. This is the
  // first-order approximation of the angle-axis rotations. This should only be
  // called once.
  void SetupLinearSystem(const ViewGraph& view_graph,
                         std::unordered_map<image_t, Image>& images);

  // Performs the L1 robust loss minimization.
  bool SolveL1Regression(const ViewGraph& view_graph,
                         std::unordered_map<image_t, Image>& images);

  // Performs the iteratively reweighted least squares.
  std::pair<bool, std::unordered_map<image_pair_t, double>> SolveIRLS(
      const ViewGraph& view_graph, std::unordered_map<image_t, Image>& images);

  // Performs weighted least squares using pre-computed weights from image_pairs
  std::pair<bool, std::unordered_map<image_pair_t, double>> SolveWeightedLS(
      const ViewGraph& view_graph, std::unordered_map<image_t, Image>& images);

  // Performs video-aware rotation averaging using Ceres with differential
  // losses.
  bool SolveCeres(const ViewGraph& view_graph,
                  std::unordered_map<image_t, Image>& images);

  // Helper to classify if a pair is a tracking pair (normal inliers >= LC
  // inliers).
  static bool IsTrackingPair(const ImagePair& image_pair);

  // Updates the global rotations based on the current rotation change.
  void UpdateGlobalRotations(const ViewGraph& view_graph,
                             std::unordered_map<image_t, Image>& images);

  // Computes the relative rotation (tangent space) residuals based on the
  // current global orientation estimates.
  void ComputeResiduals(const ViewGraph& view_graph,
                        std::unordered_map<image_t, Image>& images);

  // Computes the average size of the most recent step of the algorithm.
  // The is the average over all non-fixed global_orientations_ of their
  // rotation magnitudes.
  double ComputeAverageStepSize(
      const std::unordered_map<image_t, Image>& images);

  // Data
  // Options for the solver.
  const RotationEstimatorOptions& options_;

  // The sparse matrix used to maintain the linear system. This is matrix A in
  // Ax = b.
  Eigen::SparseMatrix<double> sparse_matrix_;

  // x in the linear system Ax = b.
  Eigen::VectorXd tangent_space_step_;

  // b in the linear system Ax = b.
  Eigen::VectorXd tangent_space_residual_;

  Eigen::VectorXd rotation_estimated_;

  // Varaibles for intermidiate results
  std::unordered_map<image_t, image_t> image_id_to_idx_;
  std::unordered_map<image_pair_t, ImagePairTempInfo> rel_temp_info_;

  // The fixed camera id. This is used to remove the ambiguity of the linear
  image_t fixed_camera_id_ = -1;

  // The fixed camera rotation (if with initialization, it would not be identity
  // matrix)
  Eigen::Vector3d fixed_camera_rotation_;
};

}  // namespace colmap::glomap
