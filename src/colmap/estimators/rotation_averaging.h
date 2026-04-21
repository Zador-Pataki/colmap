#pragma once

#include "colmap/geometry/pose_prior.h"
#include "colmap/scene/pose_graph.h"
#include "colmap/scene/reconstruction.h"

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <Eigen/Core>

// Code is adapted from Theia's RobustRotationEstimator
// (http://www.theia-sfm.org/). For gravity aligned rotation averaging, refer
// to the paper "Gravity Aligned Rotation Averaging"
namespace colmap {

struct RotationEstimatorOptions {
  // PRNG seed for stochastic methods during rotation averaging.
  // If -1 (default), the seed is derived from the current time
  // (non-deterministic). If >= 0, the rotation averaging is deterministic with
  // the given seed.
  int random_seed = -1;

  // Maximum number of times to run L1 minimization.
  int max_num_l1_iterations = 5;

  // Average step size threshold to terminate the L1 minimization.
  double l1_step_convergence_threshold = 0.001;

  // The number of iterative reweighted least squares iterations to perform.
  int max_num_irls_iterations = 100;

  // Average step size threshold to terminate the IRLS minimization.
  double irls_step_convergence_threshold = 0.001;

  // Gravity direction.
  Eigen::Vector3d gravity_dir = Eigen::Vector3d(0, 1, 0);

  // The point where the Huber-like cost function switches from L1 to L2.
  double irls_loss_parameter_sigma = 5.0;  // in degrees

  enum WeightType {
    // Geman-McClure weight from "Efficient and robust large-scale rotation
    // averaging" (Chatterjee et al., 2013)
    GEMAN_MCCLURE,
    // Half norm from "Robust Relative Rotation Averaging"
    // (Chatterjee et al., 2017)
    HALF_NORM,
  } weight_type = GEMAN_MCCLURE;

  // Flag to skip maximum spanning tree initialization.
  bool skip_initialization = false;

  // Flag to use gravity priors for rotation averaging.
  bool use_gravity = false;

  // Flag to use stratified solving for mixed gravity systems.
  // If true and use_gravity is true, first solves the 1-DOF system with
  // gravity-only pairs, then solves the full 3-DOF system.
  bool use_stratified = true;

  // If true, only consider frames with existing poses when computing
  // connected components. Set to true for refinement passes.
  bool filter_unregistered = false;

  // If > 0, filter image pairs with rotation error exceeding this threshold
  // after solving, then recompute active set.
  double max_rotation_error_deg = 10.0;

  // --- Fork-only fields ---

  // Fork's gravity axis (matches upstream gravity_dir default; kept separate
  // to avoid renaming call sites in the fork subclass).
  Eigen::Vector3d axis = Eigen::Vector3d(0, 1, 0);

  // If true, use pre-computed weights from image pairs instead of IRLS.
  bool use_precomputed_weights = false;

  // If true, fix weights for non-LC pairs during IRLS at fixed_non_lc_weight.
  bool fix_non_lc_weights = false;

  // Fixed weight for non-LC pairs when fix_non_lc_weights is true.
  // Default 138.0 is the theoretical maximum from Chatterjee et al.
  double fixed_non_lc_weight = 138.0;

  // If true, skip pairs where LC inliers > normal inliers (risky pairs).
  bool skip_risky_LC_pairs = false;

  // Enable video-aware Ceres solver with differential loss functions.
  bool use_video_constraints = false;

  // Loss scales for the video-aware Ceres solver.
  double video_tracking_huber_scale = 0.1;  // radians (~5.7 degrees)
  double video_lc_cauchy_scale = 0.05;      // radians (~2.8 degrees)

  // Soft gravity prior for Ceres solver.
  bool use_gravity_prior = false;
  Eigen::Vector3d gravity_world = Eigen::Vector3d(0, -1, 0);
  double default_gravity_sigma = 0.035;  // radians (~2 degrees)
  double gravity_loss_scale = 0.05;      // radians (~2.87 degrees)
  std::string gravity_loss_type = "cauchy";  // "cauchy" | "huber" | "trivial"
};

// High-level interface for rotation averaging.
// Combines problem setup and solving into a single call.
// TODO: Refactor this class into free functions (e.g., EstimateGlobalRotations)
// since it holds no state other than options.
class RotationEstimator {
 public:
  explicit RotationEstimator(const RotationEstimatorOptions& options)
      : options_(options) {}
  virtual ~RotationEstimator() = default;

  // Estimates the global orientations of all views.
  // Solves rotation averaging and registers frames with computed poses.
  // active_image_ids defines which images to include.
  // Returns true on successful estimation.
  virtual bool EstimateRotations(const PoseGraph& pose_graph,
                         const std::vector<PosePrior>& pose_priors,
                         const std::unordered_set<image_t>& active_image_ids,
                         Reconstruction& reconstruction);

  // Extended variant returning per-pair IRLS weights (or edge.weight for Ceres
  // path). Weight map has one entry per edge in pose_graph.Edges().
  // In the IRLS path, returns edge metadata weights; Ceres path returns
  // edge.weight (with fix_non_lc_weights override when enabled).
  std::pair<bool, std::unordered_map<image_pair_t, double>>
  EstimateRotationsWithWeights(
      const PoseGraph& pose_graph,
      const std::vector<PosePrior>& pose_priors,
      const std::unordered_set<image_t>& active_image_ids,
      Reconstruction& reconstruction);

 private:
  // Maybe solves 1-DOF rotation averaging on the gravity-aligned subset.
  // This is the first phase of stratified solving for mixed gravity systems.
  bool MaybeSolveGravityAlignedSubset(
      const PoseGraph& pose_graph,
      const std::vector<PosePrior>& pose_priors,
      const std::unordered_set<image_t>& active_image_ids,
      Reconstruction& reconstruction);

  // Core rotation averaging solver.
  bool SolveRotationAveraging(
      const PoseGraph& pose_graph,
      const std::vector<PosePrior>& pose_priors,
      const std::unordered_set<image_t>& active_image_ids,
      Reconstruction& reconstruction);

  // Initializes rotations from maximum spanning tree.
  void InitializeFromMaximumSpanningTree(
      const PoseGraph& pose_graph,
      const std::unordered_set<image_t>& active_image_ids,
      Reconstruction& reconstruction);

  const RotationEstimatorOptions options_;
};

// Initialize rig rotations by averaging per-image rotations.
// Estimates cam_from_rig for cameras with unknown calibration,
// then computes rig_from_world for each frame.
bool InitializeRigRotationsFromImages(
    const std::unordered_map<image_t, Rigid3d>& cams_from_world,
    Reconstruction& reconstruction);

// High-level rotation averaging solver that handles rig expansion.
// For cameras with unknown cam_from_rig, first estimates their orientations
// independently using an expanded reconstruction, then initializes the
// cam_from_rig and runs rotation averaging on the original reconstruction.
bool RunRotationAveraging(const RotationEstimatorOptions& options,
                          PoseGraph& pose_graph,
                          Reconstruction& reconstruction,
                          const std::vector<PosePrior>& pose_priors);

// Extended variant of RunRotationAveraging that returns per-pair weights.
// Does NOT apply the post-solve edge filtering (max_rotation_error_deg) —
// filtering is handled by the caller (glomap pipeline).
std::pair<bool, std::unordered_map<image_pair_t, double>>
RunRotationAveragingWithWeights(const RotationEstimatorOptions& options,
                                PoseGraph& pose_graph,
                                Reconstruction& reconstruction,
                                const std::vector<PosePrior>& pose_priors = {});

}  // namespace colmap
