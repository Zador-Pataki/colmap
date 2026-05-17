#pragma once

#include <cstdint>
#include <vector>

#include "shared_indices.h"
#include "solver_params.h"
#include <cuda_runtime.h>

namespace caspar {

enum class ExitReason {
  MAX_ITERATIONS,
  CONVERGED_SCORE_THRESHOLD,
  CONVERGED_DIAG_EXIT
};

struct IterationData {
  int solver_iter;
  int pcg_iter;
  double score_current;
  double score_best;
  double step_quality;
  double diag;
  double dt_inc;
  double dt_tot;
  bool step_accepted;
};

struct SolveResult {
  double initial_score;
  double final_score;
  int iteration_count;
  double runtime;
  ExitReason exit_reason;
  std::vector<IterationData> iterations;
};

class GraphSolver {
 public:
  /**
   * Base constructor.
   *
   * @param params: The params to use for the solver
   * @param DepthScale_num_max the maximum number of DepthScales
   * @param PinholeCalib_num_max the maximum number of PinholeCalibs
   * @param PinholeFocal_num_max the maximum number of PinholeFocals
   * @param PinholePose_num_max the maximum number of PinholePoses
   * @param PinholePrincipalPoint_num_max the maximum number of
   * PinholePrincipalPoints
   * @param PinholeTranslation_num_max the maximum number of PinholeTranslations
   * @param Point_num_max the maximum number of Points
   * @param SimpleRadialCalib_num_max the maximum number of SimpleRadialCalibs
   * @param SimpleRadialFocalAndDistortion_num_max the maximum number of
   * SimpleRadialFocalAndDistortions
   * @param SimpleRadialPose_num_max the maximum number of SimpleRadialPoses
   * @param SimpleRadialPrincipalPoint_num_max the maximum number of
   * SimpleRadialPrincipalPoints
   * @param simple_radial_num_max the maximum number of simple_radials
   * @param simple_radial_fixed_pose_num_max the maximum number of
   * simple_radial_fixed_poses
   * @param simple_radial_fixed_point_num_max the maximum number of
   * simple_radial_fixed_points
   * @param simple_radial_fixed_pose_fixed_point_num_max the maximum number of
   * simple_radial_fixed_pose_fixed_points
   * @param pinhole_num_max the maximum number of pinholes
   * @param pinhole_fixed_pose_num_max the maximum number of pinhole_fixed_poses
   * @param pinhole_fixed_point_num_max the maximum number of
   * pinhole_fixed_points
   * @param pinhole_fixed_pose_fixed_point_num_max the maximum number of
   * pinhole_fixed_pose_fixed_points
   * @param pinhole_log_depth_num_max the maximum number of pinhole_log_depths
   * @param pinhole_log_depth_fixed_pose_num_max the maximum number of
   * pinhole_log_depth_fixed_poses
   * @param pinhole_log_depth_fixed_scale_num_max the maximum number of
   * pinhole_log_depth_fixed_scales
   * @param pinhole_log_depth_fixed_point_num_max the maximum number of
   * pinhole_log_depth_fixed_points
   * @param pinhole_log_depth_fixed_pose_fixed_scale_num_max the maximum number
   * of pinhole_log_depth_fixed_pose_fixed_scales
   * @param pinhole_log_depth_fixed_pose_fixed_point_num_max the maximum number
   * of pinhole_log_depth_fixed_pose_fixed_points
   * @param pinhole_log_depth_fixed_scale_fixed_point_num_max the maximum number
   * of pinhole_log_depth_fixed_scale_fixed_points
   * @param pinhole_fixed_rotation_num_max the maximum number of
   * pinhole_fixed_rotations
   * @param pinhole_fixed_rotation_fixed_calib_num_max the maximum number of
   * pinhole_fixed_rotation_fixed_calibs
   * @param pinhole_fixed_rotation_fixed_point_num_max the maximum number of
   * pinhole_fixed_rotation_fixed_points
   * @param pinhole_fixed_rotation_fixed_calib_fixed_point_num_max the maximum
   * number of pinhole_fixed_rotation_fixed_calib_fixed_points
   * @param pinhole_log_depth_fixed_rotation_num_max the maximum number of
   * pinhole_log_depth_fixed_rotations
   * @param pinhole_log_depth_fixed_rotation_fixed_scale_num_max the maximum
   * number of pinhole_log_depth_fixed_rotation_fixed_scales
   * @param pinhole_log_depth_fixed_rotation_fixed_point_num_max the maximum
   * number of pinhole_log_depth_fixed_rotation_fixed_points
   * @param pinhole_log_depth_fixed_rotation_fixed_scale_fixed_point_num_max the
   * maximum number of pinhole_log_depth_fixed_rotation_fixed_scale_fixed_points
   * @param pinhole_intrinsics_prior_num_max the maximum number of
   * pinhole_intrinsics_priors
   * @param pinhole_intrinsics_random_walk_num_max the maximum number of
   * pinhole_intrinsics_random_walks
   * @param scale_prior_num_max the maximum number of scale_priors
   * @param simple_radial_split_fixed_focal_and_distortion_num_max the maximum
   * number of simple_radial_split_fixed_focal_and_distortions
   * @param simple_radial_split_fixed_principal_point_num_max the maximum number
   * of simple_radial_split_fixed_principal_points
   * @param simple_radial_split_fixed_pose_fixed_focal_and_distortion_num_max
   * the maximum number of
   * simple_radial_split_fixed_pose_fixed_focal_and_distortions
   * @param simple_radial_split_fixed_pose_fixed_principal_point_num_max the
   * maximum number of simple_radial_split_fixed_pose_fixed_principal_points
   * @param
   * simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_num_max
   * the maximum number of
   * simple_radial_split_fixed_focal_and_distortion_fixed_principal_points
   * @param simple_radial_split_fixed_focal_and_distortion_fixed_point_num_max
   * the maximum number of
   * simple_radial_split_fixed_focal_and_distortion_fixed_points
   * @param simple_radial_split_fixed_principal_point_fixed_point_num_max the
   * maximum number of simple_radial_split_fixed_principal_point_fixed_points
   * @param
   * simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_principal_point_num_max
   * the maximum number of
   * simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_principal_points
   * @param
   * simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_point_num_max
   * the maximum number of
   * simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_points
   * @param
   * simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_max
   * the maximum number of
   * simple_radial_split_fixed_pose_fixed_principal_point_fixed_points
   * @param
   * simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_fixed_point_num_max
   * the maximum number of
   * simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_fixed_points
   * @param pinhole_split_fixed_focal_num_max the maximum number of
   * pinhole_split_fixed_focals
   * @param pinhole_split_fixed_principal_point_num_max the maximum number of
   * pinhole_split_fixed_principal_points
   * @param pinhole_split_fixed_pose_fixed_focal_num_max the maximum number of
   * pinhole_split_fixed_pose_fixed_focals
   * @param pinhole_split_fixed_pose_fixed_principal_point_num_max the maximum
   * number of pinhole_split_fixed_pose_fixed_principal_points
   * @param pinhole_split_fixed_focal_fixed_principal_point_num_max the maximum
   * number of pinhole_split_fixed_focal_fixed_principal_points
   * @param pinhole_split_fixed_focal_fixed_point_num_max the maximum number of
   * pinhole_split_fixed_focal_fixed_points
   * @param pinhole_split_fixed_principal_point_fixed_point_num_max the maximum
   * number of pinhole_split_fixed_principal_point_fixed_points
   * @param pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_max
   * the maximum number of
   * pinhole_split_fixed_pose_fixed_focal_fixed_principal_points
   * @param pinhole_split_fixed_pose_fixed_focal_fixed_point_num_max the maximum
   * number of pinhole_split_fixed_pose_fixed_focal_fixed_points
   * @param pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_max
   * the maximum number of
   * pinhole_split_fixed_pose_fixed_principal_point_fixed_points
   * @param pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_max
   * the maximum number of
   * pinhole_split_fixed_focal_fixed_principal_point_fixed_points
   * @param pinhole_split_fixed_rotation_fixed_focal_num_max the maximum number
   * of pinhole_split_fixed_rotation_fixed_focals
   * @param pinhole_split_fixed_rotation_fixed_principal_point_num_max the
   * maximum number of pinhole_split_fixed_rotation_fixed_principal_points
   * @param
   * pinhole_split_fixed_rotation_fixed_focal_fixed_principal_point_num_max the
   * maximum number of
   * pinhole_split_fixed_rotation_fixed_focal_fixed_principal_points
   * @param pinhole_split_fixed_rotation_fixed_focal_fixed_point_num_max the
   * maximum number of pinhole_split_fixed_rotation_fixed_focal_fixed_points
   * @param
   * pinhole_split_fixed_rotation_fixed_principal_point_fixed_point_num_max the
   * maximum number of
   * pinhole_split_fixed_rotation_fixed_principal_point_fixed_points
   * @param
   * pinhole_split_fixed_rotation_fixed_focal_fixed_principal_point_fixed_point_num_max
   * the maximum number of
   * pinhole_split_fixed_rotation_fixed_focal_fixed_principal_point_fixed_points
   * @param pinhole_split_intrinsics_prior_fixed_focal_num_max the maximum
   * number of pinhole_split_intrinsics_prior_fixed_focals
   * @param pinhole_split_intrinsics_prior_fixed_principal_point_num_max the
   * maximum number of pinhole_split_intrinsics_prior_fixed_principal_points
   * @param pinhole_split_intrinsics_random_walk_fixed_prev_focal_num_max the
   * maximum number of pinhole_split_intrinsics_random_walk_fixed_prev_focals
   * @param
   * pinhole_split_intrinsics_random_walk_fixed_prev_principal_point_num_max the
   * maximum number of
   * pinhole_split_intrinsics_random_walk_fixed_prev_principal_points
   * @param pinhole_split_intrinsics_random_walk_fixed_next_focal_num_max the
   * maximum number of pinhole_split_intrinsics_random_walk_fixed_next_focals
   * @param
   * pinhole_split_intrinsics_random_walk_fixed_next_principal_point_num_max the
   * maximum number of
   * pinhole_split_intrinsics_random_walk_fixed_next_principal_points
   * @param
   * pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_prev_principal_point_num_max
   * the maximum number of
   * pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_prev_principal_points
   * @param
   * pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_next_focal_num_max
   * the maximum number of
   * pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_next_focals
   * @param
   * pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_next_principal_point_num_max
   * the maximum number of
   * pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_next_principal_points
   * @param
   * pinhole_split_intrinsics_random_walk_fixed_prev_principal_point_fixed_next_focal_num_max
   * the maximum number of
   * pinhole_split_intrinsics_random_walk_fixed_prev_principal_point_fixed_next_focals
   * @param
   * pinhole_split_intrinsics_random_walk_fixed_prev_principal_point_fixed_next_principal_point_num_max
   * the maximum number of
   * pinhole_split_intrinsics_random_walk_fixed_prev_principal_point_fixed_next_principal_points
   * @param
   * pinhole_split_intrinsics_random_walk_fixed_next_focal_fixed_next_principal_point_num_max
   * the maximum number of
   * pinhole_split_intrinsics_random_walk_fixed_next_focal_fixed_next_principal_points
   * @param
   * pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_prev_principal_point_fixed_next_focal_num_max
   * the maximum number of
   * pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_prev_principal_point_fixed_next_focals
   * @param
   * pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_prev_principal_point_fixed_next_principal_point_num_max
   * the maximum number of
   * pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_prev_principal_point_fixed_next_principal_points
   * @param
   * pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_next_focal_fixed_next_principal_point_num_max
   * the maximum number of
   * pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_next_focal_fixed_next_principal_points
   * @param
   * pinhole_split_intrinsics_random_walk_fixed_prev_principal_point_fixed_next_focal_fixed_next_principal_point_num_max
   * the maximum number of
   * pinhole_split_intrinsics_random_walk_fixed_prev_principal_point_fixed_next_focal_fixed_next_principal_points
   */
  GraphSolver(
      const SolverParams<double>& params,
      size_t DepthScale_num_max,
      size_t PinholeCalib_num_max,
      size_t PinholeFocal_num_max,
      size_t PinholePose_num_max,
      size_t PinholePrincipalPoint_num_max,
      size_t PinholeTranslation_num_max,
      size_t Point_num_max,
      size_t SimpleRadialCalib_num_max,
      size_t SimpleRadialFocalAndDistortion_num_max,
      size_t SimpleRadialPose_num_max,
      size_t SimpleRadialPrincipalPoint_num_max,
      size_t simple_radial_num_max,
      size_t simple_radial_fixed_pose_num_max,
      size_t simple_radial_fixed_point_num_max,
      size_t simple_radial_fixed_pose_fixed_point_num_max,
      size_t pinhole_num_max,
      size_t pinhole_fixed_pose_num_max,
      size_t pinhole_fixed_point_num_max,
      size_t pinhole_fixed_pose_fixed_point_num_max,
      size_t pinhole_log_depth_num_max,
      size_t pinhole_log_depth_fixed_pose_num_max,
      size_t pinhole_log_depth_fixed_scale_num_max,
      size_t pinhole_log_depth_fixed_point_num_max,
      size_t pinhole_log_depth_fixed_pose_fixed_scale_num_max,
      size_t pinhole_log_depth_fixed_pose_fixed_point_num_max,
      size_t pinhole_log_depth_fixed_scale_fixed_point_num_max,
      size_t pinhole_fixed_rotation_num_max,
      size_t pinhole_fixed_rotation_fixed_calib_num_max,
      size_t pinhole_fixed_rotation_fixed_point_num_max,
      size_t pinhole_fixed_rotation_fixed_calib_fixed_point_num_max,
      size_t pinhole_log_depth_fixed_rotation_num_max,
      size_t pinhole_log_depth_fixed_rotation_fixed_scale_num_max,
      size_t pinhole_log_depth_fixed_rotation_fixed_point_num_max,
      size_t pinhole_log_depth_fixed_rotation_fixed_scale_fixed_point_num_max,
      size_t pinhole_intrinsics_prior_num_max,
      size_t pinhole_intrinsics_random_walk_num_max,
      size_t scale_prior_num_max,
      size_t simple_radial_split_fixed_focal_and_distortion_num_max,
      size_t simple_radial_split_fixed_principal_point_num_max,
      size_t simple_radial_split_fixed_pose_fixed_focal_and_distortion_num_max,
      size_t simple_radial_split_fixed_pose_fixed_principal_point_num_max,
      size_t
          simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_num_max,
      size_t simple_radial_split_fixed_focal_and_distortion_fixed_point_num_max,
      size_t simple_radial_split_fixed_principal_point_fixed_point_num_max,
      size_t
          simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_principal_point_num_max,
      size_t
          simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_point_num_max,
      size_t
          simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_max,
      size_t
          simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_fixed_point_num_max,
      size_t pinhole_split_fixed_focal_num_max,
      size_t pinhole_split_fixed_principal_point_num_max,
      size_t pinhole_split_fixed_pose_fixed_focal_num_max,
      size_t pinhole_split_fixed_pose_fixed_principal_point_num_max,
      size_t pinhole_split_fixed_focal_fixed_principal_point_num_max,
      size_t pinhole_split_fixed_focal_fixed_point_num_max,
      size_t pinhole_split_fixed_principal_point_fixed_point_num_max,
      size_t pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_max,
      size_t pinhole_split_fixed_pose_fixed_focal_fixed_point_num_max,
      size_t pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_max,
      size_t
          pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_max,
      size_t pinhole_split_fixed_rotation_fixed_focal_num_max,
      size_t pinhole_split_fixed_rotation_fixed_principal_point_num_max,
      size_t
          pinhole_split_fixed_rotation_fixed_focal_fixed_principal_point_num_max,
      size_t pinhole_split_fixed_rotation_fixed_focal_fixed_point_num_max,
      size_t
          pinhole_split_fixed_rotation_fixed_principal_point_fixed_point_num_max,
      size_t
          pinhole_split_fixed_rotation_fixed_focal_fixed_principal_point_fixed_point_num_max,
      size_t pinhole_split_intrinsics_prior_fixed_focal_num_max,
      size_t pinhole_split_intrinsics_prior_fixed_principal_point_num_max,
      size_t pinhole_split_intrinsics_random_walk_fixed_prev_focal_num_max,
      size_t
          pinhole_split_intrinsics_random_walk_fixed_prev_principal_point_num_max,
      size_t pinhole_split_intrinsics_random_walk_fixed_next_focal_num_max,
      size_t
          pinhole_split_intrinsics_random_walk_fixed_next_principal_point_num_max,
      size_t
          pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_prev_principal_point_num_max,
      size_t
          pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_next_focal_num_max,
      size_t
          pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_next_principal_point_num_max,
      size_t
          pinhole_split_intrinsics_random_walk_fixed_prev_principal_point_fixed_next_focal_num_max,
      size_t
          pinhole_split_intrinsics_random_walk_fixed_prev_principal_point_fixed_next_principal_point_num_max,
      size_t
          pinhole_split_intrinsics_random_walk_fixed_next_focal_fixed_next_principal_point_num_max,
      size_t
          pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_prev_principal_point_fixed_next_focal_num_max,
      size_t
          pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_prev_principal_point_fixed_next_principal_point_num_max,
      size_t
          pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_next_focal_fixed_next_principal_point_num_max,
      size_t
          pinhole_split_intrinsics_random_walk_fixed_prev_principal_point_fixed_next_focal_fixed_next_principal_point_num_max);

  // This class is managing cuda memory and cannot be copied.
  GraphSolver(const GraphSolver&) = delete;
  GraphSolver& operator=(const GraphSolver&) = delete;

  GraphSolver(GraphSolver&&) = default;
  GraphSolver& operator=(GraphSolver&&) = default;

  ~GraphSolver();

  /**
   * Set the solver parameters.
   */
  void set_params(const SolverParams<double>& params);

  /**
   * Run the solver.
   */
  SolveResult solve(bool print_progress = false, bool verbose_logging = false);

  /**
   * Finish the indices.
   *
   * This function has to be called after all indices are set and before the
   * solve function is called.
   */
  void finish_indices();

  /**
   * Get the number of allocated bytes.
   */
  size_t get_allocation_size();

  /**
   * Set the current value for the DepthScale nodes from the stacked host data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetDepthScaleNodesFromStackedHost(const double* const data,
                                         size_t offset,
                                         size_t num);

  /**
   * Set the current value for the DepthScale nodes from the stacked device
   * data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetDepthScaleNodesFromStackedDevice(const double* const data,
                                           size_t offset,
                                           size_t num);

  /**
   * Read the current value for the DepthScale nodes into the stacked output
   * host data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetDepthScaleNodesToStackedHost(double* const data,
                                       size_t offset,
                                       size_t num);

  /**
   * Read the current value for the DepthScale nodes into the stacked output
   * device data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetDepthScaleNodesToStackedDevice(double* const data,
                                         size_t offset,
                                         size_t num);

  /**
   * Set the current number of active nodes of type DepthScale.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetDepthScaleNum(size_t num);

  /**
   * Set the current value for the PinholeCalib nodes from the stacked host
   * data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetPinholeCalibNodesFromStackedHost(const double* const data,
                                           size_t offset,
                                           size_t num);

  /**
   * Set the current value for the PinholeCalib nodes from the stacked device
   * data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetPinholeCalibNodesFromStackedDevice(const double* const data,
                                             size_t offset,
                                             size_t num);

  /**
   * Read the current value for the PinholeCalib nodes into the stacked output
   * host data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetPinholeCalibNodesToStackedHost(double* const data,
                                         size_t offset,
                                         size_t num);

  /**
   * Read the current value for the PinholeCalib nodes into the stacked output
   * device data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetPinholeCalibNodesToStackedDevice(double* const data,
                                           size_t offset,
                                           size_t num);

  /**
   * Set the current number of active nodes of type PinholeCalib.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeCalibNum(size_t num);

  /**
   * Set the current value for the PinholeFocal nodes from the stacked host
   * data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetPinholeFocalNodesFromStackedHost(const double* const data,
                                           size_t offset,
                                           size_t num);

  /**
   * Set the current value for the PinholeFocal nodes from the stacked device
   * data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetPinholeFocalNodesFromStackedDevice(const double* const data,
                                             size_t offset,
                                             size_t num);

  /**
   * Read the current value for the PinholeFocal nodes into the stacked output
   * host data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetPinholeFocalNodesToStackedHost(double* const data,
                                         size_t offset,
                                         size_t num);

  /**
   * Read the current value for the PinholeFocal nodes into the stacked output
   * device data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetPinholeFocalNodesToStackedDevice(double* const data,
                                           size_t offset,
                                           size_t num);

  /**
   * Set the current number of active nodes of type PinholeFocal.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeFocalNum(size_t num);

  /**
   * Set the current value for the PinholePose nodes from the stacked host data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetPinholePoseNodesFromStackedHost(const double* const data,
                                          size_t offset,
                                          size_t num);

  /**
   * Set the current value for the PinholePose nodes from the stacked device
   * data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetPinholePoseNodesFromStackedDevice(const double* const data,
                                            size_t offset,
                                            size_t num);

  /**
   * Read the current value for the PinholePose nodes into the stacked output
   * host data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetPinholePoseNodesToStackedHost(double* const data,
                                        size_t offset,
                                        size_t num);

  /**
   * Read the current value for the PinholePose nodes into the stacked output
   * device data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetPinholePoseNodesToStackedDevice(double* const data,
                                          size_t offset,
                                          size_t num);

  /**
   * Set the current number of active nodes of type PinholePose.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholePoseNum(size_t num);

  /**
   * Set the current value for the PinholePrincipalPoint nodes from the stacked
   * host data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetPinholePrincipalPointNodesFromStackedHost(const double* const data,
                                                    size_t offset,
                                                    size_t num);

  /**
   * Set the current value for the PinholePrincipalPoint nodes from the stacked
   * device data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetPinholePrincipalPointNodesFromStackedDevice(const double* const data,
                                                      size_t offset,
                                                      size_t num);

  /**
   * Read the current value for the PinholePrincipalPoint nodes into the stacked
   * output host data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetPinholePrincipalPointNodesToStackedHost(double* const data,
                                                  size_t offset,
                                                  size_t num);

  /**
   * Read the current value for the PinholePrincipalPoint nodes into the stacked
   * output device data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetPinholePrincipalPointNodesToStackedDevice(double* const data,
                                                    size_t offset,
                                                    size_t num);

  /**
   * Set the current number of active nodes of type PinholePrincipalPoint.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholePrincipalPointNum(size_t num);

  /**
   * Set the current value for the PinholeTranslation nodes from the stacked
   * host data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetPinholeTranslationNodesFromStackedHost(const double* const data,
                                                 size_t offset,
                                                 size_t num);

  /**
   * Set the current value for the PinholeTranslation nodes from the stacked
   * device data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetPinholeTranslationNodesFromStackedDevice(const double* const data,
                                                   size_t offset,
                                                   size_t num);

  /**
   * Read the current value for the PinholeTranslation nodes into the stacked
   * output host data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetPinholeTranslationNodesToStackedHost(double* const data,
                                               size_t offset,
                                               size_t num);

  /**
   * Read the current value for the PinholeTranslation nodes into the stacked
   * output device data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetPinholeTranslationNodesToStackedDevice(double* const data,
                                                 size_t offset,
                                                 size_t num);

  /**
   * Set the current number of active nodes of type PinholeTranslation.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeTranslationNum(size_t num);

  /**
   * Set the current value for the Point nodes from the stacked host data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetPointNodesFromStackedHost(const double* const data,
                                    size_t offset,
                                    size_t num);

  /**
   * Set the current value for the Point nodes from the stacked device data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetPointNodesFromStackedDevice(const double* const data,
                                      size_t offset,
                                      size_t num);

  /**
   * Read the current value for the Point nodes into the stacked output host
   * data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetPointNodesToStackedHost(double* const data,
                                  size_t offset,
                                  size_t num);

  /**
   * Read the current value for the Point nodes into the stacked output device
   * data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetPointNodesToStackedDevice(double* const data,
                                    size_t offset,
                                    size_t num);

  /**
   * Set the current number of active nodes of type Point.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPointNum(size_t num);

  /**
   * Set the current value for the SimpleRadialCalib nodes from the stacked host
   * data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetSimpleRadialCalibNodesFromStackedHost(const double* const data,
                                                size_t offset,
                                                size_t num);

  /**
   * Set the current value for the SimpleRadialCalib nodes from the stacked
   * device data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetSimpleRadialCalibNodesFromStackedDevice(const double* const data,
                                                  size_t offset,
                                                  size_t num);

  /**
   * Read the current value for the SimpleRadialCalib nodes into the stacked
   * output host data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetSimpleRadialCalibNodesToStackedHost(double* const data,
                                              size_t offset,
                                              size_t num);

  /**
   * Read the current value for the SimpleRadialCalib nodes into the stacked
   * output device data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetSimpleRadialCalibNodesToStackedDevice(double* const data,
                                                size_t offset,
                                                size_t num);

  /**
   * Set the current number of active nodes of type SimpleRadialCalib.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetSimpleRadialCalibNum(size_t num);

  /**
   * Set the current value for the SimpleRadialFocalAndDistortion nodes from the
   * stacked host data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetSimpleRadialFocalAndDistortionNodesFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current value for the SimpleRadialFocalAndDistortion nodes from the
   * stacked device data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetSimpleRadialFocalAndDistortionNodesFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Read the current value for the SimpleRadialFocalAndDistortion nodes into
   * the stacked output host data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetSimpleRadialFocalAndDistortionNodesToStackedHost(double* const data,
                                                           size_t offset,
                                                           size_t num);

  /**
   * Read the current value for the SimpleRadialFocalAndDistortion nodes into
   * the stacked output device data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetSimpleRadialFocalAndDistortionNodesToStackedDevice(double* const data,
                                                             size_t offset,
                                                             size_t num);

  /**
   * Set the current number of active nodes of type
   * SimpleRadialFocalAndDistortion.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetSimpleRadialFocalAndDistortionNum(size_t num);

  /**
   * Set the current value for the SimpleRadialPose nodes from the stacked host
   * data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetSimpleRadialPoseNodesFromStackedHost(const double* const data,
                                               size_t offset,
                                               size_t num);

  /**
   * Set the current value for the SimpleRadialPose nodes from the stacked
   * device data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetSimpleRadialPoseNodesFromStackedDevice(const double* const data,
                                                 size_t offset,
                                                 size_t num);

  /**
   * Read the current value for the SimpleRadialPose nodes into the stacked
   * output host data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetSimpleRadialPoseNodesToStackedHost(double* const data,
                                             size_t offset,
                                             size_t num);

  /**
   * Read the current value for the SimpleRadialPose nodes into the stacked
   * output device data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetSimpleRadialPoseNodesToStackedDevice(double* const data,
                                               size_t offset,
                                               size_t num);

  /**
   * Set the current number of active nodes of type SimpleRadialPose.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetSimpleRadialPoseNum(size_t num);

  /**
   * Set the current value for the SimpleRadialPrincipalPoint nodes from the
   * stacked host data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetSimpleRadialPrincipalPointNodesFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current value for the SimpleRadialPrincipalPoint nodes from the
   * stacked device data.
   *
   * The offset can be used to start writing at a specific index.
   */
  void SetSimpleRadialPrincipalPointNodesFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Read the current value for the SimpleRadialPrincipalPoint nodes into the
   * stacked output host data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetSimpleRadialPrincipalPointNodesToStackedHost(double* const data,
                                                       size_t offset,
                                                       size_t num);

  /**
   * Read the current value for the SimpleRadialPrincipalPoint nodes into the
   * stacked output device data.
   *
   * The offset can be used to start reading from a specific index.
   */
  void GetSimpleRadialPrincipalPointNodesToStackedDevice(double* const data,
                                                         size_t offset,
                                                         size_t num);

  /**
   * Set the current number of active nodes of type SimpleRadialPrincipalPoint.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetSimpleRadialPrincipalPointNum(size_t num);

  /**
   * Set the indices for the pose argument for the SimpleRadial factor from
   * host.
   */
  void SetSimpleRadialPoseIndicesFromHost(const unsigned int* const indices,
                                          size_t num);

  /**
   * Set the indices for the pose argument for the SimpleRadial factor from
   * device.
   */
  void SetSimpleRadialPoseIndicesFromDevice(const unsigned int* const indices,
                                            size_t num);

  /**
   * Set the indices for the calib argument for the SimpleRadial factor from
   * host.
   */
  void SetSimpleRadialCalibIndicesFromHost(const unsigned int* const indices,
                                           size_t num);

  /**
   * Set the indices for the calib argument for the SimpleRadial factor from
   * device.
   */
  void SetSimpleRadialCalibIndicesFromDevice(const unsigned int* const indices,
                                             size_t num);

  /**
   * Set the indices for the point argument for the SimpleRadial factor from
   * host.
   */
  void SetSimpleRadialPointIndicesFromHost(const unsigned int* const indices,
                                           size_t num);

  /**
   * Set the indices for the point argument for the SimpleRadial factor from
   * device.
   */
  void SetSimpleRadialPointIndicesFromDevice(const unsigned int* const indices,
                                             size_t num);

  /**
   * Set the values for the pixel consts SimpleRadial factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialPixelDataFromStackedHost(const double* const data,
                                               size_t offset,
                                               size_t num);

  /**
   * Set the values for the pixel consts SimpleRadial factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialPixelDataFromStackedDevice(const double* const data,
                                                 size_t offset,
                                                 size_t num);

  /**
   * Set the current number of SimpleRadial factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetSimpleRadialNum(size_t num);

  /**
   * Set the indices for the calib argument for the SimpleRadialFixedPose factor
   * from host.
   */
  void SetSimpleRadialFixedPoseCalibIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the calib argument for the SimpleRadialFixedPose factor
   * from device.
   */
  void SetSimpleRadialFixedPoseCalibIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the SimpleRadialFixedPose factor
   * from host.
   */
  void SetSimpleRadialFixedPosePointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the SimpleRadialFixedPose factor
   * from device.
   */
  void SetSimpleRadialFixedPosePointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts SimpleRadialFixedPose factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPosePixelDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts SimpleRadialFixedPose factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPosePixelDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts SimpleRadialFixedPose factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPosePoseDataFromStackedHost(const double* const data,
                                                       size_t offset,
                                                       size_t num);

  /**
   * Set the values for the pose consts SimpleRadialFixedPose factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPosePoseDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of SimpleRadialFixedPose factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetSimpleRadialFixedPoseNum(size_t num);

  /**
   * Set the indices for the pose argument for the SimpleRadialFixedPoint factor
   * from host.
   */
  void SetSimpleRadialFixedPointPoseIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the SimpleRadialFixedPoint factor
   * from device.
   */
  void SetSimpleRadialFixedPointPoseIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the calib argument for the SimpleRadialFixedPoint
   * factor from host.
   */
  void SetSimpleRadialFixedPointCalibIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the calib argument for the SimpleRadialFixedPoint
   * factor from device.
   */
  void SetSimpleRadialFixedPointCalibIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts SimpleRadialFixedPoint factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPointPixelDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts SimpleRadialFixedPoint factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPointPixelDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts SimpleRadialFixedPoint factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPointPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts SimpleRadialFixedPoint factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPointPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of SimpleRadialFixedPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetSimpleRadialFixedPointNum(size_t num);

  /**
   * Set the indices for the calib argument for the
   * SimpleRadialFixedPoseFixedPoint factor from host.
   */
  void SetSimpleRadialFixedPoseFixedPointCalibIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the calib argument for the
   * SimpleRadialFixedPoseFixedPoint factor from device.
   */
  void SetSimpleRadialFixedPoseFixedPointCalibIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts SimpleRadialFixedPoseFixedPoint factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPoseFixedPointPixelDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts SimpleRadialFixedPoseFixedPoint factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPoseFixedPointPixelDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts SimpleRadialFixedPoseFixedPoint factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPoseFixedPointPoseDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts SimpleRadialFixedPoseFixedPoint factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPoseFixedPointPoseDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts SimpleRadialFixedPoseFixedPoint factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPoseFixedPointPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts SimpleRadialFixedPoseFixedPoint factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialFixedPoseFixedPointPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of SimpleRadialFixedPoseFixedPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetSimpleRadialFixedPoseFixedPointNum(size_t num);

  /**
   * Set the indices for the pose argument for the Pinhole factor from host.
   */
  void SetPinholePoseIndicesFromHost(const unsigned int* const indices,
                                     size_t num);

  /**
   * Set the indices for the pose argument for the Pinhole factor from device.
   */
  void SetPinholePoseIndicesFromDevice(const unsigned int* const indices,
                                       size_t num);

  /**
   * Set the indices for the calib argument for the Pinhole factor from host.
   */
  void SetPinholeCalibIndicesFromHost(const unsigned int* const indices,
                                      size_t num);

  /**
   * Set the indices for the calib argument for the Pinhole factor from device.
   */
  void SetPinholeCalibIndicesFromDevice(const unsigned int* const indices,
                                        size_t num);

  /**
   * Set the indices for the point argument for the Pinhole factor from host.
   */
  void SetPinholePointIndicesFromHost(const unsigned int* const indices,
                                      size_t num);

  /**
   * Set the indices for the point argument for the Pinhole factor from device.
   */
  void SetPinholePointIndicesFromDevice(const unsigned int* const indices,
                                        size_t num);

  /**
   * Set the values for the pixel consts Pinhole factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholePixelDataFromStackedHost(const double* const data,
                                          size_t offset,
                                          size_t num);

  /**
   * Set the values for the pixel consts Pinhole factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholePixelDataFromStackedDevice(const double* const data,
                                            size_t offset,
                                            size_t num);

  /**
   * Set the values for the weight_loss consts Pinhole factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeWeightLossDataFromStackedHost(const double* const data,
                                               size_t offset,
                                               size_t num);

  /**
   * Set the values for the weight_loss consts Pinhole factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeWeightLossDataFromStackedDevice(const double* const data,
                                                 size_t offset,
                                                 size_t num);

  /**
   * Set the current number of Pinhole factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeNum(size_t num);

  /**
   * Set the indices for the calib argument for the PinholeFixedPose factor from
   * host.
   */
  void SetPinholeFixedPoseCalibIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the calib argument for the PinholeFixedPose factor from
   * device.
   */
  void SetPinholeFixedPoseCalibIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the PinholeFixedPose factor from
   * host.
   */
  void SetPinholeFixedPosePointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the PinholeFixedPose factor from
   * device.
   */
  void SetPinholeFixedPosePointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts PinholeFixedPose factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPosePixelDataFromStackedHost(const double* const data,
                                                   size_t offset,
                                                   size_t num);

  /**
   * Set the values for the pixel consts PinholeFixedPose factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPosePixelDataFromStackedDevice(const double* const data,
                                                     size_t offset,
                                                     size_t num);

  /**
   * Set the values for the weight_loss consts PinholeFixedPose factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPoseWeightLossDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the weight_loss consts PinholeFixedPose factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPoseWeightLossDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts PinholeFixedPose factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPosePoseDataFromStackedHost(const double* const data,
                                                  size_t offset,
                                                  size_t num);

  /**
   * Set the values for the pose consts PinholeFixedPose factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPosePoseDataFromStackedDevice(const double* const data,
                                                    size_t offset,
                                                    size_t num);

  /**
   * Set the current number of PinholeFixedPose factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeFixedPoseNum(size_t num);

  /**
   * Set the indices for the pose argument for the PinholeFixedPoint factor from
   * host.
   */
  void SetPinholeFixedPointPoseIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the PinholeFixedPoint factor from
   * device.
   */
  void SetPinholeFixedPointPoseIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the calib argument for the PinholeFixedPoint factor
   * from host.
   */
  void SetPinholeFixedPointCalibIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the calib argument for the PinholeFixedPoint factor
   * from device.
   */
  void SetPinholeFixedPointCalibIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts PinholeFixedPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPointPixelDataFromStackedHost(const double* const data,
                                                    size_t offset,
                                                    size_t num);

  /**
   * Set the values for the pixel consts PinholeFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPointPixelDataFromStackedDevice(const double* const data,
                                                      size_t offset,
                                                      size_t num);

  /**
   * Set the values for the weight_loss consts PinholeFixedPoint factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPointWeightLossDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the weight_loss consts PinholeFixedPoint factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPointWeightLossDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts PinholeFixedPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPointPointDataFromStackedHost(const double* const data,
                                                    size_t offset,
                                                    size_t num);

  /**
   * Set the values for the point consts PinholeFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPointPointDataFromStackedDevice(const double* const data,
                                                      size_t offset,
                                                      size_t num);

  /**
   * Set the current number of PinholeFixedPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeFixedPointNum(size_t num);

  /**
   * Set the indices for the calib argument for the PinholeFixedPoseFixedPoint
   * factor from host.
   */
  void SetPinholeFixedPoseFixedPointCalibIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the calib argument for the PinholeFixedPoseFixedPoint
   * factor from device.
   */
  void SetPinholeFixedPoseFixedPointCalibIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts PinholeFixedPoseFixedPoint factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPoseFixedPointPixelDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts PinholeFixedPoseFixedPoint factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPoseFixedPointPixelDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the weight_loss consts PinholeFixedPoseFixedPoint factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPoseFixedPointWeightLossDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the weight_loss consts PinholeFixedPoseFixedPoint factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPoseFixedPointWeightLossDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts PinholeFixedPoseFixedPoint factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPoseFixedPointPoseDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts PinholeFixedPoseFixedPoint factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPoseFixedPointPoseDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts PinholeFixedPoseFixedPoint factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPoseFixedPointPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts PinholeFixedPoseFixedPoint factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedPoseFixedPointPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of PinholeFixedPoseFixedPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeFixedPoseFixedPointNum(size_t num);

  /**
   * Set the indices for the pose argument for the PinholeLogDepth factor from
   * host.
   */
  void SetPinholeLogDepthPoseIndicesFromHost(const unsigned int* const indices,
                                             size_t num);

  /**
   * Set the indices for the pose argument for the PinholeLogDepth factor from
   * device.
   */
  void SetPinholeLogDepthPoseIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the scale argument for the PinholeLogDepth factor from
   * host.
   */
  void SetPinholeLogDepthScaleIndicesFromHost(const unsigned int* const indices,
                                              size_t num);

  /**
   * Set the indices for the scale argument for the PinholeLogDepth factor from
   * device.
   */
  void SetPinholeLogDepthScaleIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the PinholeLogDepth factor from
   * host.
   */
  void SetPinholeLogDepthPointIndicesFromHost(const unsigned int* const indices,
                                              size_t num);

  /**
   * Set the indices for the point argument for the PinholeLogDepth factor from
   * device.
   */
  void SetPinholeLogDepthPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the log_depth consts PinholeLogDepth factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthLogDepthDataFromStackedHost(const double* const data,
                                                     size_t offset,
                                                     size_t num);

  /**
   * Set the values for the log_depth consts PinholeLogDepth factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthLogDepthDataFromStackedDevice(const double* const data,
                                                       size_t offset,
                                                       size_t num);

  /**
   * Set the values for the loss consts PinholeLogDepth factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthLossDataFromStackedHost(const double* const data,
                                                 size_t offset,
                                                 size_t num);

  /**
   * Set the values for the loss consts PinholeLogDepth factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthLossDataFromStackedDevice(const double* const data,
                                                   size_t offset,
                                                   size_t num);

  /**
   * Set the current number of PinholeLogDepth factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeLogDepthNum(size_t num);

  /**
   * Set the indices for the scale argument for the PinholeLogDepthFixedPose
   * factor from host.
   */
  void SetPinholeLogDepthFixedPoseScaleIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the scale argument for the PinholeLogDepthFixedPose
   * factor from device.
   */
  void SetPinholeLogDepthFixedPoseScaleIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the PinholeLogDepthFixedPose
   * factor from host.
   */
  void SetPinholeLogDepthFixedPosePointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the PinholeLogDepthFixedPose
   * factor from device.
   */
  void SetPinholeLogDepthFixedPosePointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the log_depth consts PinholeLogDepthFixedPose factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthFixedPoseLogDepthDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the log_depth consts PinholeLogDepthFixedPose factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthFixedPoseLogDepthDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the loss consts PinholeLogDepthFixedPose factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthFixedPoseLossDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the loss consts PinholeLogDepthFixedPose factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthFixedPoseLossDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts PinholeLogDepthFixedPose factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthFixedPosePoseDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts PinholeLogDepthFixedPose factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthFixedPosePoseDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of PinholeLogDepthFixedPose factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeLogDepthFixedPoseNum(size_t num);

  /**
   * Set the indices for the pose argument for the PinholeLogDepthFixedScale
   * factor from host.
   */
  void SetPinholeLogDepthFixedScalePoseIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the PinholeLogDepthFixedScale
   * factor from device.
   */
  void SetPinholeLogDepthFixedScalePoseIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the PinholeLogDepthFixedScale
   * factor from host.
   */
  void SetPinholeLogDepthFixedScalePointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the PinholeLogDepthFixedScale
   * factor from device.
   */
  void SetPinholeLogDepthFixedScalePointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the log_depth consts PinholeLogDepthFixedScale factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthFixedScaleLogDepthDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the log_depth consts PinholeLogDepthFixedScale factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthFixedScaleLogDepthDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the loss consts PinholeLogDepthFixedScale factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthFixedScaleLossDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the loss consts PinholeLogDepthFixedScale factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthFixedScaleLossDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the scale consts PinholeLogDepthFixedScale factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthFixedScaleScaleDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the scale consts PinholeLogDepthFixedScale factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthFixedScaleScaleDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of PinholeLogDepthFixedScale factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeLogDepthFixedScaleNum(size_t num);

  /**
   * Set the indices for the pose argument for the PinholeLogDepthFixedPoint
   * factor from host.
   */
  void SetPinholeLogDepthFixedPointPoseIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the PinholeLogDepthFixedPoint
   * factor from device.
   */
  void SetPinholeLogDepthFixedPointPoseIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the scale argument for the PinholeLogDepthFixedPoint
   * factor from host.
   */
  void SetPinholeLogDepthFixedPointScaleIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the scale argument for the PinholeLogDepthFixedPoint
   * factor from device.
   */
  void SetPinholeLogDepthFixedPointScaleIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the log_depth consts PinholeLogDepthFixedPoint factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthFixedPointLogDepthDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the log_depth consts PinholeLogDepthFixedPoint factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthFixedPointLogDepthDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the loss consts PinholeLogDepthFixedPoint factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthFixedPointLossDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the loss consts PinholeLogDepthFixedPoint factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthFixedPointLossDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts PinholeLogDepthFixedPoint factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthFixedPointPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts PinholeLogDepthFixedPoint factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthFixedPointPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of PinholeLogDepthFixedPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeLogDepthFixedPointNum(size_t num);

  /**
   * Set the indices for the point argument for the
   * PinholeLogDepthFixedPoseFixedScale factor from host.
   */
  void SetPinholeLogDepthFixedPoseFixedScalePointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * PinholeLogDepthFixedPoseFixedScale factor from device.
   */
  void SetPinholeLogDepthFixedPoseFixedScalePointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the log_depth consts PinholeLogDepthFixedPoseFixedScale
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthFixedPoseFixedScaleLogDepthDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the log_depth consts PinholeLogDepthFixedPoseFixedScale
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthFixedPoseFixedScaleLogDepthDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the loss consts PinholeLogDepthFixedPoseFixedScale
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthFixedPoseFixedScaleLossDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the loss consts PinholeLogDepthFixedPoseFixedScale
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthFixedPoseFixedScaleLossDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts PinholeLogDepthFixedPoseFixedScale
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthFixedPoseFixedScalePoseDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts PinholeLogDepthFixedPoseFixedScale
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthFixedPoseFixedScalePoseDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the scale consts PinholeLogDepthFixedPoseFixedScale
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthFixedPoseFixedScaleScaleDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the scale consts PinholeLogDepthFixedPoseFixedScale
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthFixedPoseFixedScaleScaleDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of PinholeLogDepthFixedPoseFixedScale factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeLogDepthFixedPoseFixedScaleNum(size_t num);

  /**
   * Set the indices for the scale argument for the
   * PinholeLogDepthFixedPoseFixedPoint factor from host.
   */
  void SetPinholeLogDepthFixedPoseFixedPointScaleIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the scale argument for the
   * PinholeLogDepthFixedPoseFixedPoint factor from device.
   */
  void SetPinholeLogDepthFixedPoseFixedPointScaleIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the log_depth consts PinholeLogDepthFixedPoseFixedPoint
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthFixedPoseFixedPointLogDepthDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the log_depth consts PinholeLogDepthFixedPoseFixedPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthFixedPoseFixedPointLogDepthDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the loss consts PinholeLogDepthFixedPoseFixedPoint
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthFixedPoseFixedPointLossDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the loss consts PinholeLogDepthFixedPoseFixedPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthFixedPoseFixedPointLossDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts PinholeLogDepthFixedPoseFixedPoint
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthFixedPoseFixedPointPoseDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts PinholeLogDepthFixedPoseFixedPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthFixedPoseFixedPointPoseDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts PinholeLogDepthFixedPoseFixedPoint
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthFixedPoseFixedPointPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts PinholeLogDepthFixedPoseFixedPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthFixedPoseFixedPointPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of PinholeLogDepthFixedPoseFixedPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeLogDepthFixedPoseFixedPointNum(size_t num);

  /**
   * Set the indices for the pose argument for the
   * PinholeLogDepthFixedScaleFixedPoint factor from host.
   */
  void SetPinholeLogDepthFixedScaleFixedPointPoseIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the
   * PinholeLogDepthFixedScaleFixedPoint factor from device.
   */
  void SetPinholeLogDepthFixedScaleFixedPointPoseIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the log_depth consts PinholeLogDepthFixedScaleFixedPoint
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthFixedScaleFixedPointLogDepthDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the log_depth consts PinholeLogDepthFixedScaleFixedPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthFixedScaleFixedPointLogDepthDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the loss consts PinholeLogDepthFixedScaleFixedPoint
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthFixedScaleFixedPointLossDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the loss consts PinholeLogDepthFixedScaleFixedPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthFixedScaleFixedPointLossDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the scale consts PinholeLogDepthFixedScaleFixedPoint
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthFixedScaleFixedPointScaleDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the scale consts PinholeLogDepthFixedScaleFixedPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthFixedScaleFixedPointScaleDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts PinholeLogDepthFixedScaleFixedPoint
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthFixedScaleFixedPointPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts PinholeLogDepthFixedScaleFixedPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthFixedScaleFixedPointPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of PinholeLogDepthFixedScaleFixedPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeLogDepthFixedScaleFixedPointNum(size_t num);

  /**
   * Set the indices for the translation argument for the PinholeFixedRotation
   * factor from host.
   */
  void SetPinholeFixedRotationTranslationIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the translation argument for the PinholeFixedRotation
   * factor from device.
   */
  void SetPinholeFixedRotationTranslationIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the calib argument for the PinholeFixedRotation factor
   * from host.
   */
  void SetPinholeFixedRotationCalibIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the calib argument for the PinholeFixedRotation factor
   * from device.
   */
  void SetPinholeFixedRotationCalibIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the PinholeFixedRotation factor
   * from host.
   */
  void SetPinholeFixedRotationPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the PinholeFixedRotation factor
   * from device.
   */
  void SetPinholeFixedRotationPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the rotation consts PinholeFixedRotation factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedRotationRotationDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the rotation consts PinholeFixedRotation factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedRotationRotationDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts PinholeFixedRotation factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedRotationPixelDataFromStackedHost(const double* const data,
                                                       size_t offset,
                                                       size_t num);

  /**
   * Set the values for the pixel consts PinholeFixedRotation factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedRotationPixelDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the weight_loss consts PinholeFixedRotation factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedRotationWeightLossDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the weight_loss consts PinholeFixedRotation factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedRotationWeightLossDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of PinholeFixedRotation factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeFixedRotationNum(size_t num);

  /**
   * Set the indices for the translation argument for the
   * PinholeFixedRotationFixedCalib factor from host.
   */
  void SetPinholeFixedRotationFixedCalibTranslationIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the translation argument for the
   * PinholeFixedRotationFixedCalib factor from device.
   */
  void SetPinholeFixedRotationFixedCalibTranslationIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * PinholeFixedRotationFixedCalib factor from host.
   */
  void SetPinholeFixedRotationFixedCalibPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * PinholeFixedRotationFixedCalib factor from device.
   */
  void SetPinholeFixedRotationFixedCalibPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the rotation consts PinholeFixedRotationFixedCalib
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedRotationFixedCalibRotationDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the rotation consts PinholeFixedRotationFixedCalib
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedRotationFixedCalibRotationDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts PinholeFixedRotationFixedCalib factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedRotationFixedCalibPixelDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts PinholeFixedRotationFixedCalib factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedRotationFixedCalibPixelDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the weight_loss consts PinholeFixedRotationFixedCalib
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedRotationFixedCalibWeightLossDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the weight_loss consts PinholeFixedRotationFixedCalib
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedRotationFixedCalibWeightLossDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the calib consts PinholeFixedRotationFixedCalib factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedRotationFixedCalibCalibDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the calib consts PinholeFixedRotationFixedCalib factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedRotationFixedCalibCalibDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of PinholeFixedRotationFixedCalib factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeFixedRotationFixedCalibNum(size_t num);

  /**
   * Set the indices for the translation argument for the
   * PinholeFixedRotationFixedPoint factor from host.
   */
  void SetPinholeFixedRotationFixedPointTranslationIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the translation argument for the
   * PinholeFixedRotationFixedPoint factor from device.
   */
  void SetPinholeFixedRotationFixedPointTranslationIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the calib argument for the
   * PinholeFixedRotationFixedPoint factor from host.
   */
  void SetPinholeFixedRotationFixedPointCalibIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the calib argument for the
   * PinholeFixedRotationFixedPoint factor from device.
   */
  void SetPinholeFixedRotationFixedPointCalibIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the rotation consts PinholeFixedRotationFixedPoint
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedRotationFixedPointRotationDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the rotation consts PinholeFixedRotationFixedPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedRotationFixedPointRotationDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts PinholeFixedRotationFixedPoint factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedRotationFixedPointPixelDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts PinholeFixedRotationFixedPoint factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedRotationFixedPointPixelDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the weight_loss consts PinholeFixedRotationFixedPoint
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedRotationFixedPointWeightLossDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the weight_loss consts PinholeFixedRotationFixedPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedRotationFixedPointWeightLossDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts PinholeFixedRotationFixedPoint factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedRotationFixedPointPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts PinholeFixedRotationFixedPoint factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedRotationFixedPointPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of PinholeFixedRotationFixedPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeFixedRotationFixedPointNum(size_t num);

  /**
   * Set the indices for the translation argument for the
   * PinholeFixedRotationFixedCalibFixedPoint factor from host.
   */
  void SetPinholeFixedRotationFixedCalibFixedPointTranslationIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the translation argument for the
   * PinholeFixedRotationFixedCalibFixedPoint factor from device.
   */
  void SetPinholeFixedRotationFixedCalibFixedPointTranslationIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the rotation consts
   * PinholeFixedRotationFixedCalibFixedPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedRotationFixedCalibFixedPointRotationDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the rotation consts
   * PinholeFixedRotationFixedCalibFixedPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedRotationFixedCalibFixedPointRotationDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * PinholeFixedRotationFixedCalibFixedPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedRotationFixedCalibFixedPointPixelDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * PinholeFixedRotationFixedCalibFixedPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedRotationFixedCalibFixedPointPixelDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the weight_loss consts
   * PinholeFixedRotationFixedCalibFixedPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedRotationFixedCalibFixedPointWeightLossDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the weight_loss consts
   * PinholeFixedRotationFixedCalibFixedPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeFixedRotationFixedCalibFixedPointWeightLossDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the calib consts
   * PinholeFixedRotationFixedCalibFixedPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedRotationFixedCalibFixedPointCalibDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the calib consts
   * PinholeFixedRotationFixedCalibFixedPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedRotationFixedCalibFixedPointCalibDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * PinholeFixedRotationFixedCalibFixedPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedRotationFixedCalibFixedPointPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * PinholeFixedRotationFixedCalibFixedPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeFixedRotationFixedCalibFixedPointPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of PinholeFixedRotationFixedCalibFixedPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeFixedRotationFixedCalibFixedPointNum(size_t num);

  /**
   * Set the indices for the translation argument for the
   * PinholeLogDepthFixedRotation factor from host.
   */
  void SetPinholeLogDepthFixedRotationTranslationIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the translation argument for the
   * PinholeLogDepthFixedRotation factor from device.
   */
  void SetPinholeLogDepthFixedRotationTranslationIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the scale argument for the PinholeLogDepthFixedRotation
   * factor from host.
   */
  void SetPinholeLogDepthFixedRotationScaleIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the scale argument for the PinholeLogDepthFixedRotation
   * factor from device.
   */
  void SetPinholeLogDepthFixedRotationScaleIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the PinholeLogDepthFixedRotation
   * factor from host.
   */
  void SetPinholeLogDepthFixedRotationPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the PinholeLogDepthFixedRotation
   * factor from device.
   */
  void SetPinholeLogDepthFixedRotationPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the rotation consts PinholeLogDepthFixedRotation factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthFixedRotationRotationDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the rotation consts PinholeLogDepthFixedRotation factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthFixedRotationRotationDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the log_depth consts PinholeLogDepthFixedRotation factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthFixedRotationLogDepthDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the log_depth consts PinholeLogDepthFixedRotation factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthFixedRotationLogDepthDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the loss consts PinholeLogDepthFixedRotation factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthFixedRotationLossDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the loss consts PinholeLogDepthFixedRotation factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthFixedRotationLossDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of PinholeLogDepthFixedRotation factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeLogDepthFixedRotationNum(size_t num);

  /**
   * Set the indices for the translation argument for the
   * PinholeLogDepthFixedRotationFixedScale factor from host.
   */
  void SetPinholeLogDepthFixedRotationFixedScaleTranslationIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the translation argument for the
   * PinholeLogDepthFixedRotationFixedScale factor from device.
   */
  void SetPinholeLogDepthFixedRotationFixedScaleTranslationIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * PinholeLogDepthFixedRotationFixedScale factor from host.
   */
  void SetPinholeLogDepthFixedRotationFixedScalePointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * PinholeLogDepthFixedRotationFixedScale factor from device.
   */
  void SetPinholeLogDepthFixedRotationFixedScalePointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the rotation consts
   * PinholeLogDepthFixedRotationFixedScale factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthFixedRotationFixedScaleRotationDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the rotation consts
   * PinholeLogDepthFixedRotationFixedScale factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthFixedRotationFixedScaleRotationDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the log_depth consts
   * PinholeLogDepthFixedRotationFixedScale factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthFixedRotationFixedScaleLogDepthDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the log_depth consts
   * PinholeLogDepthFixedRotationFixedScale factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthFixedRotationFixedScaleLogDepthDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the loss consts PinholeLogDepthFixedRotationFixedScale
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthFixedRotationFixedScaleLossDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the loss consts PinholeLogDepthFixedRotationFixedScale
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthFixedRotationFixedScaleLossDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the scale consts PinholeLogDepthFixedRotationFixedScale
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthFixedRotationFixedScaleScaleDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the scale consts PinholeLogDepthFixedRotationFixedScale
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthFixedRotationFixedScaleScaleDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of PinholeLogDepthFixedRotationFixedScale factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeLogDepthFixedRotationFixedScaleNum(size_t num);

  /**
   * Set the indices for the translation argument for the
   * PinholeLogDepthFixedRotationFixedPoint factor from host.
   */
  void SetPinholeLogDepthFixedRotationFixedPointTranslationIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the translation argument for the
   * PinholeLogDepthFixedRotationFixedPoint factor from device.
   */
  void SetPinholeLogDepthFixedRotationFixedPointTranslationIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the scale argument for the
   * PinholeLogDepthFixedRotationFixedPoint factor from host.
   */
  void SetPinholeLogDepthFixedRotationFixedPointScaleIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the scale argument for the
   * PinholeLogDepthFixedRotationFixedPoint factor from device.
   */
  void SetPinholeLogDepthFixedRotationFixedPointScaleIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the rotation consts
   * PinholeLogDepthFixedRotationFixedPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthFixedRotationFixedPointRotationDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the rotation consts
   * PinholeLogDepthFixedRotationFixedPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthFixedRotationFixedPointRotationDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the log_depth consts
   * PinholeLogDepthFixedRotationFixedPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthFixedRotationFixedPointLogDepthDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the log_depth consts
   * PinholeLogDepthFixedRotationFixedPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthFixedRotationFixedPointLogDepthDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the loss consts PinholeLogDepthFixedRotationFixedPoint
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthFixedRotationFixedPointLossDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the loss consts PinholeLogDepthFixedRotationFixedPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthFixedRotationFixedPointLossDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts PinholeLogDepthFixedRotationFixedPoint
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthFixedRotationFixedPointPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts PinholeLogDepthFixedRotationFixedPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeLogDepthFixedRotationFixedPointPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of PinholeLogDepthFixedRotationFixedPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeLogDepthFixedRotationFixedPointNum(size_t num);

  /**
   * Set the indices for the translation argument for the
   * PinholeLogDepthFixedRotationFixedScaleFixedPoint factor from host.
   */
  void
  SetPinholeLogDepthFixedRotationFixedScaleFixedPointTranslationIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the translation argument for the
   * PinholeLogDepthFixedRotationFixedScaleFixedPoint factor from device.
   */
  void
  SetPinholeLogDepthFixedRotationFixedScaleFixedPointTranslationIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the rotation consts
   * PinholeLogDepthFixedRotationFixedScaleFixedPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeLogDepthFixedRotationFixedScaleFixedPointRotationDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the rotation consts
   * PinholeLogDepthFixedRotationFixedScaleFixedPoint factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeLogDepthFixedRotationFixedScaleFixedPointRotationDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the log_depth consts
   * PinholeLogDepthFixedRotationFixedScaleFixedPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeLogDepthFixedRotationFixedScaleFixedPointLogDepthDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the log_depth consts
   * PinholeLogDepthFixedRotationFixedScaleFixedPoint factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeLogDepthFixedRotationFixedScaleFixedPointLogDepthDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the loss consts
   * PinholeLogDepthFixedRotationFixedScaleFixedPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeLogDepthFixedRotationFixedScaleFixedPointLossDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the loss consts
   * PinholeLogDepthFixedRotationFixedScaleFixedPoint factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeLogDepthFixedRotationFixedScaleFixedPointLossDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the scale consts
   * PinholeLogDepthFixedRotationFixedScaleFixedPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeLogDepthFixedRotationFixedScaleFixedPointScaleDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the scale consts
   * PinholeLogDepthFixedRotationFixedScaleFixedPoint factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeLogDepthFixedRotationFixedScaleFixedPointScaleDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * PinholeLogDepthFixedRotationFixedScaleFixedPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeLogDepthFixedRotationFixedScaleFixedPointPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * PinholeLogDepthFixedRotationFixedScaleFixedPoint factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeLogDepthFixedRotationFixedScaleFixedPointPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of PinholeLogDepthFixedRotationFixedScaleFixedPoint
   * factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeLogDepthFixedRotationFixedScaleFixedPointNum(size_t num);

  /**
   * Set the indices for the calib argument for the PinholeIntrinsicsPrior
   * factor from host.
   */
  void SetPinholeIntrinsicsPriorCalibIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the calib argument for the PinholeIntrinsicsPrior
   * factor from device.
   */
  void SetPinholeIntrinsicsPriorCalibIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the prior consts PinholeIntrinsicsPrior factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeIntrinsicsPriorPriorDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the prior consts PinholeIntrinsicsPrior factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeIntrinsicsPriorPriorDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the inv_std consts PinholeIntrinsicsPrior factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeIntrinsicsPriorInvStdDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the inv_std consts PinholeIntrinsicsPrior factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeIntrinsicsPriorInvStdDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of PinholeIntrinsicsPrior factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeIntrinsicsPriorNum(size_t num);

  /**
   * Set the indices for the prev_calib argument for the
   * PinholeIntrinsicsRandomWalk factor from host.
   */
  void SetPinholeIntrinsicsRandomWalkPrevCalibIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the prev_calib argument for the
   * PinholeIntrinsicsRandomWalk factor from device.
   */
  void SetPinholeIntrinsicsRandomWalkPrevCalibIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the next_calib argument for the
   * PinholeIntrinsicsRandomWalk factor from host.
   */
  void SetPinholeIntrinsicsRandomWalkNextCalibIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the next_calib argument for the
   * PinholeIntrinsicsRandomWalk factor from device.
   */
  void SetPinholeIntrinsicsRandomWalkNextCalibIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the inv_std consts PinholeIntrinsicsRandomWalk factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeIntrinsicsRandomWalkInvStdDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the inv_std consts PinholeIntrinsicsRandomWalk factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeIntrinsicsRandomWalkInvStdDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of PinholeIntrinsicsRandomWalk factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeIntrinsicsRandomWalkNum(size_t num);

  /**
   * Set the indices for the scale argument for the ScalePrior factor from host.
   */
  void SetScalePriorScaleIndicesFromHost(const unsigned int* const indices,
                                         size_t num);

  /**
   * Set the indices for the scale argument for the ScalePrior factor from
   * device.
   */
  void SetScalePriorScaleIndicesFromDevice(const unsigned int* const indices,
                                           size_t num);

  /**
   * Set the values for the inv_std consts ScalePrior factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetScalePriorInvStdDataFromStackedHost(const double* const data,
                                              size_t offset,
                                              size_t num);

  /**
   * Set the values for the inv_std consts ScalePrior factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetScalePriorInvStdDataFromStackedDevice(const double* const data,
                                                size_t offset,
                                                size_t num);

  /**
   * Set the values for the loss consts ScalePrior factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetScalePriorLossDataFromStackedHost(const double* const data,
                                            size_t offset,
                                            size_t num);

  /**
   * Set the values for the loss consts ScalePrior factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetScalePriorLossDataFromStackedDevice(const double* const data,
                                              size_t offset,
                                              size_t num);

  /**
   * Set the current number of ScalePrior factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetScalePriorNum(size_t num);

  /**
   * Set the indices for the pose argument for the
   * SimpleRadialSplitFixedFocalAndDistortion factor from host.
   */
  void SetSimpleRadialSplitFixedFocalAndDistortionPoseIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the
   * SimpleRadialSplitFixedFocalAndDistortion factor from device.
   */
  void SetSimpleRadialSplitFixedFocalAndDistortionPoseIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the principal_point argument for the
   * SimpleRadialSplitFixedFocalAndDistortion factor from host.
   */
  void SetSimpleRadialSplitFixedFocalAndDistortionPrincipalPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the principal_point argument for the
   * SimpleRadialSplitFixedFocalAndDistortion factor from device.
   */
  void
  SetSimpleRadialSplitFixedFocalAndDistortionPrincipalPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * SimpleRadialSplitFixedFocalAndDistortion factor from host.
   */
  void SetSimpleRadialSplitFixedFocalAndDistortionPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * SimpleRadialSplitFixedFocalAndDistortion factor from device.
   */
  void SetSimpleRadialSplitFixedFocalAndDistortionPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialSplitFixedFocalAndDistortion factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialSplitFixedFocalAndDistortionPixelDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialSplitFixedFocalAndDistortion factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialSplitFixedFocalAndDistortionPixelDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_distortion consts
   * SimpleRadialSplitFixedFocalAndDistortion factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndDistortionFocalAndDistortionDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_distortion consts
   * SimpleRadialSplitFixedFocalAndDistortion factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndDistortionFocalAndDistortionDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of SimpleRadialSplitFixedFocalAndDistortion factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetSimpleRadialSplitFixedFocalAndDistortionNum(size_t num);

  /**
   * Set the indices for the pose argument for the
   * SimpleRadialSplitFixedPrincipalPoint factor from host.
   */
  void SetSimpleRadialSplitFixedPrincipalPointPoseIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the
   * SimpleRadialSplitFixedPrincipalPoint factor from device.
   */
  void SetSimpleRadialSplitFixedPrincipalPointPoseIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal_and_distortion argument for the
   * SimpleRadialSplitFixedPrincipalPoint factor from host.
   */
  void SetSimpleRadialSplitFixedPrincipalPointFocalAndDistortionIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal_and_distortion argument for the
   * SimpleRadialSplitFixedPrincipalPoint factor from device.
   */
  void
  SetSimpleRadialSplitFixedPrincipalPointFocalAndDistortionIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * SimpleRadialSplitFixedPrincipalPoint factor from host.
   */
  void SetSimpleRadialSplitFixedPrincipalPointPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * SimpleRadialSplitFixedPrincipalPoint factor from device.
   */
  void SetSimpleRadialSplitFixedPrincipalPointPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts SimpleRadialSplitFixedPrincipalPoint
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialSplitFixedPrincipalPointPixelDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts SimpleRadialSplitFixedPrincipalPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialSplitFixedPrincipalPointPixelDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * SimpleRadialSplitFixedPrincipalPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialSplitFixedPrincipalPointPrincipalPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * SimpleRadialSplitFixedPrincipalPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPrincipalPointPrincipalPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of SimpleRadialSplitFixedPrincipalPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetSimpleRadialSplitFixedPrincipalPointNum(size_t num);

  /**
   * Set the indices for the principal_point argument for the
   * SimpleRadialSplitFixedPoseFixedFocalAndDistortion factor from host.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionPrincipalPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the principal_point argument for the
   * SimpleRadialSplitFixedPoseFixedFocalAndDistortion factor from device.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionPrincipalPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * SimpleRadialSplitFixedPoseFixedFocalAndDistortion factor from host.
   */
  void SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * SimpleRadialSplitFixedPoseFixedFocalAndDistortion factor from device.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialSplitFixedPoseFixedFocalAndDistortion factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionPixelDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialSplitFixedPoseFixedFocalAndDistortion factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionPixelDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * SimpleRadialSplitFixedPoseFixedFocalAndDistortion factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionPoseDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * SimpleRadialSplitFixedPoseFixedFocalAndDistortion factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionPoseDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_distortion consts
   * SimpleRadialSplitFixedPoseFixedFocalAndDistortion factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionFocalAndDistortionDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_distortion consts
   * SimpleRadialSplitFixedPoseFixedFocalAndDistortion factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionFocalAndDistortionDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of SimpleRadialSplitFixedPoseFixedFocalAndDistortion
   * factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionNum(size_t num);

  /**
   * Set the indices for the focal_and_distortion argument for the
   * SimpleRadialSplitFixedPoseFixedPrincipalPoint factor from host.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedPrincipalPointFocalAndDistortionIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal_and_distortion argument for the
   * SimpleRadialSplitFixedPoseFixedPrincipalPoint factor from device.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedPrincipalPointFocalAndDistortionIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * SimpleRadialSplitFixedPoseFixedPrincipalPoint factor from host.
   */
  void SetSimpleRadialSplitFixedPoseFixedPrincipalPointPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * SimpleRadialSplitFixedPoseFixedPrincipalPoint factor from device.
   */
  void SetSimpleRadialSplitFixedPoseFixedPrincipalPointPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialSplitFixedPoseFixedPrincipalPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialSplitFixedPoseFixedPrincipalPointPixelDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialSplitFixedPoseFixedPrincipalPoint factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedPrincipalPointPixelDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * SimpleRadialSplitFixedPoseFixedPrincipalPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetSimpleRadialSplitFixedPoseFixedPrincipalPointPoseDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * SimpleRadialSplitFixedPoseFixedPrincipalPoint factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedPrincipalPointPoseDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * SimpleRadialSplitFixedPoseFixedPrincipalPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedPrincipalPointPrincipalPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * SimpleRadialSplitFixedPoseFixedPrincipalPoint factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedPrincipalPointPrincipalPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of SimpleRadialSplitFixedPoseFixedPrincipalPoint
   * factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetSimpleRadialSplitFixedPoseFixedPrincipalPointNum(size_t num);

  /**
   * Set the indices for the pose argument for the
   * SimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPoint factor from
   * host.
   */
  void
  SetSimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointPoseIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the
   * SimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPoint factor from
   * device.
   */
  void
  SetSimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointPoseIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * SimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPoint factor from
   * host.
   */
  void
  SetSimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * SimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPoint factor from
   * device.
   */
  void
  SetSimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPoint factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointPixelDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPoint factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointPixelDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_distortion consts
   * SimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPoint factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointFocalAndDistortionDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_distortion consts
   * SimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPoint factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointFocalAndDistortionDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * SimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPoint factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointPrincipalPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * SimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPoint factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointPrincipalPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of
   * SimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetSimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointNum(
      size_t num);

  /**
   * Set the indices for the pose argument for the
   * SimpleRadialSplitFixedFocalAndDistortionFixedPoint factor from host.
   */
  void SetSimpleRadialSplitFixedFocalAndDistortionFixedPointPoseIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the
   * SimpleRadialSplitFixedFocalAndDistortionFixedPoint factor from device.
   */
  void
  SetSimpleRadialSplitFixedFocalAndDistortionFixedPointPoseIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the principal_point argument for the
   * SimpleRadialSplitFixedFocalAndDistortionFixedPoint factor from host.
   */
  void
  SetSimpleRadialSplitFixedFocalAndDistortionFixedPointPrincipalPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the principal_point argument for the
   * SimpleRadialSplitFixedFocalAndDistortionFixedPoint factor from device.
   */
  void
  SetSimpleRadialSplitFixedFocalAndDistortionFixedPointPrincipalPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialSplitFixedFocalAndDistortionFixedPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndDistortionFixedPointPixelDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialSplitFixedFocalAndDistortionFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndDistortionFixedPointPixelDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_distortion consts
   * SimpleRadialSplitFixedFocalAndDistortionFixedPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndDistortionFixedPointFocalAndDistortionDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_distortion consts
   * SimpleRadialSplitFixedFocalAndDistortionFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndDistortionFixedPointFocalAndDistortionDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * SimpleRadialSplitFixedFocalAndDistortionFixedPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndDistortionFixedPointPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * SimpleRadialSplitFixedFocalAndDistortionFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndDistortionFixedPointPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of
   * SimpleRadialSplitFixedFocalAndDistortionFixedPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetSimpleRadialSplitFixedFocalAndDistortionFixedPointNum(size_t num);

  /**
   * Set the indices for the pose argument for the
   * SimpleRadialSplitFixedPrincipalPointFixedPoint factor from host.
   */
  void SetSimpleRadialSplitFixedPrincipalPointFixedPointPoseIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the
   * SimpleRadialSplitFixedPrincipalPointFixedPoint factor from device.
   */
  void SetSimpleRadialSplitFixedPrincipalPointFixedPointPoseIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal_and_distortion argument for the
   * SimpleRadialSplitFixedPrincipalPointFixedPoint factor from host.
   */
  void
  SetSimpleRadialSplitFixedPrincipalPointFixedPointFocalAndDistortionIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal_and_distortion argument for the
   * SimpleRadialSplitFixedPrincipalPointFixedPoint factor from device.
   */
  void
  SetSimpleRadialSplitFixedPrincipalPointFixedPointFocalAndDistortionIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialSplitFixedPrincipalPointFixedPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPrincipalPointFixedPointPixelDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialSplitFixedPrincipalPointFixedPoint factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPrincipalPointFixedPointPixelDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * SimpleRadialSplitFixedPrincipalPointFixedPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * SimpleRadialSplitFixedPrincipalPointFixedPoint factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPrincipalPointFixedPointPrincipalPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * SimpleRadialSplitFixedPrincipalPointFixedPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPrincipalPointFixedPointPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * SimpleRadialSplitFixedPrincipalPointFixedPoint factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPrincipalPointFixedPointPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of SimpleRadialSplitFixedPrincipalPointFixedPoint
   * factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetSimpleRadialSplitFixedPrincipalPointFixedPointNum(size_t num);

  /**
   * Set the indices for the point argument for the
   * SimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPrincipalPoint factor
   * from host.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPrincipalPointPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * SimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPrincipalPoint factor
   * from device.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPrincipalPointPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPrincipalPoint factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPrincipalPointPixelDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPrincipalPoint factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPrincipalPointPixelDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * SimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPrincipalPoint factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPrincipalPointPoseDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * SimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPrincipalPoint factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPrincipalPointPoseDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_distortion consts
   * SimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPrincipalPoint factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPrincipalPointFocalAndDistortionDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_distortion consts
   * SimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPrincipalPoint factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPrincipalPointFocalAndDistortionDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * SimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPrincipalPoint factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPrincipalPointPrincipalPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * SimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPrincipalPoint factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPrincipalPointPrincipalPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of
   * SimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPrincipalPoint
   * factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPrincipalPointNum(
      size_t num);

  /**
   * Set the indices for the principal_point argument for the
   * SimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPoint factor from
   * host.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPointPrincipalPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the principal_point argument for the
   * SimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPoint factor from
   * device.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPointPrincipalPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPoint factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPointPixelDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPoint factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPointPixelDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * SimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPoint factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPointPoseDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * SimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPoint factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPointPoseDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_distortion consts
   * SimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPoint factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPointFocalAndDistortionDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_distortion consts
   * SimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPoint factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPointFocalAndDistortionDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * SimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPoint factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPointPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * SimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPoint factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPointPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of
   * SimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetSimpleRadialSplitFixedPoseFixedFocalAndDistortionFixedPointNum(
      size_t num);

  /**
   * Set the indices for the focal_and_distortion argument for the
   * SimpleRadialSplitFixedPoseFixedPrincipalPointFixedPoint factor from host.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointFocalAndDistortionIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal_and_distortion argument for the
   * SimpleRadialSplitFixedPoseFixedPrincipalPointFixedPoint factor from device.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointFocalAndDistortionIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialSplitFixedPoseFixedPrincipalPointFixedPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointPixelDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialSplitFixedPoseFixedPrincipalPointFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointPixelDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * SimpleRadialSplitFixedPoseFixedPrincipalPointFixedPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointPoseDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * SimpleRadialSplitFixedPoseFixedPrincipalPointFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointPoseDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * SimpleRadialSplitFixedPoseFixedPrincipalPointFixedPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * SimpleRadialSplitFixedPoseFixedPrincipalPointFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointPrincipalPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * SimpleRadialSplitFixedPoseFixedPrincipalPointFixedPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * SimpleRadialSplitFixedPoseFixedPrincipalPointFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of
   * SimpleRadialSplitFixedPoseFixedPrincipalPointFixedPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetSimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointNum(
      size_t num);

  /**
   * Set the indices for the pose argument for the
   * SimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointFixedPoint
   * factor from host.
   */
  void
  SetSimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointFixedPointPoseIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the
   * SimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointFixedPoint
   * factor from device.
   */
  void
  SetSimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointFixedPointPoseIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointFixedPoint
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointFixedPointPixelDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * SimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointFixedPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointFixedPointPixelDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_distortion consts
   * SimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointFixedPoint
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointFixedPointFocalAndDistortionDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal_and_distortion consts
   * SimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointFixedPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointFixedPointFocalAndDistortionDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * SimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointFixedPoint
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * SimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointFixedPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointFixedPointPrincipalPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * SimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointFixedPoint
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointFixedPointPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * SimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointFixedPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetSimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointFixedPointPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of
   * SimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointFixedPoint
   * factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void
  SetSimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointFixedPointNum(
      size_t num);

  /**
   * Set the indices for the pose argument for the PinholeSplitFixedFocal factor
   * from host.
   */
  void SetPinholeSplitFixedFocalPoseIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the PinholeSplitFixedFocal factor
   * from device.
   */
  void SetPinholeSplitFixedFocalPoseIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the principal_point argument for the
   * PinholeSplitFixedFocal factor from host.
   */
  void SetPinholeSplitFixedFocalPrincipalPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the principal_point argument for the
   * PinholeSplitFixedFocal factor from device.
   */
  void SetPinholeSplitFixedFocalPrincipalPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the PinholeSplitFixedFocal
   * factor from host.
   */
  void SetPinholeSplitFixedFocalPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the PinholeSplitFixedFocal
   * factor from device.
   */
  void SetPinholeSplitFixedFocalPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts PinholeSplitFixedFocal factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedFocalPixelDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts PinholeSplitFixedFocal factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedFocalPixelDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the weight_loss consts PinholeSplitFixedFocal factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedFocalWeightLossDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the weight_loss consts PinholeSplitFixedFocal factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedFocalWeightLossDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts PinholeSplitFixedFocal factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedFocalFocalDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts PinholeSplitFixedFocal factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedFocalFocalDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of PinholeSplitFixedFocal factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeSplitFixedFocalNum(size_t num);

  /**
   * Set the indices for the pose argument for the
   * PinholeSplitFixedPrincipalPoint factor from host.
   */
  void SetPinholeSplitFixedPrincipalPointPoseIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the
   * PinholeSplitFixedPrincipalPoint factor from device.
   */
  void SetPinholeSplitFixedPrincipalPointPoseIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal argument for the
   * PinholeSplitFixedPrincipalPoint factor from host.
   */
  void SetPinholeSplitFixedPrincipalPointFocalIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal argument for the
   * PinholeSplitFixedPrincipalPoint factor from device.
   */
  void SetPinholeSplitFixedPrincipalPointFocalIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * PinholeSplitFixedPrincipalPoint factor from host.
   */
  void SetPinholeSplitFixedPrincipalPointPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * PinholeSplitFixedPrincipalPoint factor from device.
   */
  void SetPinholeSplitFixedPrincipalPointPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts PinholeSplitFixedPrincipalPoint factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPrincipalPointPixelDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts PinholeSplitFixedPrincipalPoint factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPrincipalPointPixelDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the weight_loss consts PinholeSplitFixedPrincipalPoint
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPrincipalPointWeightLossDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the weight_loss consts PinholeSplitFixedPrincipalPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPrincipalPointWeightLossDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * PinholeSplitFixedPrincipalPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPrincipalPointPrincipalPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * PinholeSplitFixedPrincipalPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPrincipalPointPrincipalPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of PinholeSplitFixedPrincipalPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeSplitFixedPrincipalPointNum(size_t num);

  /**
   * Set the indices for the principal_point argument for the
   * PinholeSplitFixedPoseFixedFocal factor from host.
   */
  void SetPinholeSplitFixedPoseFixedFocalPrincipalPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the principal_point argument for the
   * PinholeSplitFixedPoseFixedFocal factor from device.
   */
  void SetPinholeSplitFixedPoseFixedFocalPrincipalPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * PinholeSplitFixedPoseFixedFocal factor from host.
   */
  void SetPinholeSplitFixedPoseFixedFocalPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * PinholeSplitFixedPoseFixedFocal factor from device.
   */
  void SetPinholeSplitFixedPoseFixedFocalPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts PinholeSplitFixedPoseFixedFocal factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPoseFixedFocalPixelDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts PinholeSplitFixedPoseFixedFocal factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPoseFixedFocalPixelDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the weight_loss consts PinholeSplitFixedPoseFixedFocal
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPoseFixedFocalWeightLossDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the weight_loss consts PinholeSplitFixedPoseFixedFocal
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPoseFixedFocalWeightLossDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts PinholeSplitFixedPoseFixedFocal factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPoseFixedFocalPoseDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts PinholeSplitFixedPoseFixedFocal factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPoseFixedFocalPoseDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts PinholeSplitFixedPoseFixedFocal factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPoseFixedFocalFocalDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts PinholeSplitFixedPoseFixedFocal factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPoseFixedFocalFocalDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of PinholeSplitFixedPoseFixedFocal factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeSplitFixedPoseFixedFocalNum(size_t num);

  /**
   * Set the indices for the focal argument for the
   * PinholeSplitFixedPoseFixedPrincipalPoint factor from host.
   */
  void SetPinholeSplitFixedPoseFixedPrincipalPointFocalIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal argument for the
   * PinholeSplitFixedPoseFixedPrincipalPoint factor from device.
   */
  void SetPinholeSplitFixedPoseFixedPrincipalPointFocalIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * PinholeSplitFixedPoseFixedPrincipalPoint factor from host.
   */
  void SetPinholeSplitFixedPoseFixedPrincipalPointPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * PinholeSplitFixedPoseFixedPrincipalPoint factor from device.
   */
  void SetPinholeSplitFixedPoseFixedPrincipalPointPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts
   * PinholeSplitFixedPoseFixedPrincipalPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPoseFixedPrincipalPointPixelDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * PinholeSplitFixedPoseFixedPrincipalPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPoseFixedPrincipalPointPixelDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the weight_loss consts
   * PinholeSplitFixedPoseFixedPrincipalPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPoseFixedPrincipalPointWeightLossDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the weight_loss consts
   * PinholeSplitFixedPoseFixedPrincipalPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedPrincipalPointWeightLossDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts PinholeSplitFixedPoseFixedPrincipalPoint
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPoseFixedPrincipalPointPoseDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts PinholeSplitFixedPoseFixedPrincipalPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPoseFixedPrincipalPointPoseDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * PinholeSplitFixedPoseFixedPrincipalPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedPrincipalPointPrincipalPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * PinholeSplitFixedPoseFixedPrincipalPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedPrincipalPointPrincipalPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of PinholeSplitFixedPoseFixedPrincipalPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeSplitFixedPoseFixedPrincipalPointNum(size_t num);

  /**
   * Set the indices for the pose argument for the
   * PinholeSplitFixedFocalFixedPrincipalPoint factor from host.
   */
  void SetPinholeSplitFixedFocalFixedPrincipalPointPoseIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the
   * PinholeSplitFixedFocalFixedPrincipalPoint factor from device.
   */
  void SetPinholeSplitFixedFocalFixedPrincipalPointPoseIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * PinholeSplitFixedFocalFixedPrincipalPoint factor from host.
   */
  void SetPinholeSplitFixedFocalFixedPrincipalPointPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * PinholeSplitFixedFocalFixedPrincipalPoint factor from device.
   */
  void SetPinholeSplitFixedFocalFixedPrincipalPointPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts
   * PinholeSplitFixedFocalFixedPrincipalPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedFocalFixedPrincipalPointPixelDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * PinholeSplitFixedFocalFixedPrincipalPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedFocalFixedPrincipalPointPixelDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the weight_loss consts
   * PinholeSplitFixedFocalFixedPrincipalPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedFocalFixedPrincipalPointWeightLossDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the weight_loss consts
   * PinholeSplitFixedFocalFixedPrincipalPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedFocalFixedPrincipalPointWeightLossDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts
   * PinholeSplitFixedFocalFixedPrincipalPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedFocalFixedPrincipalPointFocalDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts
   * PinholeSplitFixedFocalFixedPrincipalPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedFocalFixedPrincipalPointFocalDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * PinholeSplitFixedFocalFixedPrincipalPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedFocalFixedPrincipalPointPrincipalPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * PinholeSplitFixedFocalFixedPrincipalPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedFocalFixedPrincipalPointPrincipalPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of PinholeSplitFixedFocalFixedPrincipalPoint
   * factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeSplitFixedFocalFixedPrincipalPointNum(size_t num);

  /**
   * Set the indices for the pose argument for the
   * PinholeSplitFixedFocalFixedPoint factor from host.
   */
  void SetPinholeSplitFixedFocalFixedPointPoseIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the
   * PinholeSplitFixedFocalFixedPoint factor from device.
   */
  void SetPinholeSplitFixedFocalFixedPointPoseIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the principal_point argument for the
   * PinholeSplitFixedFocalFixedPoint factor from host.
   */
  void SetPinholeSplitFixedFocalFixedPointPrincipalPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the principal_point argument for the
   * PinholeSplitFixedFocalFixedPoint factor from device.
   */
  void SetPinholeSplitFixedFocalFixedPointPrincipalPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts PinholeSplitFixedFocalFixedPoint factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedFocalFixedPointPixelDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts PinholeSplitFixedFocalFixedPoint factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedFocalFixedPointPixelDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the weight_loss consts PinholeSplitFixedFocalFixedPoint
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedFocalFixedPointWeightLossDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the weight_loss consts PinholeSplitFixedFocalFixedPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedFocalFixedPointWeightLossDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts PinholeSplitFixedFocalFixedPoint factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedFocalFixedPointFocalDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts PinholeSplitFixedFocalFixedPoint factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedFocalFixedPointFocalDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts PinholeSplitFixedFocalFixedPoint factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedFocalFixedPointPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts PinholeSplitFixedFocalFixedPoint factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedFocalFixedPointPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of PinholeSplitFixedFocalFixedPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeSplitFixedFocalFixedPointNum(size_t num);

  /**
   * Set the indices for the pose argument for the
   * PinholeSplitFixedPrincipalPointFixedPoint factor from host.
   */
  void SetPinholeSplitFixedPrincipalPointFixedPointPoseIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the
   * PinholeSplitFixedPrincipalPointFixedPoint factor from device.
   */
  void SetPinholeSplitFixedPrincipalPointFixedPointPoseIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal argument for the
   * PinholeSplitFixedPrincipalPointFixedPoint factor from host.
   */
  void SetPinholeSplitFixedPrincipalPointFixedPointFocalIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal argument for the
   * PinholeSplitFixedPrincipalPointFixedPoint factor from device.
   */
  void SetPinholeSplitFixedPrincipalPointFixedPointFocalIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts
   * PinholeSplitFixedPrincipalPointFixedPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPrincipalPointFixedPointPixelDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * PinholeSplitFixedPrincipalPointFixedPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPrincipalPointFixedPointPixelDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the weight_loss consts
   * PinholeSplitFixedPrincipalPointFixedPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPrincipalPointFixedPointWeightLossDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the weight_loss consts
   * PinholeSplitFixedPrincipalPointFixedPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPrincipalPointFixedPointWeightLossDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * PinholeSplitFixedPrincipalPointFixedPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * PinholeSplitFixedPrincipalPointFixedPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPrincipalPointFixedPointPrincipalPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * PinholeSplitFixedPrincipalPointFixedPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPrincipalPointFixedPointPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * PinholeSplitFixedPrincipalPointFixedPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPrincipalPointFixedPointPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of PinholeSplitFixedPrincipalPointFixedPoint
   * factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeSplitFixedPrincipalPointFixedPointNum(size_t num);

  /**
   * Set the indices for the point argument for the
   * PinholeSplitFixedPoseFixedFocalFixedPrincipalPoint factor from host.
   */
  void
  SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * PinholeSplitFixedPoseFixedFocalFixedPrincipalPoint factor from device.
   */
  void
  SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts
   * PinholeSplitFixedPoseFixedFocalFixedPrincipalPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointPixelDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * PinholeSplitFixedPoseFixedFocalFixedPrincipalPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointPixelDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the weight_loss consts
   * PinholeSplitFixedPoseFixedFocalFixedPrincipalPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointWeightLossDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the weight_loss consts
   * PinholeSplitFixedPoseFixedFocalFixedPrincipalPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointWeightLossDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * PinholeSplitFixedPoseFixedFocalFixedPrincipalPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointPoseDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * PinholeSplitFixedPoseFixedFocalFixedPrincipalPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointPoseDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts
   * PinholeSplitFixedPoseFixedFocalFixedPrincipalPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointFocalDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts
   * PinholeSplitFixedPoseFixedFocalFixedPrincipalPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointFocalDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * PinholeSplitFixedPoseFixedFocalFixedPrincipalPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointPrincipalPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * PinholeSplitFixedPoseFixedFocalFixedPrincipalPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointPrincipalPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of
   * PinholeSplitFixedPoseFixedFocalFixedPrincipalPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeSplitFixedPoseFixedFocalFixedPrincipalPointNum(size_t num);

  /**
   * Set the indices for the principal_point argument for the
   * PinholeSplitFixedPoseFixedFocalFixedPoint factor from host.
   */
  void
  SetPinholeSplitFixedPoseFixedFocalFixedPointPrincipalPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the principal_point argument for the
   * PinholeSplitFixedPoseFixedFocalFixedPoint factor from device.
   */
  void
  SetPinholeSplitFixedPoseFixedFocalFixedPointPrincipalPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts
   * PinholeSplitFixedPoseFixedFocalFixedPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPoseFixedFocalFixedPointPixelDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * PinholeSplitFixedPoseFixedFocalFixedPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPoseFixedFocalFixedPointPixelDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the weight_loss consts
   * PinholeSplitFixedPoseFixedFocalFixedPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedFocalFixedPointWeightLossDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the weight_loss consts
   * PinholeSplitFixedPoseFixedFocalFixedPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedFocalFixedPointWeightLossDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * PinholeSplitFixedPoseFixedFocalFixedPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPoseFixedFocalFixedPointPoseDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * PinholeSplitFixedPoseFixedFocalFixedPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPoseFixedFocalFixedPointPoseDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts
   * PinholeSplitFixedPoseFixedFocalFixedPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPoseFixedFocalFixedPointFocalDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts
   * PinholeSplitFixedPoseFixedFocalFixedPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPoseFixedFocalFixedPointFocalDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * PinholeSplitFixedPoseFixedFocalFixedPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPoseFixedFocalFixedPointPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * PinholeSplitFixedPoseFixedFocalFixedPoint factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedPoseFixedFocalFixedPointPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of PinholeSplitFixedPoseFixedFocalFixedPoint
   * factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeSplitFixedPoseFixedFocalFixedPointNum(size_t num);

  /**
   * Set the indices for the focal argument for the
   * PinholeSplitFixedPoseFixedPrincipalPointFixedPoint factor from host.
   */
  void
  SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointFocalIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal argument for the
   * PinholeSplitFixedPoseFixedPrincipalPointFixedPoint factor from device.
   */
  void
  SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointFocalIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts
   * PinholeSplitFixedPoseFixedPrincipalPointFixedPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointPixelDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * PinholeSplitFixedPoseFixedPrincipalPointFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointPixelDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the weight_loss consts
   * PinholeSplitFixedPoseFixedPrincipalPointFixedPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointWeightLossDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the weight_loss consts
   * PinholeSplitFixedPoseFixedPrincipalPointFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointWeightLossDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * PinholeSplitFixedPoseFixedPrincipalPointFixedPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointPoseDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pose consts
   * PinholeSplitFixedPoseFixedPrincipalPointFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointPoseDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * PinholeSplitFixedPoseFixedPrincipalPointFixedPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * PinholeSplitFixedPoseFixedPrincipalPointFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointPrincipalPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * PinholeSplitFixedPoseFixedPrincipalPointFixedPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * PinholeSplitFixedPoseFixedPrincipalPointFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of
   * PinholeSplitFixedPoseFixedPrincipalPointFixedPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeSplitFixedPoseFixedPrincipalPointFixedPointNum(size_t num);

  /**
   * Set the indices for the pose argument for the
   * PinholeSplitFixedFocalFixedPrincipalPointFixedPoint factor from host.
   */
  void
  SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointPoseIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the pose argument for the
   * PinholeSplitFixedFocalFixedPrincipalPointFixedPoint factor from device.
   */
  void
  SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointPoseIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the pixel consts
   * PinholeSplitFixedFocalFixedPrincipalPointFixedPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointPixelDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * PinholeSplitFixedFocalFixedPrincipalPointFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointPixelDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the weight_loss consts
   * PinholeSplitFixedFocalFixedPrincipalPointFixedPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointWeightLossDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the weight_loss consts
   * PinholeSplitFixedFocalFixedPrincipalPointFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointWeightLossDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts
   * PinholeSplitFixedFocalFixedPrincipalPointFixedPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointFocalDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts
   * PinholeSplitFixedFocalFixedPrincipalPointFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointFocalDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * PinholeSplitFixedFocalFixedPrincipalPointFixedPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * PinholeSplitFixedFocalFixedPrincipalPointFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointPrincipalPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * PinholeSplitFixedFocalFixedPrincipalPointFixedPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * PinholeSplitFixedFocalFixedPrincipalPointFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of
   * PinholeSplitFixedFocalFixedPrincipalPointFixedPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeSplitFixedFocalFixedPrincipalPointFixedPointNum(size_t num);

  /**
   * Set the indices for the translation argument for the
   * PinholeSplitFixedRotationFixedFocal factor from host.
   */
  void SetPinholeSplitFixedRotationFixedFocalTranslationIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the translation argument for the
   * PinholeSplitFixedRotationFixedFocal factor from device.
   */
  void SetPinholeSplitFixedRotationFixedFocalTranslationIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the principal_point argument for the
   * PinholeSplitFixedRotationFixedFocal factor from host.
   */
  void SetPinholeSplitFixedRotationFixedFocalPrincipalPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the principal_point argument for the
   * PinholeSplitFixedRotationFixedFocal factor from device.
   */
  void SetPinholeSplitFixedRotationFixedFocalPrincipalPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * PinholeSplitFixedRotationFixedFocal factor from host.
   */
  void SetPinholeSplitFixedRotationFixedFocalPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * PinholeSplitFixedRotationFixedFocal factor from device.
   */
  void SetPinholeSplitFixedRotationFixedFocalPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the rotation consts PinholeSplitFixedRotationFixedFocal
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedRotationFixedFocalRotationDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the rotation consts PinholeSplitFixedRotationFixedFocal
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedRotationFixedFocalRotationDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts PinholeSplitFixedRotationFixedFocal
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedRotationFixedFocalPixelDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts PinholeSplitFixedRotationFixedFocal
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedRotationFixedFocalPixelDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the weight_loss consts
   * PinholeSplitFixedRotationFixedFocal factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedRotationFixedFocalWeightLossDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the weight_loss consts
   * PinholeSplitFixedRotationFixedFocal factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedRotationFixedFocalWeightLossDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts PinholeSplitFixedRotationFixedFocal
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedRotationFixedFocalFocalDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts PinholeSplitFixedRotationFixedFocal
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedRotationFixedFocalFocalDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of PinholeSplitFixedRotationFixedFocal factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeSplitFixedRotationFixedFocalNum(size_t num);

  /**
   * Set the indices for the translation argument for the
   * PinholeSplitFixedRotationFixedPrincipalPoint factor from host.
   */
  void
  SetPinholeSplitFixedRotationFixedPrincipalPointTranslationIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the translation argument for the
   * PinholeSplitFixedRotationFixedPrincipalPoint factor from device.
   */
  void
  SetPinholeSplitFixedRotationFixedPrincipalPointTranslationIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal argument for the
   * PinholeSplitFixedRotationFixedPrincipalPoint factor from host.
   */
  void SetPinholeSplitFixedRotationFixedPrincipalPointFocalIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal argument for the
   * PinholeSplitFixedRotationFixedPrincipalPoint factor from device.
   */
  void SetPinholeSplitFixedRotationFixedPrincipalPointFocalIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * PinholeSplitFixedRotationFixedPrincipalPoint factor from host.
   */
  void SetPinholeSplitFixedRotationFixedPrincipalPointPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * PinholeSplitFixedRotationFixedPrincipalPoint factor from device.
   */
  void SetPinholeSplitFixedRotationFixedPrincipalPointPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the rotation consts
   * PinholeSplitFixedRotationFixedPrincipalPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedRotationFixedPrincipalPointRotationDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the rotation consts
   * PinholeSplitFixedRotationFixedPrincipalPoint factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedRotationFixedPrincipalPointRotationDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * PinholeSplitFixedRotationFixedPrincipalPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedRotationFixedPrincipalPointPixelDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * PinholeSplitFixedRotationFixedPrincipalPoint factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedRotationFixedPrincipalPointPixelDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the weight_loss consts
   * PinholeSplitFixedRotationFixedPrincipalPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedRotationFixedPrincipalPointWeightLossDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the weight_loss consts
   * PinholeSplitFixedRotationFixedPrincipalPoint factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedRotationFixedPrincipalPointWeightLossDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * PinholeSplitFixedRotationFixedPrincipalPoint factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedRotationFixedPrincipalPointPrincipalPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * PinholeSplitFixedRotationFixedPrincipalPoint factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedRotationFixedPrincipalPointPrincipalPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of PinholeSplitFixedRotationFixedPrincipalPoint
   * factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeSplitFixedRotationFixedPrincipalPointNum(size_t num);

  /**
   * Set the indices for the translation argument for the
   * PinholeSplitFixedRotationFixedFocalFixedPrincipalPoint factor from host.
   */
  void
  SetPinholeSplitFixedRotationFixedFocalFixedPrincipalPointTranslationIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the translation argument for the
   * PinholeSplitFixedRotationFixedFocalFixedPrincipalPoint factor from device.
   */
  void
  SetPinholeSplitFixedRotationFixedFocalFixedPrincipalPointTranslationIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * PinholeSplitFixedRotationFixedFocalFixedPrincipalPoint factor from host.
   */
  void
  SetPinholeSplitFixedRotationFixedFocalFixedPrincipalPointPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the point argument for the
   * PinholeSplitFixedRotationFixedFocalFixedPrincipalPoint factor from device.
   */
  void
  SetPinholeSplitFixedRotationFixedFocalFixedPrincipalPointPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the rotation consts
   * PinholeSplitFixedRotationFixedFocalFixedPrincipalPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedRotationFixedFocalFixedPrincipalPointRotationDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the rotation consts
   * PinholeSplitFixedRotationFixedFocalFixedPrincipalPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedRotationFixedFocalFixedPrincipalPointRotationDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * PinholeSplitFixedRotationFixedFocalFixedPrincipalPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedRotationFixedFocalFixedPrincipalPointPixelDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * PinholeSplitFixedRotationFixedFocalFixedPrincipalPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedRotationFixedFocalFixedPrincipalPointPixelDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the weight_loss consts
   * PinholeSplitFixedRotationFixedFocalFixedPrincipalPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedRotationFixedFocalFixedPrincipalPointWeightLossDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the weight_loss consts
   * PinholeSplitFixedRotationFixedFocalFixedPrincipalPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedRotationFixedFocalFixedPrincipalPointWeightLossDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts
   * PinholeSplitFixedRotationFixedFocalFixedPrincipalPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedRotationFixedFocalFixedPrincipalPointFocalDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts
   * PinholeSplitFixedRotationFixedFocalFixedPrincipalPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedRotationFixedFocalFixedPrincipalPointFocalDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * PinholeSplitFixedRotationFixedFocalFixedPrincipalPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedRotationFixedFocalFixedPrincipalPointPrincipalPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * PinholeSplitFixedRotationFixedFocalFixedPrincipalPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedRotationFixedFocalFixedPrincipalPointPrincipalPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of
   * PinholeSplitFixedRotationFixedFocalFixedPrincipalPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeSplitFixedRotationFixedFocalFixedPrincipalPointNum(size_t num);

  /**
   * Set the indices for the translation argument for the
   * PinholeSplitFixedRotationFixedFocalFixedPoint factor from host.
   */
  void
  SetPinholeSplitFixedRotationFixedFocalFixedPointTranslationIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the translation argument for the
   * PinholeSplitFixedRotationFixedFocalFixedPoint factor from device.
   */
  void
  SetPinholeSplitFixedRotationFixedFocalFixedPointTranslationIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the principal_point argument for the
   * PinholeSplitFixedRotationFixedFocalFixedPoint factor from host.
   */
  void
  SetPinholeSplitFixedRotationFixedFocalFixedPointPrincipalPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the principal_point argument for the
   * PinholeSplitFixedRotationFixedFocalFixedPoint factor from device.
   */
  void
  SetPinholeSplitFixedRotationFixedFocalFixedPointPrincipalPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the rotation consts
   * PinholeSplitFixedRotationFixedFocalFixedPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedRotationFixedFocalFixedPointRotationDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the rotation consts
   * PinholeSplitFixedRotationFixedFocalFixedPoint factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedRotationFixedFocalFixedPointRotationDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * PinholeSplitFixedRotationFixedFocalFixedPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedRotationFixedFocalFixedPointPixelDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * PinholeSplitFixedRotationFixedFocalFixedPoint factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedRotationFixedFocalFixedPointPixelDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the weight_loss consts
   * PinholeSplitFixedRotationFixedFocalFixedPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedRotationFixedFocalFixedPointWeightLossDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the weight_loss consts
   * PinholeSplitFixedRotationFixedFocalFixedPoint factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedRotationFixedFocalFixedPointWeightLossDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts
   * PinholeSplitFixedRotationFixedFocalFixedPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedRotationFixedFocalFixedPointFocalDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts
   * PinholeSplitFixedRotationFixedFocalFixedPoint factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedRotationFixedFocalFixedPointFocalDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * PinholeSplitFixedRotationFixedFocalFixedPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitFixedRotationFixedFocalFixedPointPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * PinholeSplitFixedRotationFixedFocalFixedPoint factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedRotationFixedFocalFixedPointPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of PinholeSplitFixedRotationFixedFocalFixedPoint
   * factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeSplitFixedRotationFixedFocalFixedPointNum(size_t num);

  /**
   * Set the indices for the translation argument for the
   * PinholeSplitFixedRotationFixedPrincipalPointFixedPoint factor from host.
   */
  void
  SetPinholeSplitFixedRotationFixedPrincipalPointFixedPointTranslationIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the translation argument for the
   * PinholeSplitFixedRotationFixedPrincipalPointFixedPoint factor from device.
   */
  void
  SetPinholeSplitFixedRotationFixedPrincipalPointFixedPointTranslationIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal argument for the
   * PinholeSplitFixedRotationFixedPrincipalPointFixedPoint factor from host.
   */
  void
  SetPinholeSplitFixedRotationFixedPrincipalPointFixedPointFocalIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal argument for the
   * PinholeSplitFixedRotationFixedPrincipalPointFixedPoint factor from device.
   */
  void
  SetPinholeSplitFixedRotationFixedPrincipalPointFixedPointFocalIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the rotation consts
   * PinholeSplitFixedRotationFixedPrincipalPointFixedPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedRotationFixedPrincipalPointFixedPointRotationDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the rotation consts
   * PinholeSplitFixedRotationFixedPrincipalPointFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedRotationFixedPrincipalPointFixedPointRotationDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * PinholeSplitFixedRotationFixedPrincipalPointFixedPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedRotationFixedPrincipalPointFixedPointPixelDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * PinholeSplitFixedRotationFixedPrincipalPointFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedRotationFixedPrincipalPointFixedPointPixelDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the weight_loss consts
   * PinholeSplitFixedRotationFixedPrincipalPointFixedPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedRotationFixedPrincipalPointFixedPointWeightLossDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the weight_loss consts
   * PinholeSplitFixedRotationFixedPrincipalPointFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedRotationFixedPrincipalPointFixedPointWeightLossDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * PinholeSplitFixedRotationFixedPrincipalPointFixedPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedRotationFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * PinholeSplitFixedRotationFixedPrincipalPointFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedRotationFixedPrincipalPointFixedPointPrincipalPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * PinholeSplitFixedRotationFixedPrincipalPointFixedPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedRotationFixedPrincipalPointFixedPointPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * PinholeSplitFixedRotationFixedPrincipalPointFixedPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedRotationFixedPrincipalPointFixedPointPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of
   * PinholeSplitFixedRotationFixedPrincipalPointFixedPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeSplitFixedRotationFixedPrincipalPointFixedPointNum(size_t num);

  /**
   * Set the indices for the translation argument for the
   * PinholeSplitFixedRotationFixedFocalFixedPrincipalPointFixedPoint factor
   * from host.
   */
  void
  SetPinholeSplitFixedRotationFixedFocalFixedPrincipalPointFixedPointTranslationIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the translation argument for the
   * PinholeSplitFixedRotationFixedFocalFixedPrincipalPointFixedPoint factor
   * from device.
   */
  void
  SetPinholeSplitFixedRotationFixedFocalFixedPrincipalPointFixedPointTranslationIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the rotation consts
   * PinholeSplitFixedRotationFixedFocalFixedPrincipalPointFixedPoint factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedRotationFixedFocalFixedPrincipalPointFixedPointRotationDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the rotation consts
   * PinholeSplitFixedRotationFixedFocalFixedPrincipalPointFixedPoint factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedRotationFixedFocalFixedPrincipalPointFixedPointRotationDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * PinholeSplitFixedRotationFixedFocalFixedPrincipalPointFixedPoint factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedRotationFixedFocalFixedPrincipalPointFixedPointPixelDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the pixel consts
   * PinholeSplitFixedRotationFixedFocalFixedPrincipalPointFixedPoint factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedRotationFixedFocalFixedPrincipalPointFixedPointPixelDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the weight_loss consts
   * PinholeSplitFixedRotationFixedFocalFixedPrincipalPointFixedPoint factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedRotationFixedFocalFixedPrincipalPointFixedPointWeightLossDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the weight_loss consts
   * PinholeSplitFixedRotationFixedFocalFixedPrincipalPointFixedPoint factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedRotationFixedFocalFixedPrincipalPointFixedPointWeightLossDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts
   * PinholeSplitFixedRotationFixedFocalFixedPrincipalPointFixedPoint factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedRotationFixedFocalFixedPrincipalPointFixedPointFocalDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts
   * PinholeSplitFixedRotationFixedFocalFixedPrincipalPointFixedPoint factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedRotationFixedFocalFixedPrincipalPointFixedPointFocalDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * PinholeSplitFixedRotationFixedFocalFixedPrincipalPointFixedPoint factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedRotationFixedFocalFixedPrincipalPointFixedPointPrincipalPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * PinholeSplitFixedRotationFixedFocalFixedPrincipalPointFixedPoint factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedRotationFixedFocalFixedPrincipalPointFixedPointPrincipalPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * PinholeSplitFixedRotationFixedFocalFixedPrincipalPointFixedPoint factor
   * from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedRotationFixedFocalFixedPrincipalPointFixedPointPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the point consts
   * PinholeSplitFixedRotationFixedFocalFixedPrincipalPointFixedPoint factor
   * from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitFixedRotationFixedFocalFixedPrincipalPointFixedPointPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of
   * PinholeSplitFixedRotationFixedFocalFixedPrincipalPointFixedPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeSplitFixedRotationFixedFocalFixedPrincipalPointFixedPointNum(
      size_t num);

  /**
   * Set the indices for the principal_point argument for the
   * PinholeSplitIntrinsicsPriorFixedFocal factor from host.
   */
  void SetPinholeSplitIntrinsicsPriorFixedFocalPrincipalPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the principal_point argument for the
   * PinholeSplitIntrinsicsPriorFixedFocal factor from device.
   */
  void SetPinholeSplitIntrinsicsPriorFixedFocalPrincipalPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the prior consts PinholeSplitIntrinsicsPriorFixedFocal
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitIntrinsicsPriorFixedFocalPriorDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the prior consts PinholeSplitIntrinsicsPriorFixedFocal
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitIntrinsicsPriorFixedFocalPriorDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the inv_std consts PinholeSplitIntrinsicsPriorFixedFocal
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitIntrinsicsPriorFixedFocalInvStdDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the inv_std consts PinholeSplitIntrinsicsPriorFixedFocal
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitIntrinsicsPriorFixedFocalInvStdDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts PinholeSplitIntrinsicsPriorFixedFocal
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitIntrinsicsPriorFixedFocalFocalDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the focal consts PinholeSplitIntrinsicsPriorFixedFocal
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void SetPinholeSplitIntrinsicsPriorFixedFocalFocalDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of PinholeSplitIntrinsicsPriorFixedFocal factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeSplitIntrinsicsPriorFixedFocalNum(size_t num);

  /**
   * Set the indices for the focal argument for the
   * PinholeSplitIntrinsicsPriorFixedPrincipalPoint factor from host.
   */
  void SetPinholeSplitIntrinsicsPriorFixedPrincipalPointFocalIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the focal argument for the
   * PinholeSplitIntrinsicsPriorFixedPrincipalPoint factor from device.
   */
  void SetPinholeSplitIntrinsicsPriorFixedPrincipalPointFocalIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the prior consts
   * PinholeSplitIntrinsicsPriorFixedPrincipalPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsPriorFixedPrincipalPointPriorDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the prior consts
   * PinholeSplitIntrinsicsPriorFixedPrincipalPoint factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsPriorFixedPrincipalPointPriorDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the inv_std consts
   * PinholeSplitIntrinsicsPriorFixedPrincipalPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsPriorFixedPrincipalPointInvStdDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the inv_std consts
   * PinholeSplitIntrinsicsPriorFixedPrincipalPoint factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsPriorFixedPrincipalPointInvStdDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * PinholeSplitIntrinsicsPriorFixedPrincipalPoint factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsPriorFixedPrincipalPointPrincipalPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the principal_point consts
   * PinholeSplitIntrinsicsPriorFixedPrincipalPoint factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsPriorFixedPrincipalPointPrincipalPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of PinholeSplitIntrinsicsPriorFixedPrincipalPoint
   * factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeSplitIntrinsicsPriorFixedPrincipalPointNum(size_t num);

  /**
   * Set the indices for the prev_principal_point argument for the
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocal factor from host.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalPrevPrincipalPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the prev_principal_point argument for the
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocal factor from device.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalPrevPrincipalPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the next_focal argument for the
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocal factor from host.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalNextFocalIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the next_focal argument for the
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocal factor from device.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalNextFocalIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the next_principal_point argument for the
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocal factor from host.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalNextPrincipalPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the next_principal_point argument for the
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocal factor from device.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalNextPrincipalPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the inv_std consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocal factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalInvStdDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the inv_std consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocal factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalInvStdDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the prev_focal consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocal factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalPrevFocalDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the prev_focal consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocal factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalPrevFocalDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of PinholeSplitIntrinsicsRandomWalkFixedPrevFocal
   * factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalNum(size_t num);

  /**
   * Set the indices for the prev_focal argument for the
   * PinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPoint factor from host.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointPrevFocalIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the prev_focal argument for the
   * PinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPoint factor from device.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointPrevFocalIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the next_focal argument for the
   * PinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPoint factor from host.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointNextFocalIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the next_focal argument for the
   * PinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPoint factor from device.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointNextFocalIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the next_principal_point argument for the
   * PinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPoint factor from host.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointNextPrincipalPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the next_principal_point argument for the
   * PinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPoint factor from device.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointNextPrincipalPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the inv_std consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointInvStdDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the inv_std consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointInvStdDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the prev_principal_point consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointPrevPrincipalPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the prev_principal_point consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointPrevPrincipalPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of
   * PinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointNum(
      size_t num);

  /**
   * Set the indices for the prev_focal argument for the
   * PinholeSplitIntrinsicsRandomWalkFixedNextFocal factor from host.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedNextFocalPrevFocalIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the prev_focal argument for the
   * PinholeSplitIntrinsicsRandomWalkFixedNextFocal factor from device.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedNextFocalPrevFocalIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the prev_principal_point argument for the
   * PinholeSplitIntrinsicsRandomWalkFixedNextFocal factor from host.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedNextFocalPrevPrincipalPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the prev_principal_point argument for the
   * PinholeSplitIntrinsicsRandomWalkFixedNextFocal factor from device.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedNextFocalPrevPrincipalPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the next_principal_point argument for the
   * PinholeSplitIntrinsicsRandomWalkFixedNextFocal factor from host.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedNextFocalNextPrincipalPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the next_principal_point argument for the
   * PinholeSplitIntrinsicsRandomWalkFixedNextFocal factor from device.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedNextFocalNextPrincipalPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the inv_std consts
   * PinholeSplitIntrinsicsRandomWalkFixedNextFocal factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedNextFocalInvStdDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the inv_std consts
   * PinholeSplitIntrinsicsRandomWalkFixedNextFocal factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedNextFocalInvStdDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the next_focal consts
   * PinholeSplitIntrinsicsRandomWalkFixedNextFocal factor from stacked host
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedNextFocalNextFocalDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the next_focal consts
   * PinholeSplitIntrinsicsRandomWalkFixedNextFocal factor from stacked device
   * data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedNextFocalNextFocalDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of PinholeSplitIntrinsicsRandomWalkFixedNextFocal
   * factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeSplitIntrinsicsRandomWalkFixedNextFocalNum(size_t num);

  /**
   * Set the indices for the prev_focal argument for the
   * PinholeSplitIntrinsicsRandomWalkFixedNextPrincipalPoint factor from host.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedNextPrincipalPointPrevFocalIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the prev_focal argument for the
   * PinholeSplitIntrinsicsRandomWalkFixedNextPrincipalPoint factor from device.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedNextPrincipalPointPrevFocalIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the prev_principal_point argument for the
   * PinholeSplitIntrinsicsRandomWalkFixedNextPrincipalPoint factor from host.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedNextPrincipalPointPrevPrincipalPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the prev_principal_point argument for the
   * PinholeSplitIntrinsicsRandomWalkFixedNextPrincipalPoint factor from device.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedNextPrincipalPointPrevPrincipalPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the next_focal argument for the
   * PinholeSplitIntrinsicsRandomWalkFixedNextPrincipalPoint factor from host.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedNextPrincipalPointNextFocalIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the next_focal argument for the
   * PinholeSplitIntrinsicsRandomWalkFixedNextPrincipalPoint factor from device.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedNextPrincipalPointNextFocalIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the inv_std consts
   * PinholeSplitIntrinsicsRandomWalkFixedNextPrincipalPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedNextPrincipalPointInvStdDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the inv_std consts
   * PinholeSplitIntrinsicsRandomWalkFixedNextPrincipalPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedNextPrincipalPointInvStdDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the next_principal_point consts
   * PinholeSplitIntrinsicsRandomWalkFixedNextPrincipalPoint factor from stacked
   * host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedNextPrincipalPointNextPrincipalPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the next_principal_point consts
   * PinholeSplitIntrinsicsRandomWalkFixedNextPrincipalPoint factor from stacked
   * device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedNextPrincipalPointNextPrincipalPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of
   * PinholeSplitIntrinsicsRandomWalkFixedNextPrincipalPoint factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeSplitIntrinsicsRandomWalkFixedNextPrincipalPointNum(
      size_t num);

  /**
   * Set the indices for the next_focal argument for the
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPoint
   * factor from host.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPointNextFocalIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the next_focal argument for the
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPoint
   * factor from device.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPointNextFocalIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the next_principal_point argument for the
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPoint
   * factor from host.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPointNextPrincipalPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the next_principal_point argument for the
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPoint
   * factor from device.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPointNextPrincipalPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the inv_std consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPoint
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPointInvStdDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the inv_std consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPointInvStdDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the prev_focal consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPoint
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPointPrevFocalDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the prev_focal consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPointPrevFocalDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the prev_principal_point consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPoint
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPointPrevPrincipalPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the prev_principal_point consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPointPrevPrincipalPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPoint
   * factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPointNum(
      size_t num);

  /**
   * Set the indices for the prev_principal_point argument for the
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextFocal factor from
   * host.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextFocalPrevPrincipalPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the prev_principal_point argument for the
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextFocal factor from
   * device.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextFocalPrevPrincipalPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the next_principal_point argument for the
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextFocal factor from
   * host.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextFocalNextPrincipalPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the next_principal_point argument for the
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextFocal factor from
   * device.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextFocalNextPrincipalPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the inv_std consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextFocal factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextFocalInvStdDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the inv_std consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextFocal factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextFocalInvStdDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the prev_focal consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextFocal factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextFocalPrevFocalDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the prev_focal consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextFocal factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextFocalPrevFocalDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the next_focal consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextFocal factor from
   * stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextFocalNextFocalDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the next_focal consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextFocal factor from
   * stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextFocalNextFocalDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextFocal factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextFocalNum(
      size_t num);

  /**
   * Set the indices for the prev_principal_point argument for the
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextPrincipalPoint
   * factor from host.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextPrincipalPointPrevPrincipalPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the prev_principal_point argument for the
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextPrincipalPoint
   * factor from device.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextPrincipalPointPrevPrincipalPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the next_focal argument for the
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextPrincipalPoint
   * factor from host.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextPrincipalPointNextFocalIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the next_focal argument for the
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextPrincipalPoint
   * factor from device.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextPrincipalPointNextFocalIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the inv_std consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextPrincipalPoint
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextPrincipalPointInvStdDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the inv_std consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextPrincipalPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextPrincipalPointInvStdDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the prev_focal consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextPrincipalPoint
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextPrincipalPointPrevFocalDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the prev_focal consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextPrincipalPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextPrincipalPointPrevFocalDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the next_principal_point consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextPrincipalPoint
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextPrincipalPointNextPrincipalPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the next_principal_point consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextPrincipalPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextPrincipalPointNextPrincipalPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextPrincipalPoint
   * factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextPrincipalPointNum(
      size_t num);

  /**
   * Set the indices for the prev_focal argument for the
   * PinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextFocal
   * factor from host.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextFocalPrevFocalIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the prev_focal argument for the
   * PinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextFocal
   * factor from device.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextFocalPrevFocalIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the next_principal_point argument for the
   * PinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextFocal
   * factor from host.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextFocalNextPrincipalPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the next_principal_point argument for the
   * PinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextFocal
   * factor from device.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextFocalNextPrincipalPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the inv_std consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextFocal
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextFocalInvStdDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the inv_std consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextFocal
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextFocalInvStdDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the prev_principal_point consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextFocal
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextFocalPrevPrincipalPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the prev_principal_point consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextFocal
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextFocalPrevPrincipalPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the next_focal consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextFocal
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextFocalNextFocalDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the next_focal consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextFocal
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextFocalNextFocalDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of
   * PinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextFocal
   * factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextFocalNum(
      size_t num);

  /**
   * Set the indices for the prev_focal argument for the
   * PinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextPrincipalPoint
   * factor from host.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextPrincipalPointPrevFocalIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the prev_focal argument for the
   * PinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextPrincipalPoint
   * factor from device.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextPrincipalPointPrevFocalIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the next_focal argument for the
   * PinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextPrincipalPoint
   * factor from host.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextPrincipalPointNextFocalIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the next_focal argument for the
   * PinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextPrincipalPoint
   * factor from device.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextPrincipalPointNextFocalIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the inv_std consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextPrincipalPoint
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextPrincipalPointInvStdDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the inv_std consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextPrincipalPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextPrincipalPointInvStdDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the prev_principal_point consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextPrincipalPoint
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextPrincipalPointPrevPrincipalPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the prev_principal_point consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextPrincipalPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextPrincipalPointPrevPrincipalPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the next_principal_point consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextPrincipalPoint
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextPrincipalPointNextPrincipalPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the next_principal_point consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextPrincipalPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextPrincipalPointNextPrincipalPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of
   * PinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextPrincipalPoint
   * factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextPrincipalPointNum(
      size_t num);

  /**
   * Set the indices for the prev_focal argument for the
   * PinholeSplitIntrinsicsRandomWalkFixedNextFocalFixedNextPrincipalPoint
   * factor from host.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedNextFocalFixedNextPrincipalPointPrevFocalIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the prev_focal argument for the
   * PinholeSplitIntrinsicsRandomWalkFixedNextFocalFixedNextPrincipalPoint
   * factor from device.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedNextFocalFixedNextPrincipalPointPrevFocalIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the prev_principal_point argument for the
   * PinholeSplitIntrinsicsRandomWalkFixedNextFocalFixedNextPrincipalPoint
   * factor from host.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedNextFocalFixedNextPrincipalPointPrevPrincipalPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the prev_principal_point argument for the
   * PinholeSplitIntrinsicsRandomWalkFixedNextFocalFixedNextPrincipalPoint
   * factor from device.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedNextFocalFixedNextPrincipalPointPrevPrincipalPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the inv_std consts
   * PinholeSplitIntrinsicsRandomWalkFixedNextFocalFixedNextPrincipalPoint
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedNextFocalFixedNextPrincipalPointInvStdDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the inv_std consts
   * PinholeSplitIntrinsicsRandomWalkFixedNextFocalFixedNextPrincipalPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedNextFocalFixedNextPrincipalPointInvStdDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the next_focal consts
   * PinholeSplitIntrinsicsRandomWalkFixedNextFocalFixedNextPrincipalPoint
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedNextFocalFixedNextPrincipalPointNextFocalDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the next_focal consts
   * PinholeSplitIntrinsicsRandomWalkFixedNextFocalFixedNextPrincipalPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedNextFocalFixedNextPrincipalPointNextFocalDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the next_principal_point consts
   * PinholeSplitIntrinsicsRandomWalkFixedNextFocalFixedNextPrincipalPoint
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedNextFocalFixedNextPrincipalPointNextPrincipalPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the next_principal_point consts
   * PinholeSplitIntrinsicsRandomWalkFixedNextFocalFixedNextPrincipalPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedNextFocalFixedNextPrincipalPointNextPrincipalPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of
   * PinholeSplitIntrinsicsRandomWalkFixedNextFocalFixedNextPrincipalPoint
   * factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedNextFocalFixedNextPrincipalPointNum(
      size_t num);

  /**
   * Set the indices for the next_principal_point argument for the
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPointFixedNextFocal
   * factor from host.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPointFixedNextFocalNextPrincipalPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the next_principal_point argument for the
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPointFixedNextFocal
   * factor from device.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPointFixedNextFocalNextPrincipalPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the inv_std consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPointFixedNextFocal
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPointFixedNextFocalInvStdDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the inv_std consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPointFixedNextFocal
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPointFixedNextFocalInvStdDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the prev_focal consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPointFixedNextFocal
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPointFixedNextFocalPrevFocalDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the prev_focal consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPointFixedNextFocal
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPointFixedNextFocalPrevFocalDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the prev_principal_point consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPointFixedNextFocal
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPointFixedNextFocalPrevPrincipalPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the prev_principal_point consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPointFixedNextFocal
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPointFixedNextFocalPrevPrincipalPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the next_focal consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPointFixedNextFocal
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPointFixedNextFocalNextFocalDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the next_focal consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPointFixedNextFocal
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPointFixedNextFocalNextFocalDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPointFixedNextFocal
   * factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPointFixedNextFocalNum(
      size_t num);

  /**
   * Set the indices for the next_focal argument for the
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPointFixedNextPrincipalPoint
   * factor from host.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPointFixedNextPrincipalPointNextFocalIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the next_focal argument for the
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPointFixedNextPrincipalPoint
   * factor from device.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPointFixedNextPrincipalPointNextFocalIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the inv_std consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPointFixedNextPrincipalPoint
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPointFixedNextPrincipalPointInvStdDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the inv_std consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPointFixedNextPrincipalPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPointFixedNextPrincipalPointInvStdDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the prev_focal consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPointFixedNextPrincipalPoint
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPointFixedNextPrincipalPointPrevFocalDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the prev_focal consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPointFixedNextPrincipalPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPointFixedNextPrincipalPointPrevFocalDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the prev_principal_point consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPointFixedNextPrincipalPoint
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPointFixedNextPrincipalPointPrevPrincipalPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the prev_principal_point consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPointFixedNextPrincipalPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPointFixedNextPrincipalPointPrevPrincipalPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the next_principal_point consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPointFixedNextPrincipalPoint
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPointFixedNextPrincipalPointNextPrincipalPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the next_principal_point consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPointFixedNextPrincipalPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPointFixedNextPrincipalPointNextPrincipalPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPointFixedNextPrincipalPoint
   * factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPointFixedNextPrincipalPointNum(
      size_t num);

  /**
   * Set the indices for the prev_principal_point argument for the
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextFocalFixedNextPrincipalPoint
   * factor from host.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextFocalFixedNextPrincipalPointPrevPrincipalPointIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the prev_principal_point argument for the
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextFocalFixedNextPrincipalPoint
   * factor from device.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextFocalFixedNextPrincipalPointPrevPrincipalPointIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the inv_std consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextFocalFixedNextPrincipalPoint
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextFocalFixedNextPrincipalPointInvStdDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the inv_std consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextFocalFixedNextPrincipalPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextFocalFixedNextPrincipalPointInvStdDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the prev_focal consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextFocalFixedNextPrincipalPoint
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextFocalFixedNextPrincipalPointPrevFocalDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the prev_focal consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextFocalFixedNextPrincipalPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextFocalFixedNextPrincipalPointPrevFocalDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the next_focal consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextFocalFixedNextPrincipalPoint
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextFocalFixedNextPrincipalPointNextFocalDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the next_focal consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextFocalFixedNextPrincipalPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextFocalFixedNextPrincipalPointNextFocalDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the next_principal_point consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextFocalFixedNextPrincipalPoint
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextFocalFixedNextPrincipalPointNextPrincipalPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the next_principal_point consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextFocalFixedNextPrincipalPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextFocalFixedNextPrincipalPointNextPrincipalPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of
   * PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextFocalFixedNextPrincipalPoint
   * factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextFocalFixedNextPrincipalPointNum(
      size_t num);

  /**
   * Set the indices for the prev_focal argument for the
   * PinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextFocalFixedNextPrincipalPoint
   * factor from host.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextFocalFixedNextPrincipalPointPrevFocalIndicesFromHost(
      const unsigned int* const indices, size_t num);

  /**
   * Set the indices for the prev_focal argument for the
   * PinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextFocalFixedNextPrincipalPoint
   * factor from device.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextFocalFixedNextPrincipalPointPrevFocalIndicesFromDevice(
      const unsigned int* const indices, size_t num);

  /**
   * Set the values for the inv_std consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextFocalFixedNextPrincipalPoint
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextFocalFixedNextPrincipalPointInvStdDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the inv_std consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextFocalFixedNextPrincipalPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextFocalFixedNextPrincipalPointInvStdDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the prev_principal_point consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextFocalFixedNextPrincipalPoint
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextFocalFixedNextPrincipalPointPrevPrincipalPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the prev_principal_point consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextFocalFixedNextPrincipalPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextFocalFixedNextPrincipalPointPrevPrincipalPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the next_focal consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextFocalFixedNextPrincipalPoint
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextFocalFixedNextPrincipalPointNextFocalDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the next_focal consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextFocalFixedNextPrincipalPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextFocalFixedNextPrincipalPointNextFocalDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the next_principal_point consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextFocalFixedNextPrincipalPoint
   * factor from stacked host data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextFocalFixedNextPrincipalPointNextPrincipalPointDataFromStackedHost(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the values for the next_principal_point consts
   * PinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextFocalFixedNextPrincipalPoint
   * factor from stacked device data.
   *
   * The offset can be used to start writing from a specific index.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextFocalFixedNextPrincipalPointNextPrincipalPointDataFromStackedDevice(
      const double* const data, size_t offset, size_t num);

  /**
   * Set the current number of
   * PinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextFocalFixedNextPrincipalPoint
   * factors.
   *
   * The value is set during initialization and this function is only needed if
   * you want to change the problem between optimization runs. This is work in
   * progress and can have performance impacts.
   */
  void
  SetPinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextFocalFixedNextPrincipalPointNum(
      size_t num);

 private:
  SolverParams<double> params_;
  uint8_t* origin_ptr_;
  size_t scratch_inout_size_;
  size_t allocation_size_;

  int solver_iter_;
  int pcg_iter_;

  bool indices_valid_;

  double pcg_r_0_norm2_;
  double pcg_r_kp1_norm2_;

  size_t DepthScale_num_;
  size_t DepthScale_num_max_;
  size_t PinholeCalib_num_;
  size_t PinholeCalib_num_max_;
  size_t PinholeFocal_num_;
  size_t PinholeFocal_num_max_;
  size_t PinholePose_num_;
  size_t PinholePose_num_max_;
  size_t PinholePrincipalPoint_num_;
  size_t PinholePrincipalPoint_num_max_;
  size_t PinholeTranslation_num_;
  size_t PinholeTranslation_num_max_;
  size_t Point_num_;
  size_t Point_num_max_;
  size_t SimpleRadialCalib_num_;
  size_t SimpleRadialCalib_num_max_;
  size_t SimpleRadialFocalAndDistortion_num_;
  size_t SimpleRadialFocalAndDistortion_num_max_;
  size_t SimpleRadialPose_num_;
  size_t SimpleRadialPose_num_max_;
  size_t SimpleRadialPrincipalPoint_num_;
  size_t SimpleRadialPrincipalPoint_num_max_;
  size_t simple_radial_num_;
  size_t simple_radial_num_max_;
  size_t simple_radial_fixed_pose_num_;
  size_t simple_radial_fixed_pose_num_max_;
  size_t simple_radial_fixed_point_num_;
  size_t simple_radial_fixed_point_num_max_;
  size_t simple_radial_fixed_pose_fixed_point_num_;
  size_t simple_radial_fixed_pose_fixed_point_num_max_;
  size_t pinhole_num_;
  size_t pinhole_num_max_;
  size_t pinhole_fixed_pose_num_;
  size_t pinhole_fixed_pose_num_max_;
  size_t pinhole_fixed_point_num_;
  size_t pinhole_fixed_point_num_max_;
  size_t pinhole_fixed_pose_fixed_point_num_;
  size_t pinhole_fixed_pose_fixed_point_num_max_;
  size_t pinhole_log_depth_num_;
  size_t pinhole_log_depth_num_max_;
  size_t pinhole_log_depth_fixed_pose_num_;
  size_t pinhole_log_depth_fixed_pose_num_max_;
  size_t pinhole_log_depth_fixed_scale_num_;
  size_t pinhole_log_depth_fixed_scale_num_max_;
  size_t pinhole_log_depth_fixed_point_num_;
  size_t pinhole_log_depth_fixed_point_num_max_;
  size_t pinhole_log_depth_fixed_pose_fixed_scale_num_;
  size_t pinhole_log_depth_fixed_pose_fixed_scale_num_max_;
  size_t pinhole_log_depth_fixed_pose_fixed_point_num_;
  size_t pinhole_log_depth_fixed_pose_fixed_point_num_max_;
  size_t pinhole_log_depth_fixed_scale_fixed_point_num_;
  size_t pinhole_log_depth_fixed_scale_fixed_point_num_max_;
  size_t pinhole_fixed_rotation_num_;
  size_t pinhole_fixed_rotation_num_max_;
  size_t pinhole_fixed_rotation_fixed_calib_num_;
  size_t pinhole_fixed_rotation_fixed_calib_num_max_;
  size_t pinhole_fixed_rotation_fixed_point_num_;
  size_t pinhole_fixed_rotation_fixed_point_num_max_;
  size_t pinhole_fixed_rotation_fixed_calib_fixed_point_num_;
  size_t pinhole_fixed_rotation_fixed_calib_fixed_point_num_max_;
  size_t pinhole_log_depth_fixed_rotation_num_;
  size_t pinhole_log_depth_fixed_rotation_num_max_;
  size_t pinhole_log_depth_fixed_rotation_fixed_scale_num_;
  size_t pinhole_log_depth_fixed_rotation_fixed_scale_num_max_;
  size_t pinhole_log_depth_fixed_rotation_fixed_point_num_;
  size_t pinhole_log_depth_fixed_rotation_fixed_point_num_max_;
  size_t pinhole_log_depth_fixed_rotation_fixed_scale_fixed_point_num_;
  size_t pinhole_log_depth_fixed_rotation_fixed_scale_fixed_point_num_max_;
  size_t pinhole_intrinsics_prior_num_;
  size_t pinhole_intrinsics_prior_num_max_;
  size_t pinhole_intrinsics_random_walk_num_;
  size_t pinhole_intrinsics_random_walk_num_max_;
  size_t scale_prior_num_;
  size_t scale_prior_num_max_;
  size_t simple_radial_split_fixed_focal_and_distortion_num_;
  size_t simple_radial_split_fixed_focal_and_distortion_num_max_;
  size_t simple_radial_split_fixed_principal_point_num_;
  size_t simple_radial_split_fixed_principal_point_num_max_;
  size_t simple_radial_split_fixed_pose_fixed_focal_and_distortion_num_;
  size_t simple_radial_split_fixed_pose_fixed_focal_and_distortion_num_max_;
  size_t simple_radial_split_fixed_pose_fixed_principal_point_num_;
  size_t simple_radial_split_fixed_pose_fixed_principal_point_num_max_;
  size_t
      simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_num_;
  size_t
      simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_num_max_;
  size_t simple_radial_split_fixed_focal_and_distortion_fixed_point_num_;
  size_t simple_radial_split_fixed_focal_and_distortion_fixed_point_num_max_;
  size_t simple_radial_split_fixed_principal_point_fixed_point_num_;
  size_t simple_radial_split_fixed_principal_point_fixed_point_num_max_;
  size_t
      simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_principal_point_num_;
  size_t
      simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_principal_point_num_max_;
  size_t
      simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_point_num_;
  size_t
      simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_point_num_max_;
  size_t simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_;
  size_t
      simple_radial_split_fixed_pose_fixed_principal_point_fixed_point_num_max_;
  size_t
      simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_fixed_point_num_;
  size_t
      simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_fixed_point_num_max_;
  size_t pinhole_split_fixed_focal_num_;
  size_t pinhole_split_fixed_focal_num_max_;
  size_t pinhole_split_fixed_principal_point_num_;
  size_t pinhole_split_fixed_principal_point_num_max_;
  size_t pinhole_split_fixed_pose_fixed_focal_num_;
  size_t pinhole_split_fixed_pose_fixed_focal_num_max_;
  size_t pinhole_split_fixed_pose_fixed_principal_point_num_;
  size_t pinhole_split_fixed_pose_fixed_principal_point_num_max_;
  size_t pinhole_split_fixed_focal_fixed_principal_point_num_;
  size_t pinhole_split_fixed_focal_fixed_principal_point_num_max_;
  size_t pinhole_split_fixed_focal_fixed_point_num_;
  size_t pinhole_split_fixed_focal_fixed_point_num_max_;
  size_t pinhole_split_fixed_principal_point_fixed_point_num_;
  size_t pinhole_split_fixed_principal_point_fixed_point_num_max_;
  size_t pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_;
  size_t pinhole_split_fixed_pose_fixed_focal_fixed_principal_point_num_max_;
  size_t pinhole_split_fixed_pose_fixed_focal_fixed_point_num_;
  size_t pinhole_split_fixed_pose_fixed_focal_fixed_point_num_max_;
  size_t pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_;
  size_t pinhole_split_fixed_pose_fixed_principal_point_fixed_point_num_max_;
  size_t pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_;
  size_t pinhole_split_fixed_focal_fixed_principal_point_fixed_point_num_max_;
  size_t pinhole_split_fixed_rotation_fixed_focal_num_;
  size_t pinhole_split_fixed_rotation_fixed_focal_num_max_;
  size_t pinhole_split_fixed_rotation_fixed_principal_point_num_;
  size_t pinhole_split_fixed_rotation_fixed_principal_point_num_max_;
  size_t pinhole_split_fixed_rotation_fixed_focal_fixed_principal_point_num_;
  size_t
      pinhole_split_fixed_rotation_fixed_focal_fixed_principal_point_num_max_;
  size_t pinhole_split_fixed_rotation_fixed_focal_fixed_point_num_;
  size_t pinhole_split_fixed_rotation_fixed_focal_fixed_point_num_max_;
  size_t pinhole_split_fixed_rotation_fixed_principal_point_fixed_point_num_;
  size_t
      pinhole_split_fixed_rotation_fixed_principal_point_fixed_point_num_max_;
  size_t
      pinhole_split_fixed_rotation_fixed_focal_fixed_principal_point_fixed_point_num_;
  size_t
      pinhole_split_fixed_rotation_fixed_focal_fixed_principal_point_fixed_point_num_max_;
  size_t pinhole_split_intrinsics_prior_fixed_focal_num_;
  size_t pinhole_split_intrinsics_prior_fixed_focal_num_max_;
  size_t pinhole_split_intrinsics_prior_fixed_principal_point_num_;
  size_t pinhole_split_intrinsics_prior_fixed_principal_point_num_max_;
  size_t pinhole_split_intrinsics_random_walk_fixed_prev_focal_num_;
  size_t pinhole_split_intrinsics_random_walk_fixed_prev_focal_num_max_;
  size_t pinhole_split_intrinsics_random_walk_fixed_prev_principal_point_num_;
  size_t
      pinhole_split_intrinsics_random_walk_fixed_prev_principal_point_num_max_;
  size_t pinhole_split_intrinsics_random_walk_fixed_next_focal_num_;
  size_t pinhole_split_intrinsics_random_walk_fixed_next_focal_num_max_;
  size_t pinhole_split_intrinsics_random_walk_fixed_next_principal_point_num_;
  size_t
      pinhole_split_intrinsics_random_walk_fixed_next_principal_point_num_max_;
  size_t
      pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_prev_principal_point_num_;
  size_t
      pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_prev_principal_point_num_max_;
  size_t
      pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_next_focal_num_;
  size_t
      pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_next_focal_num_max_;
  size_t
      pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_next_principal_point_num_;
  size_t
      pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_next_principal_point_num_max_;
  size_t
      pinhole_split_intrinsics_random_walk_fixed_prev_principal_point_fixed_next_focal_num_;
  size_t
      pinhole_split_intrinsics_random_walk_fixed_prev_principal_point_fixed_next_focal_num_max_;
  size_t
      pinhole_split_intrinsics_random_walk_fixed_prev_principal_point_fixed_next_principal_point_num_;
  size_t
      pinhole_split_intrinsics_random_walk_fixed_prev_principal_point_fixed_next_principal_point_num_max_;
  size_t
      pinhole_split_intrinsics_random_walk_fixed_next_focal_fixed_next_principal_point_num_;
  size_t
      pinhole_split_intrinsics_random_walk_fixed_next_focal_fixed_next_principal_point_num_max_;
  size_t
      pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_prev_principal_point_fixed_next_focal_num_;
  size_t
      pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_prev_principal_point_fixed_next_focal_num_max_;
  size_t
      pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_prev_principal_point_fixed_next_principal_point_num_;
  size_t
      pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_prev_principal_point_fixed_next_principal_point_num_max_;
  size_t
      pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_next_focal_fixed_next_principal_point_num_;
  size_t
      pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_next_focal_fixed_next_principal_point_num_max_;
  size_t
      pinhole_split_intrinsics_random_walk_fixed_prev_principal_point_fixed_next_focal_fixed_next_principal_point_num_;
  size_t
      pinhole_split_intrinsics_random_walk_fixed_prev_principal_point_fixed_next_focal_fixed_next_principal_point_num_max_;

  size_t get_nbytes();
  double LinearizeFirst();
  void Linearize();
  double DoResJacFirst();
  void DoResJac();
  void DoNormalize();
  void DoJtjpDirect();
  void DoAlphaFirst();
  void DoAlpha();
  void DoUpdateStepFirst();
  void DoUpdateStep();
  void DoUpdateRFirst();
  void DoUpdateR();
  double DoRetractScore();
  void DoBeta();
  void DoUpdateP();
  void DoUpdateMp();
  double GetPredDecrease();

  double* marker__start_;
  double* nodes__DepthScale__storage_current_;
  double* nodes__DepthScale__storage_check_;
  double* nodes__DepthScale__storage_new_best_;
  double* nodes__PinholeCalib__storage_current_;
  double* nodes__PinholeCalib__storage_check_;
  double* nodes__PinholeCalib__storage_new_best_;
  double* nodes__PinholeFocal__storage_current_;
  double* nodes__PinholeFocal__storage_check_;
  double* nodes__PinholeFocal__storage_new_best_;
  double* nodes__PinholePose__storage_current_;
  double* nodes__PinholePose__storage_check_;
  double* nodes__PinholePose__storage_new_best_;
  double* nodes__PinholePrincipalPoint__storage_current_;
  double* nodes__PinholePrincipalPoint__storage_check_;
  double* nodes__PinholePrincipalPoint__storage_new_best_;
  double* nodes__PinholeTranslation__storage_current_;
  double* nodes__PinholeTranslation__storage_check_;
  double* nodes__PinholeTranslation__storage_new_best_;
  double* nodes__Point__storage_current_;
  double* nodes__Point__storage_check_;
  double* nodes__Point__storage_new_best_;
  double* nodes__SimpleRadialCalib__storage_current_;
  double* nodes__SimpleRadialCalib__storage_check_;
  double* nodes__SimpleRadialCalib__storage_new_best_;
  double* nodes__SimpleRadialFocalAndDistortion__storage_current_;
  double* nodes__SimpleRadialFocalAndDistortion__storage_check_;
  double* nodes__SimpleRadialFocalAndDistortion__storage_new_best_;
  double* nodes__SimpleRadialPose__storage_current_;
  double* nodes__SimpleRadialPose__storage_check_;
  double* nodes__SimpleRadialPose__storage_new_best_;
  double* nodes__SimpleRadialPrincipalPoint__storage_current_;
  double* nodes__SimpleRadialPrincipalPoint__storage_check_;
  double* nodes__SimpleRadialPrincipalPoint__storage_new_best_;
  SharedIndex* facs__simple_radial__args__pose__idx_shared_;
  SharedIndex* facs__simple_radial__args__calib__idx_shared_;
  SharedIndex* facs__simple_radial__args__point__idx_shared_;
  double* facs__simple_radial__args__pixel__data_;
  SharedIndex* facs__simple_radial_fixed_pose__args__calib__idx_shared_;
  SharedIndex* facs__simple_radial_fixed_pose__args__point__idx_shared_;
  double* facs__simple_radial_fixed_pose__args__pixel__data_;
  double* facs__simple_radial_fixed_pose__args__pose__data_;
  SharedIndex* facs__simple_radial_fixed_point__args__pose__idx_shared_;
  SharedIndex* facs__simple_radial_fixed_point__args__calib__idx_shared_;
  double* facs__simple_radial_fixed_point__args__pixel__data_;
  double* facs__simple_radial_fixed_point__args__point__data_;
  SharedIndex*
      facs__simple_radial_fixed_pose_fixed_point__args__calib__idx_shared_;
  double* facs__simple_radial_fixed_pose_fixed_point__args__pixel__data_;
  double* facs__simple_radial_fixed_pose_fixed_point__args__pose__data_;
  double* facs__simple_radial_fixed_pose_fixed_point__args__point__data_;
  SharedIndex* facs__pinhole__args__pose__idx_shared_;
  SharedIndex* facs__pinhole__args__calib__idx_shared_;
  SharedIndex* facs__pinhole__args__point__idx_shared_;
  double* facs__pinhole__args__pixel__data_;
  double* facs__pinhole__args__weight_loss__data_;
  SharedIndex* facs__pinhole_fixed_pose__args__calib__idx_shared_;
  SharedIndex* facs__pinhole_fixed_pose__args__point__idx_shared_;
  double* facs__pinhole_fixed_pose__args__pixel__data_;
  double* facs__pinhole_fixed_pose__args__weight_loss__data_;
  double* facs__pinhole_fixed_pose__args__pose__data_;
  SharedIndex* facs__pinhole_fixed_point__args__pose__idx_shared_;
  SharedIndex* facs__pinhole_fixed_point__args__calib__idx_shared_;
  double* facs__pinhole_fixed_point__args__pixel__data_;
  double* facs__pinhole_fixed_point__args__weight_loss__data_;
  double* facs__pinhole_fixed_point__args__point__data_;
  SharedIndex* facs__pinhole_fixed_pose_fixed_point__args__calib__idx_shared_;
  double* facs__pinhole_fixed_pose_fixed_point__args__pixel__data_;
  double* facs__pinhole_fixed_pose_fixed_point__args__weight_loss__data_;
  double* facs__pinhole_fixed_pose_fixed_point__args__pose__data_;
  double* facs__pinhole_fixed_pose_fixed_point__args__point__data_;
  SharedIndex* facs__pinhole_log_depth__args__pose__idx_shared_;
  SharedIndex* facs__pinhole_log_depth__args__scale__idx_shared_;
  SharedIndex* facs__pinhole_log_depth__args__point__idx_shared_;
  double* facs__pinhole_log_depth__args__log_depth__data_;
  double* facs__pinhole_log_depth__args__loss__data_;
  SharedIndex* facs__pinhole_log_depth_fixed_pose__args__scale__idx_shared_;
  SharedIndex* facs__pinhole_log_depth_fixed_pose__args__point__idx_shared_;
  double* facs__pinhole_log_depth_fixed_pose__args__log_depth__data_;
  double* facs__pinhole_log_depth_fixed_pose__args__loss__data_;
  double* facs__pinhole_log_depth_fixed_pose__args__pose__data_;
  SharedIndex* facs__pinhole_log_depth_fixed_scale__args__pose__idx_shared_;
  SharedIndex* facs__pinhole_log_depth_fixed_scale__args__point__idx_shared_;
  double* facs__pinhole_log_depth_fixed_scale__args__log_depth__data_;
  double* facs__pinhole_log_depth_fixed_scale__args__loss__data_;
  double* facs__pinhole_log_depth_fixed_scale__args__scale__data_;
  SharedIndex* facs__pinhole_log_depth_fixed_point__args__pose__idx_shared_;
  SharedIndex* facs__pinhole_log_depth_fixed_point__args__scale__idx_shared_;
  double* facs__pinhole_log_depth_fixed_point__args__log_depth__data_;
  double* facs__pinhole_log_depth_fixed_point__args__loss__data_;
  double* facs__pinhole_log_depth_fixed_point__args__point__data_;
  SharedIndex*
      facs__pinhole_log_depth_fixed_pose_fixed_scale__args__point__idx_shared_;
  double*
      facs__pinhole_log_depth_fixed_pose_fixed_scale__args__log_depth__data_;
  double* facs__pinhole_log_depth_fixed_pose_fixed_scale__args__loss__data_;
  double* facs__pinhole_log_depth_fixed_pose_fixed_scale__args__pose__data_;
  double* facs__pinhole_log_depth_fixed_pose_fixed_scale__args__scale__data_;
  SharedIndex*
      facs__pinhole_log_depth_fixed_pose_fixed_point__args__scale__idx_shared_;
  double*
      facs__pinhole_log_depth_fixed_pose_fixed_point__args__log_depth__data_;
  double* facs__pinhole_log_depth_fixed_pose_fixed_point__args__loss__data_;
  double* facs__pinhole_log_depth_fixed_pose_fixed_point__args__pose__data_;
  double* facs__pinhole_log_depth_fixed_pose_fixed_point__args__point__data_;
  SharedIndex*
      facs__pinhole_log_depth_fixed_scale_fixed_point__args__pose__idx_shared_;
  double*
      facs__pinhole_log_depth_fixed_scale_fixed_point__args__log_depth__data_;
  double* facs__pinhole_log_depth_fixed_scale_fixed_point__args__loss__data_;
  double* facs__pinhole_log_depth_fixed_scale_fixed_point__args__scale__data_;
  double* facs__pinhole_log_depth_fixed_scale_fixed_point__args__point__data_;
  double* facs__pinhole_fixed_rotation__args__rotation__data_;
  SharedIndex* facs__pinhole_fixed_rotation__args__translation__idx_shared_;
  SharedIndex* facs__pinhole_fixed_rotation__args__calib__idx_shared_;
  SharedIndex* facs__pinhole_fixed_rotation__args__point__idx_shared_;
  double* facs__pinhole_fixed_rotation__args__pixel__data_;
  double* facs__pinhole_fixed_rotation__args__weight_loss__data_;
  double* facs__pinhole_fixed_rotation_fixed_calib__args__rotation__data_;
  SharedIndex*
      facs__pinhole_fixed_rotation_fixed_calib__args__translation__idx_shared_;
  SharedIndex*
      facs__pinhole_fixed_rotation_fixed_calib__args__point__idx_shared_;
  double* facs__pinhole_fixed_rotation_fixed_calib__args__pixel__data_;
  double* facs__pinhole_fixed_rotation_fixed_calib__args__weight_loss__data_;
  double* facs__pinhole_fixed_rotation_fixed_calib__args__calib__data_;
  double* facs__pinhole_fixed_rotation_fixed_point__args__rotation__data_;
  SharedIndex*
      facs__pinhole_fixed_rotation_fixed_point__args__translation__idx_shared_;
  SharedIndex*
      facs__pinhole_fixed_rotation_fixed_point__args__calib__idx_shared_;
  double* facs__pinhole_fixed_rotation_fixed_point__args__pixel__data_;
  double* facs__pinhole_fixed_rotation_fixed_point__args__weight_loss__data_;
  double* facs__pinhole_fixed_rotation_fixed_point__args__point__data_;
  double*
      facs__pinhole_fixed_rotation_fixed_calib_fixed_point__args__rotation__data_;
  SharedIndex*
      facs__pinhole_fixed_rotation_fixed_calib_fixed_point__args__translation__idx_shared_;
  double*
      facs__pinhole_fixed_rotation_fixed_calib_fixed_point__args__pixel__data_;
  double*
      facs__pinhole_fixed_rotation_fixed_calib_fixed_point__args__weight_loss__data_;
  double*
      facs__pinhole_fixed_rotation_fixed_calib_fixed_point__args__calib__data_;
  double*
      facs__pinhole_fixed_rotation_fixed_calib_fixed_point__args__point__data_;
  double* facs__pinhole_log_depth_fixed_rotation__args__rotation__data_;
  SharedIndex*
      facs__pinhole_log_depth_fixed_rotation__args__translation__idx_shared_;
  SharedIndex* facs__pinhole_log_depth_fixed_rotation__args__scale__idx_shared_;
  SharedIndex* facs__pinhole_log_depth_fixed_rotation__args__point__idx_shared_;
  double* facs__pinhole_log_depth_fixed_rotation__args__log_depth__data_;
  double* facs__pinhole_log_depth_fixed_rotation__args__loss__data_;
  double*
      facs__pinhole_log_depth_fixed_rotation_fixed_scale__args__rotation__data_;
  SharedIndex*
      facs__pinhole_log_depth_fixed_rotation_fixed_scale__args__translation__idx_shared_;
  SharedIndex*
      facs__pinhole_log_depth_fixed_rotation_fixed_scale__args__point__idx_shared_;
  double*
      facs__pinhole_log_depth_fixed_rotation_fixed_scale__args__log_depth__data_;
  double* facs__pinhole_log_depth_fixed_rotation_fixed_scale__args__loss__data_;
  double*
      facs__pinhole_log_depth_fixed_rotation_fixed_scale__args__scale__data_;
  double*
      facs__pinhole_log_depth_fixed_rotation_fixed_point__args__rotation__data_;
  SharedIndex*
      facs__pinhole_log_depth_fixed_rotation_fixed_point__args__translation__idx_shared_;
  SharedIndex*
      facs__pinhole_log_depth_fixed_rotation_fixed_point__args__scale__idx_shared_;
  double*
      facs__pinhole_log_depth_fixed_rotation_fixed_point__args__log_depth__data_;
  double* facs__pinhole_log_depth_fixed_rotation_fixed_point__args__loss__data_;
  double*
      facs__pinhole_log_depth_fixed_rotation_fixed_point__args__point__data_;
  double*
      facs__pinhole_log_depth_fixed_rotation_fixed_scale_fixed_point__args__rotation__data_;
  SharedIndex*
      facs__pinhole_log_depth_fixed_rotation_fixed_scale_fixed_point__args__translation__idx_shared_;
  double*
      facs__pinhole_log_depth_fixed_rotation_fixed_scale_fixed_point__args__log_depth__data_;
  double*
      facs__pinhole_log_depth_fixed_rotation_fixed_scale_fixed_point__args__loss__data_;
  double*
      facs__pinhole_log_depth_fixed_rotation_fixed_scale_fixed_point__args__scale__data_;
  double*
      facs__pinhole_log_depth_fixed_rotation_fixed_scale_fixed_point__args__point__data_;
  SharedIndex* facs__pinhole_intrinsics_prior__args__calib__idx_shared_;
  double* facs__pinhole_intrinsics_prior__args__prior__data_;
  double* facs__pinhole_intrinsics_prior__args__inv_std__data_;
  SharedIndex*
      facs__pinhole_intrinsics_random_walk__args__prev_calib__idx_shared_;
  SharedIndex*
      facs__pinhole_intrinsics_random_walk__args__next_calib__idx_shared_;
  double* facs__pinhole_intrinsics_random_walk__args__inv_std__data_;
  SharedIndex* facs__scale_prior__args__scale__idx_shared_;
  double* facs__scale_prior__args__inv_std__data_;
  double* facs__scale_prior__args__loss__data_;
  SharedIndex*
      facs__simple_radial_split_fixed_focal_and_distortion__args__pose__idx_shared_;
  SharedIndex*
      facs__simple_radial_split_fixed_focal_and_distortion__args__principal_point__idx_shared_;
  SharedIndex*
      facs__simple_radial_split_fixed_focal_and_distortion__args__point__idx_shared_;
  double*
      facs__simple_radial_split_fixed_focal_and_distortion__args__pixel__data_;
  double*
      facs__simple_radial_split_fixed_focal_and_distortion__args__focal_and_distortion__data_;
  SharedIndex*
      facs__simple_radial_split_fixed_principal_point__args__pose__idx_shared_;
  SharedIndex*
      facs__simple_radial_split_fixed_principal_point__args__focal_and_distortion__idx_shared_;
  SharedIndex*
      facs__simple_radial_split_fixed_principal_point__args__point__idx_shared_;
  double* facs__simple_radial_split_fixed_principal_point__args__pixel__data_;
  double*
      facs__simple_radial_split_fixed_principal_point__args__principal_point__data_;
  SharedIndex*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_distortion__args__principal_point__idx_shared_;
  SharedIndex*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_distortion__args__point__idx_shared_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_distortion__args__pixel__data_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_distortion__args__pose__data_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_distortion__args__focal_and_distortion__data_;
  SharedIndex*
      facs__simple_radial_split_fixed_pose_fixed_principal_point__args__focal_and_distortion__idx_shared_;
  SharedIndex*
      facs__simple_radial_split_fixed_pose_fixed_principal_point__args__point__idx_shared_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_principal_point__args__pixel__data_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_principal_point__args__pose__data_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_principal_point__args__principal_point__data_;
  SharedIndex*
      facs__simple_radial_split_fixed_focal_and_distortion_fixed_principal_point__args__pose__idx_shared_;
  SharedIndex*
      facs__simple_radial_split_fixed_focal_and_distortion_fixed_principal_point__args__point__idx_shared_;
  double*
      facs__simple_radial_split_fixed_focal_and_distortion_fixed_principal_point__args__pixel__data_;
  double*
      facs__simple_radial_split_fixed_focal_and_distortion_fixed_principal_point__args__focal_and_distortion__data_;
  double*
      facs__simple_radial_split_fixed_focal_and_distortion_fixed_principal_point__args__principal_point__data_;
  SharedIndex*
      facs__simple_radial_split_fixed_focal_and_distortion_fixed_point__args__pose__idx_shared_;
  SharedIndex*
      facs__simple_radial_split_fixed_focal_and_distortion_fixed_point__args__principal_point__idx_shared_;
  double*
      facs__simple_radial_split_fixed_focal_and_distortion_fixed_point__args__pixel__data_;
  double*
      facs__simple_radial_split_fixed_focal_and_distortion_fixed_point__args__focal_and_distortion__data_;
  double*
      facs__simple_radial_split_fixed_focal_and_distortion_fixed_point__args__point__data_;
  SharedIndex*
      facs__simple_radial_split_fixed_principal_point_fixed_point__args__pose__idx_shared_;
  SharedIndex*
      facs__simple_radial_split_fixed_principal_point_fixed_point__args__focal_and_distortion__idx_shared_;
  double*
      facs__simple_radial_split_fixed_principal_point_fixed_point__args__pixel__data_;
  double*
      facs__simple_radial_split_fixed_principal_point_fixed_point__args__principal_point__data_;
  double*
      facs__simple_radial_split_fixed_principal_point_fixed_point__args__point__data_;
  SharedIndex*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_principal_point__args__point__idx_shared_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_principal_point__args__pixel__data_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_principal_point__args__pose__data_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_principal_point__args__focal_and_distortion__data_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_principal_point__args__principal_point__data_;
  SharedIndex*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_point__args__principal_point__idx_shared_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_point__args__pixel__data_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_point__args__pose__data_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_point__args__focal_and_distortion__data_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_point__args__point__data_;
  SharedIndex*
      facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__args__focal_and_distortion__idx_shared_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__args__pixel__data_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__args__pose__data_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__args__principal_point__data_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__args__point__data_;
  SharedIndex*
      facs__simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_fixed_point__args__pose__idx_shared_;
  double*
      facs__simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_fixed_point__args__pixel__data_;
  double*
      facs__simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_fixed_point__args__focal_and_distortion__data_;
  double*
      facs__simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_fixed_point__args__principal_point__data_;
  double*
      facs__simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_fixed_point__args__point__data_;
  SharedIndex* facs__pinhole_split_fixed_focal__args__pose__idx_shared_;
  SharedIndex*
      facs__pinhole_split_fixed_focal__args__principal_point__idx_shared_;
  SharedIndex* facs__pinhole_split_fixed_focal__args__point__idx_shared_;
  double* facs__pinhole_split_fixed_focal__args__pixel__data_;
  double* facs__pinhole_split_fixed_focal__args__weight_loss__data_;
  double* facs__pinhole_split_fixed_focal__args__focal__data_;
  SharedIndex*
      facs__pinhole_split_fixed_principal_point__args__pose__idx_shared_;
  SharedIndex*
      facs__pinhole_split_fixed_principal_point__args__focal__idx_shared_;
  SharedIndex*
      facs__pinhole_split_fixed_principal_point__args__point__idx_shared_;
  double* facs__pinhole_split_fixed_principal_point__args__pixel__data_;
  double* facs__pinhole_split_fixed_principal_point__args__weight_loss__data_;
  double*
      facs__pinhole_split_fixed_principal_point__args__principal_point__data_;
  SharedIndex*
      facs__pinhole_split_fixed_pose_fixed_focal__args__principal_point__idx_shared_;
  SharedIndex*
      facs__pinhole_split_fixed_pose_fixed_focal__args__point__idx_shared_;
  double* facs__pinhole_split_fixed_pose_fixed_focal__args__pixel__data_;
  double* facs__pinhole_split_fixed_pose_fixed_focal__args__weight_loss__data_;
  double* facs__pinhole_split_fixed_pose_fixed_focal__args__pose__data_;
  double* facs__pinhole_split_fixed_pose_fixed_focal__args__focal__data_;
  SharedIndex*
      facs__pinhole_split_fixed_pose_fixed_principal_point__args__focal__idx_shared_;
  SharedIndex*
      facs__pinhole_split_fixed_pose_fixed_principal_point__args__point__idx_shared_;
  double*
      facs__pinhole_split_fixed_pose_fixed_principal_point__args__pixel__data_;
  double*
      facs__pinhole_split_fixed_pose_fixed_principal_point__args__weight_loss__data_;
  double*
      facs__pinhole_split_fixed_pose_fixed_principal_point__args__pose__data_;
  double*
      facs__pinhole_split_fixed_pose_fixed_principal_point__args__principal_point__data_;
  SharedIndex*
      facs__pinhole_split_fixed_focal_fixed_principal_point__args__pose__idx_shared_;
  SharedIndex*
      facs__pinhole_split_fixed_focal_fixed_principal_point__args__point__idx_shared_;
  double*
      facs__pinhole_split_fixed_focal_fixed_principal_point__args__pixel__data_;
  double*
      facs__pinhole_split_fixed_focal_fixed_principal_point__args__weight_loss__data_;
  double*
      facs__pinhole_split_fixed_focal_fixed_principal_point__args__focal__data_;
  double*
      facs__pinhole_split_fixed_focal_fixed_principal_point__args__principal_point__data_;
  SharedIndex*
      facs__pinhole_split_fixed_focal_fixed_point__args__pose__idx_shared_;
  SharedIndex*
      facs__pinhole_split_fixed_focal_fixed_point__args__principal_point__idx_shared_;
  double* facs__pinhole_split_fixed_focal_fixed_point__args__pixel__data_;
  double* facs__pinhole_split_fixed_focal_fixed_point__args__weight_loss__data_;
  double* facs__pinhole_split_fixed_focal_fixed_point__args__focal__data_;
  double* facs__pinhole_split_fixed_focal_fixed_point__args__point__data_;
  SharedIndex*
      facs__pinhole_split_fixed_principal_point_fixed_point__args__pose__idx_shared_;
  SharedIndex*
      facs__pinhole_split_fixed_principal_point_fixed_point__args__focal__idx_shared_;
  double*
      facs__pinhole_split_fixed_principal_point_fixed_point__args__pixel__data_;
  double*
      facs__pinhole_split_fixed_principal_point_fixed_point__args__weight_loss__data_;
  double*
      facs__pinhole_split_fixed_principal_point_fixed_point__args__principal_point__data_;
  double*
      facs__pinhole_split_fixed_principal_point_fixed_point__args__point__data_;
  SharedIndex*
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__args__point__idx_shared_;
  double*
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__args__pixel__data_;
  double*
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__args__weight_loss__data_;
  double*
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__args__pose__data_;
  double*
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__args__focal__data_;
  double*
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__args__principal_point__data_;
  SharedIndex*
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__args__principal_point__idx_shared_;
  double*
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__args__pixel__data_;
  double*
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__args__weight_loss__data_;
  double*
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__args__pose__data_;
  double*
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__args__focal__data_;
  double*
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__args__point__data_;
  SharedIndex*
      facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__args__focal__idx_shared_;
  double*
      facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__args__pixel__data_;
  double*
      facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__args__weight_loss__data_;
  double*
      facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__args__pose__data_;
  double*
      facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__args__principal_point__data_;
  double*
      facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__args__point__data_;
  SharedIndex*
      facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__args__pose__idx_shared_;
  double*
      facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__args__pixel__data_;
  double*
      facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__args__weight_loss__data_;
  double*
      facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__args__focal__data_;
  double*
      facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__args__principal_point__data_;
  double*
      facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__args__point__data_;
  double* facs__pinhole_split_fixed_rotation_fixed_focal__args__rotation__data_;
  SharedIndex*
      facs__pinhole_split_fixed_rotation_fixed_focal__args__translation__idx_shared_;
  SharedIndex*
      facs__pinhole_split_fixed_rotation_fixed_focal__args__principal_point__idx_shared_;
  SharedIndex*
      facs__pinhole_split_fixed_rotation_fixed_focal__args__point__idx_shared_;
  double* facs__pinhole_split_fixed_rotation_fixed_focal__args__pixel__data_;
  double*
      facs__pinhole_split_fixed_rotation_fixed_focal__args__weight_loss__data_;
  double* facs__pinhole_split_fixed_rotation_fixed_focal__args__focal__data_;
  double*
      facs__pinhole_split_fixed_rotation_fixed_principal_point__args__rotation__data_;
  SharedIndex*
      facs__pinhole_split_fixed_rotation_fixed_principal_point__args__translation__idx_shared_;
  SharedIndex*
      facs__pinhole_split_fixed_rotation_fixed_principal_point__args__focal__idx_shared_;
  SharedIndex*
      facs__pinhole_split_fixed_rotation_fixed_principal_point__args__point__idx_shared_;
  double*
      facs__pinhole_split_fixed_rotation_fixed_principal_point__args__pixel__data_;
  double*
      facs__pinhole_split_fixed_rotation_fixed_principal_point__args__weight_loss__data_;
  double*
      facs__pinhole_split_fixed_rotation_fixed_principal_point__args__principal_point__data_;
  double*
      facs__pinhole_split_fixed_rotation_fixed_focal_fixed_principal_point__args__rotation__data_;
  SharedIndex*
      facs__pinhole_split_fixed_rotation_fixed_focal_fixed_principal_point__args__translation__idx_shared_;
  SharedIndex*
      facs__pinhole_split_fixed_rotation_fixed_focal_fixed_principal_point__args__point__idx_shared_;
  double*
      facs__pinhole_split_fixed_rotation_fixed_focal_fixed_principal_point__args__pixel__data_;
  double*
      facs__pinhole_split_fixed_rotation_fixed_focal_fixed_principal_point__args__weight_loss__data_;
  double*
      facs__pinhole_split_fixed_rotation_fixed_focal_fixed_principal_point__args__focal__data_;
  double*
      facs__pinhole_split_fixed_rotation_fixed_focal_fixed_principal_point__args__principal_point__data_;
  double*
      facs__pinhole_split_fixed_rotation_fixed_focal_fixed_point__args__rotation__data_;
  SharedIndex*
      facs__pinhole_split_fixed_rotation_fixed_focal_fixed_point__args__translation__idx_shared_;
  SharedIndex*
      facs__pinhole_split_fixed_rotation_fixed_focal_fixed_point__args__principal_point__idx_shared_;
  double*
      facs__pinhole_split_fixed_rotation_fixed_focal_fixed_point__args__pixel__data_;
  double*
      facs__pinhole_split_fixed_rotation_fixed_focal_fixed_point__args__weight_loss__data_;
  double*
      facs__pinhole_split_fixed_rotation_fixed_focal_fixed_point__args__focal__data_;
  double*
      facs__pinhole_split_fixed_rotation_fixed_focal_fixed_point__args__point__data_;
  double*
      facs__pinhole_split_fixed_rotation_fixed_principal_point_fixed_point__args__rotation__data_;
  SharedIndex*
      facs__pinhole_split_fixed_rotation_fixed_principal_point_fixed_point__args__translation__idx_shared_;
  SharedIndex*
      facs__pinhole_split_fixed_rotation_fixed_principal_point_fixed_point__args__focal__idx_shared_;
  double*
      facs__pinhole_split_fixed_rotation_fixed_principal_point_fixed_point__args__pixel__data_;
  double*
      facs__pinhole_split_fixed_rotation_fixed_principal_point_fixed_point__args__weight_loss__data_;
  double*
      facs__pinhole_split_fixed_rotation_fixed_principal_point_fixed_point__args__principal_point__data_;
  double*
      facs__pinhole_split_fixed_rotation_fixed_principal_point_fixed_point__args__point__data_;
  double*
      facs__pinhole_split_fixed_rotation_fixed_focal_fixed_principal_point_fixed_point__args__rotation__data_;
  SharedIndex*
      facs__pinhole_split_fixed_rotation_fixed_focal_fixed_principal_point_fixed_point__args__translation__idx_shared_;
  double*
      facs__pinhole_split_fixed_rotation_fixed_focal_fixed_principal_point_fixed_point__args__pixel__data_;
  double*
      facs__pinhole_split_fixed_rotation_fixed_focal_fixed_principal_point_fixed_point__args__weight_loss__data_;
  double*
      facs__pinhole_split_fixed_rotation_fixed_focal_fixed_principal_point_fixed_point__args__focal__data_;
  double*
      facs__pinhole_split_fixed_rotation_fixed_focal_fixed_principal_point_fixed_point__args__principal_point__data_;
  double*
      facs__pinhole_split_fixed_rotation_fixed_focal_fixed_principal_point_fixed_point__args__point__data_;
  SharedIndex*
      facs__pinhole_split_intrinsics_prior_fixed_focal__args__principal_point__idx_shared_;
  double* facs__pinhole_split_intrinsics_prior_fixed_focal__args__prior__data_;
  double*
      facs__pinhole_split_intrinsics_prior_fixed_focal__args__inv_std__data_;
  double* facs__pinhole_split_intrinsics_prior_fixed_focal__args__focal__data_;
  SharedIndex*
      facs__pinhole_split_intrinsics_prior_fixed_principal_point__args__focal__idx_shared_;
  double*
      facs__pinhole_split_intrinsics_prior_fixed_principal_point__args__prior__data_;
  double*
      facs__pinhole_split_intrinsics_prior_fixed_principal_point__args__inv_std__data_;
  double*
      facs__pinhole_split_intrinsics_prior_fixed_principal_point__args__principal_point__data_;
  SharedIndex*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_focal__args__prev_principal_point__idx_shared_;
  SharedIndex*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_focal__args__next_focal__idx_shared_;
  SharedIndex*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_focal__args__next_principal_point__idx_shared_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_focal__args__inv_std__data_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_focal__args__prev_focal__data_;
  SharedIndex*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_principal_point__args__prev_focal__idx_shared_;
  SharedIndex*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_principal_point__args__next_focal__idx_shared_;
  SharedIndex*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_principal_point__args__next_principal_point__idx_shared_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_principal_point__args__inv_std__data_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_principal_point__args__prev_principal_point__data_;
  SharedIndex*
      facs__pinhole_split_intrinsics_random_walk_fixed_next_focal__args__prev_focal__idx_shared_;
  SharedIndex*
      facs__pinhole_split_intrinsics_random_walk_fixed_next_focal__args__prev_principal_point__idx_shared_;
  SharedIndex*
      facs__pinhole_split_intrinsics_random_walk_fixed_next_focal__args__next_principal_point__idx_shared_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_next_focal__args__inv_std__data_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_next_focal__args__next_focal__data_;
  SharedIndex*
      facs__pinhole_split_intrinsics_random_walk_fixed_next_principal_point__args__prev_focal__idx_shared_;
  SharedIndex*
      facs__pinhole_split_intrinsics_random_walk_fixed_next_principal_point__args__prev_principal_point__idx_shared_;
  SharedIndex*
      facs__pinhole_split_intrinsics_random_walk_fixed_next_principal_point__args__next_focal__idx_shared_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_next_principal_point__args__inv_std__data_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_next_principal_point__args__next_principal_point__data_;
  SharedIndex*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_prev_principal_point__args__next_focal__idx_shared_;
  SharedIndex*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_prev_principal_point__args__next_principal_point__idx_shared_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_prev_principal_point__args__inv_std__data_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_prev_principal_point__args__prev_focal__data_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_prev_principal_point__args__prev_principal_point__data_;
  SharedIndex*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_next_focal__args__prev_principal_point__idx_shared_;
  SharedIndex*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_next_focal__args__next_principal_point__idx_shared_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_next_focal__args__inv_std__data_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_next_focal__args__prev_focal__data_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_next_focal__args__next_focal__data_;
  SharedIndex*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_next_principal_point__args__prev_principal_point__idx_shared_;
  SharedIndex*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_next_principal_point__args__next_focal__idx_shared_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_next_principal_point__args__inv_std__data_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_next_principal_point__args__prev_focal__data_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_next_principal_point__args__next_principal_point__data_;
  SharedIndex*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_principal_point_fixed_next_focal__args__prev_focal__idx_shared_;
  SharedIndex*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_principal_point_fixed_next_focal__args__next_principal_point__idx_shared_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_principal_point_fixed_next_focal__args__inv_std__data_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_principal_point_fixed_next_focal__args__prev_principal_point__data_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_principal_point_fixed_next_focal__args__next_focal__data_;
  SharedIndex*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_principal_point_fixed_next_principal_point__args__prev_focal__idx_shared_;
  SharedIndex*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_principal_point_fixed_next_principal_point__args__next_focal__idx_shared_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_principal_point_fixed_next_principal_point__args__inv_std__data_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_principal_point_fixed_next_principal_point__args__prev_principal_point__data_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_principal_point_fixed_next_principal_point__args__next_principal_point__data_;
  SharedIndex*
      facs__pinhole_split_intrinsics_random_walk_fixed_next_focal_fixed_next_principal_point__args__prev_focal__idx_shared_;
  SharedIndex*
      facs__pinhole_split_intrinsics_random_walk_fixed_next_focal_fixed_next_principal_point__args__prev_principal_point__idx_shared_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_next_focal_fixed_next_principal_point__args__inv_std__data_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_next_focal_fixed_next_principal_point__args__next_focal__data_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_next_focal_fixed_next_principal_point__args__next_principal_point__data_;
  SharedIndex*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_prev_principal_point_fixed_next_focal__args__next_principal_point__idx_shared_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_prev_principal_point_fixed_next_focal__args__inv_std__data_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_prev_principal_point_fixed_next_focal__args__prev_focal__data_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_prev_principal_point_fixed_next_focal__args__prev_principal_point__data_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_prev_principal_point_fixed_next_focal__args__next_focal__data_;
  SharedIndex*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_prev_principal_point_fixed_next_principal_point__args__next_focal__idx_shared_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_prev_principal_point_fixed_next_principal_point__args__inv_std__data_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_prev_principal_point_fixed_next_principal_point__args__prev_focal__data_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_prev_principal_point_fixed_next_principal_point__args__prev_principal_point__data_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_prev_principal_point_fixed_next_principal_point__args__next_principal_point__data_;
  SharedIndex*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_next_focal_fixed_next_principal_point__args__prev_principal_point__idx_shared_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_next_focal_fixed_next_principal_point__args__inv_std__data_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_next_focal_fixed_next_principal_point__args__prev_focal__data_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_next_focal_fixed_next_principal_point__args__next_focal__data_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_next_focal_fixed_next_principal_point__args__next_principal_point__data_;
  SharedIndex*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_principal_point_fixed_next_focal_fixed_next_principal_point__args__prev_focal__idx_shared_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_principal_point_fixed_next_focal_fixed_next_principal_point__args__inv_std__data_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_principal_point_fixed_next_focal_fixed_next_principal_point__args__prev_principal_point__data_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_principal_point_fixed_next_focal_fixed_next_principal_point__args__next_focal__data_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_principal_point_fixed_next_focal_fixed_next_principal_point__args__next_principal_point__data_;
  double* marker__scratch_inout_;
  double* facs__simple_radial__res_;
  double* facs__simple_radial_fixed_pose__res_;
  double* facs__simple_radial_fixed_point__res_;
  double* facs__simple_radial_fixed_pose_fixed_point__res_;
  double* facs__pinhole__res_;
  double* facs__pinhole_fixed_pose__res_;
  double* facs__pinhole_fixed_point__res_;
  double* facs__pinhole_fixed_pose_fixed_point__res_;
  double* facs__pinhole_log_depth__res_;
  double* facs__pinhole_log_depth_fixed_pose__res_;
  double* facs__pinhole_log_depth_fixed_scale__res_;
  double* facs__pinhole_log_depth_fixed_point__res_;
  double* facs__pinhole_log_depth_fixed_pose_fixed_scale__res_;
  double* facs__pinhole_log_depth_fixed_pose_fixed_point__res_;
  double* facs__pinhole_log_depth_fixed_scale_fixed_point__res_;
  double* facs__pinhole_fixed_rotation__res_;
  double* facs__pinhole_fixed_rotation_fixed_calib__res_;
  double* facs__pinhole_fixed_rotation_fixed_point__res_;
  double* facs__pinhole_fixed_rotation_fixed_calib_fixed_point__res_;
  double* facs__pinhole_log_depth_fixed_rotation__res_;
  double* facs__pinhole_log_depth_fixed_rotation_fixed_scale__res_;
  double* facs__pinhole_log_depth_fixed_rotation_fixed_point__res_;
  double* facs__pinhole_log_depth_fixed_rotation_fixed_scale_fixed_point__res_;
  double* facs__pinhole_intrinsics_prior__res_;
  double* facs__pinhole_intrinsics_random_walk__res_;
  double* facs__scale_prior__res_;
  double* facs__simple_radial_split_fixed_focal_and_distortion__res_;
  double* facs__simple_radial_split_fixed_principal_point__res_;
  double* facs__simple_radial_split_fixed_pose_fixed_focal_and_distortion__res_;
  double* facs__simple_radial_split_fixed_pose_fixed_principal_point__res_;
  double*
      facs__simple_radial_split_fixed_focal_and_distortion_fixed_principal_point__res_;
  double*
      facs__simple_radial_split_fixed_focal_and_distortion_fixed_point__res_;
  double* facs__simple_radial_split_fixed_principal_point_fixed_point__res_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_principal_point__res_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_point__res_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__res_;
  double*
      facs__simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_fixed_point__res_;
  double* facs__pinhole_split_fixed_focal__res_;
  double* facs__pinhole_split_fixed_principal_point__res_;
  double* facs__pinhole_split_fixed_pose_fixed_focal__res_;
  double* facs__pinhole_split_fixed_pose_fixed_principal_point__res_;
  double* facs__pinhole_split_fixed_focal_fixed_principal_point__res_;
  double* facs__pinhole_split_fixed_focal_fixed_point__res_;
  double* facs__pinhole_split_fixed_principal_point_fixed_point__res_;
  double*
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__res_;
  double* facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__res_;
  double*
      facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__res_;
  double*
      facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__res_;
  double* facs__pinhole_split_fixed_rotation_fixed_focal__res_;
  double* facs__pinhole_split_fixed_rotation_fixed_principal_point__res_;
  double*
      facs__pinhole_split_fixed_rotation_fixed_focal_fixed_principal_point__res_;
  double* facs__pinhole_split_fixed_rotation_fixed_focal_fixed_point__res_;
  double*
      facs__pinhole_split_fixed_rotation_fixed_principal_point_fixed_point__res_;
  double*
      facs__pinhole_split_fixed_rotation_fixed_focal_fixed_principal_point_fixed_point__res_;
  double* facs__pinhole_split_intrinsics_prior_fixed_focal__res_;
  double* facs__pinhole_split_intrinsics_prior_fixed_principal_point__res_;
  double* facs__pinhole_split_intrinsics_random_walk_fixed_prev_focal__res_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_principal_point__res_;
  double* facs__pinhole_split_intrinsics_random_walk_fixed_next_focal__res_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_next_principal_point__res_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_prev_principal_point__res_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_next_focal__res_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_next_principal_point__res_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_principal_point_fixed_next_focal__res_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_principal_point_fixed_next_principal_point__res_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_next_focal_fixed_next_principal_point__res_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_prev_principal_point_fixed_next_focal__res_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_prev_principal_point_fixed_next_principal_point__res_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_next_focal_fixed_next_principal_point__res_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_principal_point_fixed_next_focal_fixed_next_principal_point__res_;
  double* facs__simple_radial__args__pose__jac_;
  double* facs__simple_radial__args__calib__jac_;
  double* facs__simple_radial__args__point__jac_;
  double* facs__simple_radial_fixed_pose__args__calib__jac_;
  double* facs__simple_radial_fixed_pose__args__point__jac_;
  double* facs__simple_radial_fixed_point__args__pose__jac_;
  double* facs__simple_radial_fixed_point__args__calib__jac_;
  double* facs__simple_radial_fixed_pose_fixed_point__args__calib__jac_;
  double* facs__pinhole__args__pose__jac_;
  double* facs__pinhole__args__calib__jac_;
  double* facs__pinhole__args__point__jac_;
  double* facs__pinhole_fixed_pose__args__calib__jac_;
  double* facs__pinhole_fixed_pose__args__point__jac_;
  double* facs__pinhole_fixed_point__args__pose__jac_;
  double* facs__pinhole_fixed_point__args__calib__jac_;
  double* facs__pinhole_fixed_pose_fixed_point__args__calib__jac_;
  double* facs__pinhole_log_depth__args__pose__jac_;
  double* facs__pinhole_log_depth__args__scale__jac_;
  double* facs__pinhole_log_depth__args__point__jac_;
  double* facs__pinhole_log_depth_fixed_pose__args__scale__jac_;
  double* facs__pinhole_log_depth_fixed_pose__args__point__jac_;
  double* facs__pinhole_log_depth_fixed_scale__args__pose__jac_;
  double* facs__pinhole_log_depth_fixed_scale__args__point__jac_;
  double* facs__pinhole_log_depth_fixed_point__args__pose__jac_;
  double* facs__pinhole_log_depth_fixed_point__args__scale__jac_;
  double* facs__pinhole_log_depth_fixed_pose_fixed_scale__args__point__jac_;
  double* facs__pinhole_log_depth_fixed_pose_fixed_point__args__scale__jac_;
  double* facs__pinhole_log_depth_fixed_scale_fixed_point__args__pose__jac_;
  double* facs__pinhole_fixed_rotation__args__translation__jac_;
  double* facs__pinhole_fixed_rotation__args__calib__jac_;
  double* facs__pinhole_fixed_rotation__args__point__jac_;
  double* facs__pinhole_fixed_rotation_fixed_calib__args__translation__jac_;
  double* facs__pinhole_fixed_rotation_fixed_calib__args__point__jac_;
  double* facs__pinhole_fixed_rotation_fixed_point__args__translation__jac_;
  double* facs__pinhole_fixed_rotation_fixed_point__args__calib__jac_;
  double*
      facs__pinhole_fixed_rotation_fixed_calib_fixed_point__args__translation__jac_;
  double* facs__pinhole_log_depth_fixed_rotation__args__translation__jac_;
  double* facs__pinhole_log_depth_fixed_rotation__args__scale__jac_;
  double* facs__pinhole_log_depth_fixed_rotation__args__point__jac_;
  double*
      facs__pinhole_log_depth_fixed_rotation_fixed_scale__args__translation__jac_;
  double* facs__pinhole_log_depth_fixed_rotation_fixed_scale__args__point__jac_;
  double*
      facs__pinhole_log_depth_fixed_rotation_fixed_point__args__translation__jac_;
  double* facs__pinhole_log_depth_fixed_rotation_fixed_point__args__scale__jac_;
  double*
      facs__pinhole_log_depth_fixed_rotation_fixed_scale_fixed_point__args__translation__jac_;
  double* facs__pinhole_intrinsics_prior__args__calib__jac_;
  double* facs__pinhole_intrinsics_random_walk__args__prev_calib__jac_;
  double* facs__pinhole_intrinsics_random_walk__args__next_calib__jac_;
  double* facs__scale_prior__args__scale__jac_;
  double*
      facs__simple_radial_split_fixed_focal_and_distortion__args__pose__jac_;
  double*
      facs__simple_radial_split_fixed_focal_and_distortion__args__principal_point__jac_;
  double*
      facs__simple_radial_split_fixed_focal_and_distortion__args__point__jac_;
  double* facs__simple_radial_split_fixed_principal_point__args__pose__jac_;
  double*
      facs__simple_radial_split_fixed_principal_point__args__focal_and_distortion__jac_;
  double* facs__simple_radial_split_fixed_principal_point__args__point__jac_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_distortion__args__principal_point__jac_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_distortion__args__point__jac_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_principal_point__args__focal_and_distortion__jac_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_principal_point__args__point__jac_;
  double*
      facs__simple_radial_split_fixed_focal_and_distortion_fixed_principal_point__args__pose__jac_;
  double*
      facs__simple_radial_split_fixed_focal_and_distortion_fixed_principal_point__args__point__jac_;
  double*
      facs__simple_radial_split_fixed_focal_and_distortion_fixed_point__args__pose__jac_;
  double*
      facs__simple_radial_split_fixed_focal_and_distortion_fixed_point__args__principal_point__jac_;
  double*
      facs__simple_radial_split_fixed_principal_point_fixed_point__args__pose__jac_;
  double*
      facs__simple_radial_split_fixed_principal_point_fixed_point__args__focal_and_distortion__jac_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_principal_point__args__point__jac_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_point__args__principal_point__jac_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__args__focal_and_distortion__jac_;
  double*
      facs__simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_fixed_point__args__pose__jac_;
  double* facs__pinhole_split_fixed_focal__args__pose__jac_;
  double* facs__pinhole_split_fixed_focal__args__principal_point__jac_;
  double* facs__pinhole_split_fixed_focal__args__point__jac_;
  double* facs__pinhole_split_fixed_principal_point__args__pose__jac_;
  double* facs__pinhole_split_fixed_principal_point__args__focal__jac_;
  double* facs__pinhole_split_fixed_principal_point__args__point__jac_;
  double*
      facs__pinhole_split_fixed_pose_fixed_focal__args__principal_point__jac_;
  double* facs__pinhole_split_fixed_pose_fixed_focal__args__point__jac_;
  double*
      facs__pinhole_split_fixed_pose_fixed_principal_point__args__focal__jac_;
  double*
      facs__pinhole_split_fixed_pose_fixed_principal_point__args__point__jac_;
  double*
      facs__pinhole_split_fixed_focal_fixed_principal_point__args__pose__jac_;
  double*
      facs__pinhole_split_fixed_focal_fixed_principal_point__args__point__jac_;
  double* facs__pinhole_split_fixed_focal_fixed_point__args__pose__jac_;
  double*
      facs__pinhole_split_fixed_focal_fixed_point__args__principal_point__jac_;
  double*
      facs__pinhole_split_fixed_principal_point_fixed_point__args__pose__jac_;
  double*
      facs__pinhole_split_fixed_principal_point_fixed_point__args__focal__jac_;
  double*
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__args__point__jac_;
  double*
      facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__args__principal_point__jac_;
  double*
      facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__args__focal__jac_;
  double*
      facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__args__pose__jac_;
  double*
      facs__pinhole_split_fixed_rotation_fixed_focal__args__translation__jac_;
  double*
      facs__pinhole_split_fixed_rotation_fixed_focal__args__principal_point__jac_;
  double* facs__pinhole_split_fixed_rotation_fixed_focal__args__point__jac_;
  double*
      facs__pinhole_split_fixed_rotation_fixed_principal_point__args__translation__jac_;
  double*
      facs__pinhole_split_fixed_rotation_fixed_principal_point__args__focal__jac_;
  double*
      facs__pinhole_split_fixed_rotation_fixed_principal_point__args__point__jac_;
  double*
      facs__pinhole_split_fixed_rotation_fixed_focal_fixed_principal_point__args__translation__jac_;
  double*
      facs__pinhole_split_fixed_rotation_fixed_focal_fixed_principal_point__args__point__jac_;
  double*
      facs__pinhole_split_fixed_rotation_fixed_focal_fixed_point__args__translation__jac_;
  double*
      facs__pinhole_split_fixed_rotation_fixed_focal_fixed_point__args__principal_point__jac_;
  double*
      facs__pinhole_split_fixed_rotation_fixed_principal_point_fixed_point__args__translation__jac_;
  double*
      facs__pinhole_split_fixed_rotation_fixed_principal_point_fixed_point__args__focal__jac_;
  double*
      facs__pinhole_split_fixed_rotation_fixed_focal_fixed_principal_point_fixed_point__args__translation__jac_;
  double*
      facs__pinhole_split_intrinsics_prior_fixed_focal__args__principal_point__jac_;
  double*
      facs__pinhole_split_intrinsics_prior_fixed_principal_point__args__focal__jac_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_focal__args__prev_principal_point__jac_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_focal__args__next_focal__jac_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_focal__args__next_principal_point__jac_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_principal_point__args__prev_focal__jac_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_principal_point__args__next_focal__jac_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_principal_point__args__next_principal_point__jac_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_next_focal__args__prev_focal__jac_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_next_focal__args__prev_principal_point__jac_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_next_focal__args__next_principal_point__jac_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_next_principal_point__args__prev_focal__jac_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_next_principal_point__args__prev_principal_point__jac_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_next_principal_point__args__next_focal__jac_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_prev_principal_point__args__next_focal__jac_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_prev_principal_point__args__next_principal_point__jac_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_next_focal__args__prev_principal_point__jac_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_next_focal__args__next_principal_point__jac_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_next_principal_point__args__prev_principal_point__jac_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_next_principal_point__args__next_focal__jac_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_principal_point_fixed_next_focal__args__prev_focal__jac_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_principal_point_fixed_next_focal__args__next_principal_point__jac_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_principal_point_fixed_next_principal_point__args__prev_focal__jac_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_principal_point_fixed_next_principal_point__args__next_focal__jac_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_next_focal_fixed_next_principal_point__args__prev_focal__jac_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_next_focal_fixed_next_principal_point__args__prev_principal_point__jac_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_prev_principal_point_fixed_next_focal__args__next_principal_point__jac_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_prev_principal_point_fixed_next_principal_point__args__next_focal__jac_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_next_focal_fixed_next_principal_point__args__prev_principal_point__jac_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_principal_point_fixed_next_focal_fixed_next_principal_point__args__prev_focal__jac_;
  double* nodes__DepthScale__z_;
  double* nodes__DepthScale__z_end__;
  double* nodes__PinholeCalib__z_;
  double* nodes__PinholeCalib__z_end__;
  double* nodes__PinholeFocal__z_;
  double* nodes__PinholeFocal__z_end__;
  double* nodes__PinholePose__z_;
  double* nodes__PinholePose__z_end__;
  double* nodes__PinholePrincipalPoint__z_;
  double* nodes__PinholePrincipalPoint__z_end__;
  double* nodes__PinholeTranslation__z_;
  double* nodes__PinholeTranslation__z_end__;
  double* nodes__Point__z_;
  double* nodes__Point__z_end__;
  double* nodes__SimpleRadialCalib__z_;
  double* nodes__SimpleRadialCalib__z_end__;
  double* nodes__SimpleRadialFocalAndDistortion__z_;
  double* nodes__SimpleRadialFocalAndDistortion__z_end__;
  double* nodes__SimpleRadialPose__z_;
  double* nodes__SimpleRadialPose__z_end__;
  double* nodes__SimpleRadialPrincipalPoint__z_;
  double* nodes__SimpleRadialPrincipalPoint__z_end__;
  double* nodes__DepthScale__p_;
  double* nodes__DepthScale__p_end__;
  double* nodes__PinholeCalib__p_;
  double* nodes__PinholeCalib__p_end__;
  double* nodes__PinholeFocal__p_;
  double* nodes__PinholeFocal__p_end__;
  double* nodes__PinholePose__p_;
  double* nodes__PinholePose__p_end__;
  double* nodes__PinholePrincipalPoint__p_;
  double* nodes__PinholePrincipalPoint__p_end__;
  double* nodes__PinholeTranslation__p_;
  double* nodes__PinholeTranslation__p_end__;
  double* nodes__Point__p_;
  double* nodes__Point__p_end__;
  double* nodes__SimpleRadialCalib__p_;
  double* nodes__SimpleRadialCalib__p_end__;
  double* nodes__SimpleRadialFocalAndDistortion__p_;
  double* nodes__SimpleRadialFocalAndDistortion__p_end__;
  double* nodes__SimpleRadialPose__p_;
  double* nodes__SimpleRadialPose__p_end__;
  double* nodes__SimpleRadialPrincipalPoint__p_;
  double* nodes__SimpleRadialPrincipalPoint__p_end__;
  double* nodes__DepthScale__step_;
  double* nodes__DepthScale__step_end__;
  double* nodes__PinholeCalib__step_;
  double* nodes__PinholeCalib__step_end__;
  double* nodes__PinholeFocal__step_;
  double* nodes__PinholeFocal__step_end__;
  double* nodes__PinholePose__step_;
  double* nodes__PinholePose__step_end__;
  double* nodes__PinholePrincipalPoint__step_;
  double* nodes__PinholePrincipalPoint__step_end__;
  double* nodes__PinholeTranslation__step_;
  double* nodes__PinholeTranslation__step_end__;
  double* nodes__Point__step_;
  double* nodes__Point__step_end__;
  double* nodes__SimpleRadialCalib__step_;
  double* nodes__SimpleRadialCalib__step_end__;
  double* nodes__SimpleRadialFocalAndDistortion__step_;
  double* nodes__SimpleRadialFocalAndDistortion__step_end__;
  double* nodes__SimpleRadialPose__step_;
  double* nodes__SimpleRadialPose__step_end__;
  double* nodes__SimpleRadialPrincipalPoint__step_;
  double* nodes__SimpleRadialPrincipalPoint__step_end__;
  double* marker__w_start_;
  double* nodes__DepthScale__w_;
  double* nodes__PinholeCalib__w_;
  double* nodes__PinholeFocal__w_;
  double* nodes__PinholePose__w_;
  double* nodes__PinholePrincipalPoint__w_;
  double* nodes__PinholeTranslation__w_;
  double* nodes__Point__w_;
  double* nodes__SimpleRadialCalib__w_;
  double* nodes__SimpleRadialFocalAndDistortion__w_;
  double* nodes__SimpleRadialPose__w_;
  double* nodes__SimpleRadialPrincipalPoint__w_;
  double* marker__w_end_;
  double* marker__r_0_start_;
  double* nodes__DepthScale__r_0_;
  double* nodes__PinholeCalib__r_0_;
  double* nodes__PinholeFocal__r_0_;
  double* nodes__PinholePose__r_0_;
  double* nodes__PinholePrincipalPoint__r_0_;
  double* nodes__PinholeTranslation__r_0_;
  double* nodes__Point__r_0_;
  double* nodes__SimpleRadialCalib__r_0_;
  double* nodes__SimpleRadialFocalAndDistortion__r_0_;
  double* nodes__SimpleRadialPose__r_0_;
  double* nodes__SimpleRadialPrincipalPoint__r_0_;
  double* marker__r_0_end_;
  double* marker__r_k_start_;
  double* nodes__DepthScale__r_k_;
  double* nodes__PinholeCalib__r_k_;
  double* nodes__PinholeFocal__r_k_;
  double* nodes__PinholePose__r_k_;
  double* nodes__PinholePrincipalPoint__r_k_;
  double* nodes__PinholeTranslation__r_k_;
  double* nodes__Point__r_k_;
  double* nodes__SimpleRadialCalib__r_k_;
  double* nodes__SimpleRadialFocalAndDistortion__r_k_;
  double* nodes__SimpleRadialPose__r_k_;
  double* nodes__SimpleRadialPrincipalPoint__r_k_;
  double* marker__r_k_end_;
  double* marker__Mp_start_;
  double* nodes__DepthScale__Mp_;
  double* nodes__PinholeCalib__Mp_;
  double* nodes__PinholeFocal__Mp_;
  double* nodes__PinholePose__Mp_;
  double* nodes__PinholePrincipalPoint__Mp_;
  double* nodes__PinholeTranslation__Mp_;
  double* nodes__Point__Mp_;
  double* nodes__SimpleRadialCalib__Mp_;
  double* nodes__SimpleRadialFocalAndDistortion__Mp_;
  double* nodes__SimpleRadialPose__Mp_;
  double* nodes__SimpleRadialPrincipalPoint__Mp_;
  double* marker__Mp_end_;
  double* marker__precond_start_;
  double* nodes__DepthScale__precond_diag_;
  double* nodes__DepthScale__precond_tril_;
  double* nodes__PinholeCalib__precond_diag_;
  double* nodes__PinholeCalib__precond_tril_;
  double* nodes__PinholeFocal__precond_diag_;
  double* nodes__PinholeFocal__precond_tril_;
  double* nodes__PinholePose__precond_diag_;
  double* nodes__PinholePose__precond_tril_;
  double* nodes__PinholePrincipalPoint__precond_diag_;
  double* nodes__PinholePrincipalPoint__precond_tril_;
  double* nodes__PinholeTranslation__precond_diag_;
  double* nodes__PinholeTranslation__precond_tril_;
  double* nodes__Point__precond_diag_;
  double* nodes__Point__precond_tril_;
  double* nodes__SimpleRadialCalib__precond_diag_;
  double* nodes__SimpleRadialCalib__precond_tril_;
  double* nodes__SimpleRadialFocalAndDistortion__precond_diag_;
  double* nodes__SimpleRadialFocalAndDistortion__precond_tril_;
  double* nodes__SimpleRadialPose__precond_diag_;
  double* nodes__SimpleRadialPose__precond_tril_;
  double* nodes__SimpleRadialPrincipalPoint__precond_diag_;
  double* nodes__SimpleRadialPrincipalPoint__precond_tril_;
  double* marker__precond_end_;
  double* marker__jp_start_;
  double* facs__simple_radial__jp_;
  double* facs__simple_radial_fixed_pose__jp_;
  double* facs__simple_radial_fixed_point__jp_;
  double* facs__simple_radial_fixed_pose_fixed_point__jp_;
  double* facs__pinhole__jp_;
  double* facs__pinhole_fixed_pose__jp_;
  double* facs__pinhole_fixed_point__jp_;
  double* facs__pinhole_fixed_pose_fixed_point__jp_;
  double* facs__pinhole_log_depth__jp_;
  double* facs__pinhole_log_depth_fixed_pose__jp_;
  double* facs__pinhole_log_depth_fixed_scale__jp_;
  double* facs__pinhole_log_depth_fixed_point__jp_;
  double* facs__pinhole_log_depth_fixed_pose_fixed_scale__jp_;
  double* facs__pinhole_log_depth_fixed_pose_fixed_point__jp_;
  double* facs__pinhole_log_depth_fixed_scale_fixed_point__jp_;
  double* facs__pinhole_fixed_rotation__jp_;
  double* facs__pinhole_fixed_rotation_fixed_calib__jp_;
  double* facs__pinhole_fixed_rotation_fixed_point__jp_;
  double* facs__pinhole_fixed_rotation_fixed_calib_fixed_point__jp_;
  double* facs__pinhole_log_depth_fixed_rotation__jp_;
  double* facs__pinhole_log_depth_fixed_rotation_fixed_scale__jp_;
  double* facs__pinhole_log_depth_fixed_rotation_fixed_point__jp_;
  double* facs__pinhole_log_depth_fixed_rotation_fixed_scale_fixed_point__jp_;
  double* facs__pinhole_intrinsics_prior__jp_;
  double* facs__pinhole_intrinsics_random_walk__jp_;
  double* facs__scale_prior__jp_;
  double* facs__simple_radial_split_fixed_focal_and_distortion__jp_;
  double* facs__simple_radial_split_fixed_principal_point__jp_;
  double* facs__simple_radial_split_fixed_pose_fixed_focal_and_distortion__jp_;
  double* facs__simple_radial_split_fixed_pose_fixed_principal_point__jp_;
  double*
      facs__simple_radial_split_fixed_focal_and_distortion_fixed_principal_point__jp_;
  double* facs__simple_radial_split_fixed_focal_and_distortion_fixed_point__jp_;
  double* facs__simple_radial_split_fixed_principal_point_fixed_point__jp_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_principal_point__jp_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_focal_and_distortion_fixed_point__jp_;
  double*
      facs__simple_radial_split_fixed_pose_fixed_principal_point_fixed_point__jp_;
  double*
      facs__simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_fixed_point__jp_;
  double* facs__pinhole_split_fixed_focal__jp_;
  double* facs__pinhole_split_fixed_principal_point__jp_;
  double* facs__pinhole_split_fixed_pose_fixed_focal__jp_;
  double* facs__pinhole_split_fixed_pose_fixed_principal_point__jp_;
  double* facs__pinhole_split_fixed_focal_fixed_principal_point__jp_;
  double* facs__pinhole_split_fixed_focal_fixed_point__jp_;
  double* facs__pinhole_split_fixed_principal_point_fixed_point__jp_;
  double* facs__pinhole_split_fixed_pose_fixed_focal_fixed_principal_point__jp_;
  double* facs__pinhole_split_fixed_pose_fixed_focal_fixed_point__jp_;
  double* facs__pinhole_split_fixed_pose_fixed_principal_point_fixed_point__jp_;
  double*
      facs__pinhole_split_fixed_focal_fixed_principal_point_fixed_point__jp_;
  double* facs__pinhole_split_fixed_rotation_fixed_focal__jp_;
  double* facs__pinhole_split_fixed_rotation_fixed_principal_point__jp_;
  double*
      facs__pinhole_split_fixed_rotation_fixed_focal_fixed_principal_point__jp_;
  double* facs__pinhole_split_fixed_rotation_fixed_focal_fixed_point__jp_;
  double*
      facs__pinhole_split_fixed_rotation_fixed_principal_point_fixed_point__jp_;
  double*
      facs__pinhole_split_fixed_rotation_fixed_focal_fixed_principal_point_fixed_point__jp_;
  double* facs__pinhole_split_intrinsics_prior_fixed_focal__jp_;
  double* facs__pinhole_split_intrinsics_prior_fixed_principal_point__jp_;
  double* facs__pinhole_split_intrinsics_random_walk_fixed_prev_focal__jp_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_principal_point__jp_;
  double* facs__pinhole_split_intrinsics_random_walk_fixed_next_focal__jp_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_next_principal_point__jp_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_prev_principal_point__jp_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_next_focal__jp_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_next_principal_point__jp_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_principal_point_fixed_next_focal__jp_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_principal_point_fixed_next_principal_point__jp_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_next_focal_fixed_next_principal_point__jp_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_prev_principal_point_fixed_next_focal__jp_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_prev_principal_point_fixed_next_principal_point__jp_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_next_focal_fixed_next_principal_point__jp_;
  double*
      facs__pinhole_split_intrinsics_random_walk_fixed_prev_principal_point_fixed_next_focal_fixed_next_principal_point__jp_;
  double* marker__jp_end_;
  double* solver__current_diag_;
  double* solver__alpha_numerator_;
  double* solver__alpha_denominator_;
  double* solver__alpha_;
  double* solver__neg_alpha_;
  double* solver__beta_numerator_;
  double* solver__beta_;
  double* solver__r_0_norm2_tot_;
  double* solver__r_kp1_norm2_tot_;
  double* solver__pred_decrease_tot_;
  double* solver__res_tot_;
};

}  // namespace caspar