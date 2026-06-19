#include "colmap/estimators/cost_functions/metric_depth.h"
#include "colmap/estimators/cost_functions/motion_averaging.h"
#include "colmap/estimators/cost_functions/utils.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/LU>
#include <Eigen/SVD>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

namespace {

using SnapshotTable =
    std::unordered_map<std::string,
                       std::unordered_map<uint64_t, Eigen::VectorXd>>;

struct NativePointSchurAccumulator {
  Eigen::Matrix3d hpp_raw = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d hpp_rho1 = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d hpp_analytic = Eigen::Matrix3d::Zero();
  Eigen::Vector3d point_gradient_raw = Eigen::Vector3d::Zero();
  Eigen::Vector3d point_gradient_robust = Eigen::Vector3d::Zero();
  Eigen::Matrix3d left_raw = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d right_raw = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d left_rho1 = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d right_rho1 = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d left_analytic = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d right_analytic = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d left_hff_analytic = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d right_hff_analytic = Eigen::Matrix3d::Zero();
  Eigen::Vector3d left_gradient_raw = Eigen::Vector3d::Zero();
  Eigen::Vector3d right_gradient_raw = Eigen::Vector3d::Zero();
  Eigen::Vector3d left_gradient_robust = Eigen::Vector3d::Zero();
  Eigen::Vector3d right_gradient_robust = Eigen::Vector3d::Zero();
  uint64_t left_count = 0;
  uint64_t right_count = 0;
};

struct NativeTopResidualRow {
  std::string residual_id;
  std::string side;
  uint64_t frame_id = 0;
  uint64_t image_id = 0;
  uint64_t point2D_idx = 0;
  uint64_t point3D_id = 0;
  std::string residual_type;
  std::string loss_bucket;
  std::string source;
  double descent_projection = 0.0;
  double opposing_projection = 0.0;
  double raw_gradient_norm = 0.0;
  double robust_gradient_norm = 0.0;
  double raw_cost = 0.0;
  double robust_cost = 0.0;
  double rho1 = 0.0;
  std::optional<double> directional_curvature;
  std::optional<double> newton_step_along_projection;
  Eigen::Vector3d raw_gradient = Eigen::Vector3d::Zero();
  Eigen::Vector3d robust_gradient = Eigen::Vector3d::Zero();
};

struct NativeSchurResidualRow {
  std::string residual_id;
  std::string side;
  uint64_t frame_id = 0;
  uint64_t image_id = 0;
  uint64_t point2D_idx = 0;
  uint64_t point3D_id = 0;
  std::string residual_type;
  std::string loss_bucket;
  std::string source;
  double raw_cost = 0.0;
  double robust_cost = 0.0;
  double rho1 = 0.0;
  Eigen::Vector3d robust_frame_gradient = Eigen::Vector3d::Zero();
  Eigen::Vector3d robust_point_gradient = Eigen::Vector3d::Zero();
  Eigen::Matrix3d analytic_cross = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d analytic_hff = Eigen::Matrix3d::Zero();
};

struct NativeEdgeJacobianAccumulator {
  uint64_t residual_ids_selected = 0;
  uint64_t residuals_replayed = 0;
  uint64_t residuals_with_frame_jacobian = 0;
  uint64_t residuals_with_frame_point_jacobians = 0;
  double raw_cost_sum = 0.0;
  double robust_cost_sum = 0.0;
  double rho1_sum = 0.0;
  double rho2_sum = 0.0;
  double rho2_min = std::numeric_limits<double>::infinity();
  double rho2_max = -std::numeric_limits<double>::infinity();
  uint64_t rho1_downweighted_count = 0;
  Eigen::Vector3d left_gradient_raw = Eigen::Vector3d::Zero();
  Eigen::Vector3d right_gradient_raw = Eigen::Vector3d::Zero();
  Eigen::Vector3d left_gradient_robust = Eigen::Vector3d::Zero();
  Eigen::Vector3d right_gradient_robust = Eigen::Vector3d::Zero();
  std::unordered_map<uint64_t, NativePointSchurAccumulator> point_accumulators;
  std::vector<NativeTopResidualRow> top_opposing_residuals;
  std::vector<NativeTopResidualRow> top_pinning_residuals;
  std::vector<NativeSchurResidualRow> schur_residuals;
};

Eigen::VectorXd VectorFromPy(const py::handle value, const std::string& label) {
  const py::array_t<double, py::array::c_style | py::array::forcecast> array(
      py::reinterpret_borrow<py::object>(value));
  if (array.ndim() != 1) {
    throw std::runtime_error(label + ": expected a rank-1 float64 array");
  }
  Eigen::VectorXd result(array.shape(0));
  const double* data = array.data();
  for (ssize_t idx = 0; idx < array.shape(0); ++idx) {
    result[static_cast<Eigen::Index>(idx)] = data[idx];
  }
  return result;
}

Eigen::Vector3d Vector3FromPy(const py::dict& dict,
                              const char* key,
                              const std::string& label) {
  if (!dict.contains(key)) {
    throw std::runtime_error(label + ": missing " + key);
  }
  const py::list values = py::cast<py::list>(dict[py::str(key)]);
  if (values.size() != 3) {
    throw std::runtime_error(label + "." + key + ": expected length 3");
  }
  return Eigen::Vector3d(py::cast<double>(values[0]),
                         py::cast<double>(values[1]),
                         py::cast<double>(values[2]));
}

Eigen::Quaterniond QuaternionWxyzFromPy(const py::dict& dict,
                                        const char* key,
                                        const std::string& label) {
  if (!dict.contains(key)) {
    throw std::runtime_error(label + ": missing " + key);
  }
  const py::list values = py::cast<py::list>(dict[py::str(key)]);
  if (values.size() != 4) {
    throw std::runtime_error(label + "." + key + ": expected length 4");
  }
  return Eigen::Quaterniond(py::cast<double>(values[0]),
                            py::cast<double>(values[1]),
                            py::cast<double>(values[2]),
                            py::cast<double>(values[3]));
}

Eigen::Matrix3d Matrix3RowMajorFromPy(const py::dict& dict,
                                      const char* key,
                                      const std::string& label) {
  if (!dict.contains(key)) {
    throw std::runtime_error(label + ": missing " + key);
  }
  const py::list values = py::cast<py::list>(dict[py::str(key)]);
  if (values.size() != 9) {
    throw std::runtime_error(label + "." + key + ": expected length 9");
  }
  Eigen::Matrix3d matrix;
  size_t idx = 0;
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      matrix(row, col) = py::cast<double>(values[idx++]);
    }
  }
  return matrix;
}

double OptionalDoubleFromPy(const py::dict& dict,
                            const char* key,
                            const double default_value) {
  if (!dict.contains(key) || dict[py::str(key)].is_none()) {
    return default_value;
  }
  return py::cast<double>(dict[py::str(key)]);
}

uint64_t OptionalUInt64FromPy(const py::dict& dict,
                              const char* key,
                              const uint64_t default_value) {
  if (!dict.contains(key) || dict[py::str(key)].is_none()) {
    return default_value;
  }
  return py::cast<uint64_t>(dict[py::str(key)]);
}

std::string OptionalStringFromPy(const py::dict& dict,
                                 const char* key,
                                 const std::string& default_value) {
  if (!dict.contains(key) || dict[py::str(key)].is_none()) {
    return default_value;
  }
  return py::cast<std::string>(dict[py::str(key)]);
}

double RequiredDoubleFromPy(const py::dict& dict,
                            const char* key,
                            const std::string& label) {
  if (!dict.contains(key) || dict[py::str(key)].is_none()) {
    throw std::runtime_error(label + ": missing " + key);
  }
  return py::cast<double>(dict[py::str(key)]);
}

bool RequiredBoolFromPy(const py::dict& dict,
                        const char* key,
                        const std::string& label) {
  if (!dict.contains(key) || dict[py::str(key)].is_none()) {
    throw std::runtime_error(label + ": missing " + key);
  }
  return py::cast<bool>(dict[py::str(key)]);
}

std::string RequiredStringFromPy(const py::dict& dict,
                                 const char* key,
                                 const std::string& label) {
  if (!dict.contains(key) || dict[py::str(key)].is_none()) {
    throw std::runtime_error(label + ": missing " + key);
  }
  return py::cast<std::string>(dict[py::str(key)]);
}

SnapshotTable SnapshotTableFromPy(const py::dict& snapshot) {
  SnapshotTable table;
  for (const auto item : snapshot) {
    const std::string kind = py::cast<std::string>(item.first);
    const py::dict values = py::cast<py::dict>(item.second);
    auto& kind_table = table[kind];
    for (const auto value_item : values) {
      kind_table[py::cast<uint64_t>(value_item.first)] =
          VectorFromPy(value_item.second, kind);
    }
  }
  return table;
}

const Eigen::VectorXd& SnapshotValue(const SnapshotTable& table,
                                     const std::string& kind,
                                     const uint64_t id,
                                     const size_t expected_size) {
  const auto kind_it = table.find(kind);
  if (kind_it == table.end()) {
    throw std::runtime_error("snapshot missing parameter kind: " + kind);
  }
  const auto value_it = kind_it->second.find(id);
  if (value_it == kind_it->second.end()) {
    throw std::runtime_error("snapshot missing parameter kind=" + kind +
                             ", id=" + std::to_string(id));
  }
  if (static_cast<size_t>(value_it->second.size()) != expected_size) {
    throw std::runtime_error("snapshot parameter kind=" + kind + ", id=" +
                             std::to_string(id) + " has unexpected size");
  }
  return value_it->second;
}

std::unique_ptr<ceres::CostFunction> CreateCostFunctionFromRecord(
    const py::dict& attrs) {
  const std::string residual_type =
      RequiredStringFromPy(attrs, "residual_type", "residual attrs");
  const py::dict fixed = py::cast<py::dict>(attrs[py::str("fixed_parameters")]);

  if (residual_type == "bata_ref_frame") {
    const Eigen::Vector3d direction =
        Vector3FromPy(fixed, "cam_from_point3D_dir", residual_type);
    if (fixed.contains("keypoint_covariance_world_row_major")) {
      return std::unique_ptr<ceres::CostFunction>(
          CovarianceWeightedCostFunctor<BATAPairwiseDirectionCostFunctor>::
              Create(
                  Matrix3RowMajorFromPy(fixed,
                                        "keypoint_covariance_world_row_major",
                                        residual_type),
                  direction));
    }
    return std::unique_ptr<ceres::CostFunction>(
        BATAPairwiseDirectionCostFunctor::Create(direction));
  }

  if (residual_type == "bata_constant_rig") {
    return std::unique_ptr<ceres::CostFunction>(
        RigBATAPairwiseDirectionConstantRigCostFunctor::Create(
            Vector3FromPy(fixed, "cam_from_point3D_dir", residual_type),
            Vector3FromPy(fixed, "cam_from_rig_dir", residual_type)));
  }

  if (residual_type == "bata_variable_rig") {
    return std::unique_ptr<ceres::CostFunction>(
        RigBATAPairwiseDirectionCostFunctor::Create(
            Vector3FromPy(fixed, "cam_from_point3D_dir", residual_type),
            QuaternionWxyzFromPy(
                fixed, "rig_from_world_rotation_wxyz", residual_type)));
  }

  if (residual_type == "metric_depth") {
    MetricDepthOptions options;
    options.use_log_scale =
        RequiredBoolFromPy(fixed, "metric_depth_use_log_scale", residual_type);
    const std::string metric_type = RequiredStringFromPy(
        fixed, "metric_depth_residual_type", residual_type);
    if (metric_type == "linear") {
      options.residual_type = MetricDepthResidualType::kLinear;
    } else if (metric_type == "log") {
      options.residual_type = MetricDepthResidualType::kLog;
    } else if (metric_type == "log_linear") {
      options.residual_type = MetricDepthResidualType::kLogLinear;
    } else {
      throw std::runtime_error("unsupported metric depth residual type: " +
                               metric_type);
    }
    options.zero_residual_behind = RequiredBoolFromPy(
        fixed, "metric_depth_zero_residual_behind", residual_type);
    options.log_linear_threshold = RequiredDoubleFromPy(
        fixed, "metric_depth_log_linear_threshold", residual_type);
    return std::unique_ptr<ceres::CostFunction>(MetricDepthError::Create(
        QuaternionWxyzFromPy(fixed, "camera_rotation_wxyz", residual_type),
        RequiredDoubleFromPy(attrs, "depth_prior", residual_type),
        RequiredDoubleFromPy(attrs, "depth_sigma", residual_type),
        options));
  }

  if (residual_type == "scale_prior") {
    const Eigen::Matrix<double, 1, 1> prior(
        RequiredDoubleFromPy(fixed, "scale_prior_target", residual_type));
    const double stddev =
        RequiredDoubleFromPy(fixed, "scale_prior_stddev", residual_type);
    const Eigen::Matrix<double, 1, 1> covariance(stddev * stddev);
    return std::unique_ptr<ceres::CostFunction>(
        CovarianceWeightedCostFunctor<NormalPriorCostFunctor<1>>::Create(
            covariance, prior));
  }

  throw std::runtime_error("unsupported residual_type: " + residual_type);
}

Eigen::Vector3d LossRhoFromRecord(const py::dict& attrs,
                                  const double squared_norm) {
  const py::dict loss = py::cast<py::dict>(attrs[py::str("loss")]);
  const std::string type = RequiredStringFromPy(loss, "type", "loss");
  const double scale = OptionalDoubleFromPy(loss, "scale", 1.0);
  const double weight = OptionalDoubleFromPy(loss, "weight", 1.0);
  if (scale <= 0.0) {
    throw std::runtime_error("loss scale must be positive");
  }
  if (weight < 0.0) {
    throw std::runtime_error("loss weight must be non-negative");
  }
  Eigen::Vector3d rho;
  if (type == "trivial") {
    rho << squared_norm, 1.0, 0.0;
  } else if (type == "huber") {
    const double threshold = scale * scale;
    if (squared_norm <= threshold) {
      rho << squared_norm, 1.0, 0.0;
    } else {
      const double root = std::sqrt(squared_norm);
      rho << 2.0 * scale * root - threshold, scale / root,
          -0.5 * scale / (squared_norm * root);
    }
  } else if (type == "soft_l1") {
    const double z = 1.0 + squared_norm / (scale * scale);
    const double root = std::sqrt(z);
    rho << 2.0 * scale * scale * (root - 1.0), 1.0 / root,
        -0.5 / (scale * scale * z * root);
  } else if (type == "cauchy") {
    const double z = 1.0 + squared_norm / (scale * scale);
    rho << scale * scale * std::log(z), 1.0 / z, -1.0 / (scale * scale * z * z);
  } else {
    throw std::runtime_error("unsupported loss type: " + type);
  }
  return weight * rho;
}

struct NativeParameterBlock {
  std::string role;
  std::string kind;
  uint64_t id = 0;
  size_t size = 0;
};

std::vector<NativeParameterBlock> ParameterBlocksFromRecord(
    const py::dict& attrs) {
  const py::list blocks =
      py::cast<py::list>(attrs[py::str("parameter_blocks")]);
  std::vector<NativeParameterBlock> parsed;
  parsed.reserve(blocks.size());
  for (const py::handle block_handle : blocks) {
    const py::dict block = py::cast<py::dict>(block_handle);
    parsed.push_back({RequiredStringFromPy(block, "role", "parameter_block"),
                      RequiredStringFromPy(block, "kind", "parameter_block"),
                      py::cast<uint64_t>(block[py::str("id")]),
                      py::cast<size_t>(block[py::str("size")])});
  }
  return parsed;
}

class NativeGlobalPositioningTraceJacobianReducer {
 public:
  explicit NativeGlobalPositioningTraceJacobianReducer(
      const py::dict& snapshot,
      const bool accumulate_point_schur = true,
      const py::object& projection_direction = py::none(),
      const size_t top_k_residuals = 0,
      const py::object& pinning_target_step = py::none(),
      const std::string& schur_residual_source_filter = "")
      : snapshot_(SnapshotTableFromPy(snapshot)),
        accumulate_point_schur_(accumulate_point_schur),
        projection_direction_(ProjectionDirectionFromPy(projection_direction)),
        top_k_residuals_(top_k_residuals),
        pinning_target_step_(OptionalPositiveDoubleFromPy(
            pinning_target_step, "pinning_target_step")),
        schur_residual_source_filter_(schur_residual_source_filter) {}

  size_t AddChunk(const py::list& records, const py::list& memberships) {
    if (records.size() != memberships.size()) {
      throw std::runtime_error(
          "records and memberships must have equal length");
    }
    size_t replayed = 0;
    for (size_t idx = 0; idx < records.size(); ++idx) {
      AddResidual(py::cast<py::dict>(records[idx]),
                  py::cast<py::list>(memberships[idx]));
      ++replayed;
    }
    return replayed;
  }

  py::dict Summary(const double damping, const size_t top_k_points) const {
    py::dict result;
    for (const auto& [edge_key, accumulator] : accumulators_) {
      result[py::str(edge_key)] = EdgeSummary(accumulator,
                                              damping,
                                              top_k_points,
                                              top_k_residuals_,
                                              projection_direction_,
                                              pinning_target_step_);
    }
    return result;
  }

 private:
  void AddResidual(const py::dict& record, const py::list& memberships) {
    if (memberships.empty()) {
      return;
    }
    const py::dict attrs = py::cast<py::dict>(record[py::str("attrs")]);
    const std::vector<NativeParameterBlock> parameter_blocks =
        ParameterBlocksFromRecord(attrs);
    std::unique_ptr<ceres::CostFunction> cost_function =
        CreateCostFunctionFromRecord(attrs);
    if (cost_function == nullptr) {
      throw std::runtime_error("failed to create native replay cost function");
    }

    std::vector<Eigen::VectorXd> parameter_values;
    std::vector<const double*> parameter_ptrs;
    parameter_values.reserve(parameter_blocks.size());
    parameter_ptrs.reserve(parameter_blocks.size());
    for (const NativeParameterBlock& block : parameter_blocks) {
      parameter_values.push_back(
          SnapshotValue(snapshot_, block.kind, block.id, block.size));
      parameter_ptrs.push_back(parameter_values.back().data());
    }

    const int residual_dim = cost_function->num_residuals();
    Eigen::VectorXd residuals(residual_dim);
    std::vector<std::vector<double>> jacobian_storage;
    std::vector<double*> jacobian_ptrs;
    jacobian_storage.reserve(parameter_blocks.size());
    jacobian_ptrs.reserve(parameter_blocks.size());
    for (const NativeParameterBlock& block : parameter_blocks) {
      jacobian_storage.emplace_back(
          static_cast<size_t>(residual_dim) * block.size,
          std::numeric_limits<double>::quiet_NaN());
      jacobian_ptrs.push_back(jacobian_storage.back().data());
    }
    const bool success = cost_function->Evaluate(
        parameter_ptrs.data(), residuals.data(), jacobian_ptrs.data());
    if (!success) {
      return;
    }

    const double squared_norm = residuals.squaredNorm();
    const double raw_cost = 0.5 * squared_norm;
    const Eigen::Vector3d rho = LossRhoFromRecord(attrs, squared_norm);
    const double robust_cost = 0.5 * rho[0];
    const double rho1 = rho[1];
    const double rho2 = rho[2];

    const int frame_block_idx = FindBlock(parameter_blocks, "frame_center");
    const int point_block_idx = FindBlock(parameter_blocks, "point3D");
    std::optional<Eigen::Matrix<double, Eigen::Dynamic, 3>> frame_jacobian;
    std::optional<Eigen::Matrix<double, Eigen::Dynamic, 3>> point_jacobian;
    if (frame_block_idx >= 0 &&
        parameter_blocks[static_cast<size_t>(frame_block_idx)].size == 3) {
      frame_jacobian = Eigen::Map<
          const Eigen::
              Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
          jacobian_storage[static_cast<size_t>(frame_block_idx)].data(),
          residual_dim,
          3);
    }
    if (point_block_idx >= 0 &&
        parameter_blocks[static_cast<size_t>(point_block_idx)].size == 3) {
      point_jacobian = Eigen::Map<
          const Eigen::
              Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
          jacobian_storage[static_cast<size_t>(point_block_idx)].data(),
          residual_dim,
          3);
    }

    for (const py::handle membership_handle : memberships) {
      const py::tuple membership = py::cast<py::tuple>(membership_handle);
      if (membership.size() != 3) {
        throw std::runtime_error(
            "membership must be (edge_key, side, point_id)");
      }
      AddMembership(py::cast<std::string>(membership[0]),
                    py::cast<std::string>(membership[1]),
                    py::cast<uint64_t>(membership[2]),
                    residuals,
                    raw_cost,
                    robust_cost,
                    rho1,
                    rho2,
                    attrs,
                    frame_jacobian,
                    point_jacobian);
    }
  }

  static int FindBlock(const std::vector<NativeParameterBlock>& blocks,
                       const std::string& role) {
    for (size_t idx = 0; idx < blocks.size(); ++idx) {
      if (blocks[idx].role == role) {
        return static_cast<int>(idx);
      }
    }
    return -1;
  }

  void AddMembership(
      const std::string& edge_key,
      const std::string& side,
      const uint64_t point_id,
      const Eigen::VectorXd& residuals,
      const double raw_cost,
      const double robust_cost,
      const double rho1,
      const double rho2,
      const py::dict& attrs,
      const std::optional<Eigen::Matrix<double, Eigen::Dynamic, 3>>&
          frame_jacobian,
      const std::optional<Eigen::Matrix<double, Eigen::Dynamic, 3>>&
          point_jacobian) {
    NativeEdgeJacobianAccumulator& edge = accumulators_[edge_key];
    edge.residual_ids_selected += 1;
    edge.residuals_replayed += 1;
    edge.raw_cost_sum += raw_cost;
    edge.robust_cost_sum += robust_cost;
    edge.rho1_sum += rho1;
    edge.rho2_sum += rho2;
    edge.rho2_min = std::min(edge.rho2_min, rho2);
    edge.rho2_max = std::max(edge.rho2_max, rho2);
    if (rho1 < 0.999) {
      edge.rho1_downweighted_count += 1;
    }
    if (frame_jacobian.has_value()) {
      edge.residuals_with_frame_jacobian += 1;
      const Eigen::Vector3d gradient = frame_jacobian->transpose() * residuals;
      const Eigen::Vector3d robust_gradient = rho1 * gradient;
      if (side == "left") {
        edge.left_gradient_raw += gradient;
        edge.left_gradient_robust += robust_gradient;
      } else if (side == "right") {
        edge.right_gradient_raw += gradient;
        edge.right_gradient_robust += robust_gradient;
      } else if (side == "point") {
        // Point-only memberships contribute to the eliminated point state
        // but are intentionally not assigned to either camera block.
      } else {
        throw std::runtime_error(
            "membership side must be 'left', 'right', or 'point'");
      }
      if (side != "point") {
        AddTopOpposingResidual(edge,
                               attrs,
                               side,
                               point_id,
                               gradient,
                               robust_gradient,
                               raw_cost,
                               robust_cost,
                               rho1);
        AddTopPinningResidual(edge,
                              attrs,
                              side,
                              point_id,
                              gradient,
                              robust_gradient,
                              *frame_jacobian,
                              raw_cost,
                              robust_cost,
                              rho1,
                              rho2);
      }
    }
    if (!accumulate_point_schur_ || !frame_jacobian.has_value() ||
        !point_jacobian.has_value()) {
      return;
    }
    edge.residuals_with_frame_point_jacobians += 1;
    NativePointSchurAccumulator& point = edge.point_accumulators[point_id];
    const Eigen::Matrix3d hpp = point_jacobian->transpose() * *point_jacobian;
    const Eigen::Matrix3d cross = frame_jacobian->transpose() * *point_jacobian;
    const Eigen::Matrix3d hff = frame_jacobian->transpose() * *frame_jacobian;
    const Eigen::Vector3d frame_gradient =
        frame_jacobian->transpose() * residuals;
    const Eigen::Vector3d point_gradient =
        point_jacobian->transpose() * residuals;
    const Eigen::Matrix3d analytic_hpp =
        rho1 * hpp + 2.0 * rho2 * point_gradient * point_gradient.transpose();
    const Eigen::Matrix3d analytic_cross =
        rho1 * cross + 2.0 * rho2 * frame_gradient * point_gradient.transpose();
    const Eigen::Matrix3d analytic_hff =
        rho1 * hff + 2.0 * rho2 * frame_gradient * frame_gradient.transpose();
    point.hpp_raw += hpp;
    point.hpp_rho1 += rho1 * hpp;
    point.hpp_analytic += analytic_hpp;
    point.point_gradient_raw += point_gradient;
    point.point_gradient_robust += rho1 * point_gradient;
    const std::string source =
        attrs.contains("is_lc_observation") &&
                py::cast<bool>(attrs[py::str("is_lc_observation")])
            ? "lc"
            : "local";
    if (projection_direction_.has_value() && top_k_residuals_ > 0 &&
        side != "point" &&
        (schur_residual_source_filter_.empty() ||
         source == schur_residual_source_filter_)) {
      NativeSchurResidualRow row;
      row.residual_id = OptionalStringFromPy(attrs, "residual_id", "");
      row.side = side;
      row.frame_id = OptionalUInt64FromPy(attrs, "frame_id", 0);
      row.image_id = OptionalUInt64FromPy(attrs, "image_id", 0);
      row.point2D_idx = OptionalUInt64FromPy(attrs, "point2D_idx", 0);
      row.point3D_id = point_id;
      row.residual_type =
          OptionalStringFromPy(attrs, "residual_type", "unknown");
      row.loss_bucket = OptionalStringFromPy(attrs, "loss_bucket", "unknown");
      row.source = source;
      row.raw_cost = raw_cost;
      row.robust_cost = robust_cost;
      row.rho1 = rho1;
      row.robust_frame_gradient = rho1 * frame_gradient;
      row.robust_point_gradient = rho1 * point_gradient;
      row.analytic_cross = analytic_cross;
      row.analytic_hff = analytic_hff;
      edge.schur_residuals.push_back(row);
    }
    if (side == "left") {
      point.left_count += 1;
      point.left_raw += cross;
      point.left_rho1 += rho1 * cross;
      point.left_analytic += analytic_cross;
      point.left_hff_analytic += analytic_hff;
      point.left_gradient_raw += frame_gradient;
      point.left_gradient_robust += rho1 * frame_gradient;
    } else if (side == "right") {
      point.right_count += 1;
      point.right_raw += cross;
      point.right_rho1 += rho1 * cross;
      point.right_analytic += analytic_cross;
      point.right_hff_analytic += analytic_hff;
      point.right_gradient_raw += frame_gradient;
      point.right_gradient_robust += rho1 * frame_gradient;
    } else if (side == "point") {
      // The residual only conditions the eliminated point for this two-block
      // cut analysis. It should not add a left/right camera coupling row.
    } else {
      throw std::runtime_error(
          "membership side must be 'left', 'right', or 'point'");
    }
  }

  void AddTopOpposingResidual(NativeEdgeJacobianAccumulator& edge,
                              const py::dict& attrs,
                              const std::string& side,
                              const uint64_t point_id,
                              const Eigen::Vector3d& raw_gradient,
                              const Eigen::Vector3d& robust_gradient,
                              const double raw_cost,
                              const double robust_cost,
                              const double rho1) const {
    if (!projection_direction_.has_value() || top_k_residuals_ == 0) {
      return;
    }
    const double descent_projection =
        (-robust_gradient).dot(*projection_direction_);
    const double opposing_projection = -descent_projection;
    if (opposing_projection <= 0.0) {
      return;
    }

    NativeTopResidualRow row;
    row.residual_id = OptionalStringFromPy(attrs, "residual_id", "");
    row.side = side;
    row.frame_id = OptionalUInt64FromPy(attrs, "frame_id", 0);
    row.image_id = OptionalUInt64FromPy(attrs, "image_id", 0);
    row.point2D_idx = OptionalUInt64FromPy(attrs, "point2D_idx", 0);
    row.point3D_id = point_id;
    row.residual_type =
        OptionalStringFromPy(attrs, "residual_type", "unknown");
    row.loss_bucket = OptionalStringFromPy(attrs, "loss_bucket", "unknown");
    row.source = attrs.contains("is_lc_observation") &&
                         py::cast<bool>(attrs[py::str("is_lc_observation")])
                     ? "lc"
                     : "local";
    row.descent_projection = descent_projection;
    row.opposing_projection = opposing_projection;
    row.raw_gradient_norm = raw_gradient.norm();
    row.robust_gradient_norm = robust_gradient.norm();
    row.raw_cost = raw_cost;
    row.robust_cost = robust_cost;
    row.rho1 = rho1;
    row.raw_gradient = raw_gradient;
    row.robust_gradient = robust_gradient;

    if (edge.top_opposing_residuals.size() < top_k_residuals_) {
      edge.top_opposing_residuals.push_back(row);
      return;
    }
    const auto min_it = std::min_element(
        edge.top_opposing_residuals.begin(),
        edge.top_opposing_residuals.end(),
        [](const NativeTopResidualRow& lhs, const NativeTopResidualRow& rhs) {
          return lhs.opposing_projection < rhs.opposing_projection;
        });
    if (min_it != edge.top_opposing_residuals.end() &&
        row.opposing_projection > min_it->opposing_projection) {
      *min_it = row;
    }
  }

  void AddTopPinningResidual(
      NativeEdgeJacobianAccumulator& edge,
      const py::dict& attrs,
      const std::string& side,
      const uint64_t point_id,
      const Eigen::Vector3d& raw_gradient,
      const Eigen::Vector3d& robust_gradient,
      const Eigen::Matrix<double, Eigen::Dynamic, 3>& frame_jacobian,
      const double raw_cost,
      const double robust_cost,
      const double rho1,
      const double rho2) const {
    if (!projection_direction_.has_value() || top_k_residuals_ == 0) {
      return;
    }
    const Eigen::Matrix3d hff = frame_jacobian.transpose() * frame_jacobian;
    const Eigen::Matrix3d analytic_hff =
        rho1 * hff + 2.0 * rho2 * raw_gradient * raw_gradient.transpose();
    const double descent_projection =
        (-robust_gradient).dot(*projection_direction_);
    const double directional_curvature =
        projection_direction_->dot(analytic_hff * *projection_direction_);
    if (!std::isfinite(directional_curvature) ||
        directional_curvature <= 0.0) {
      return;
    }
    const double newton_step =
        std::abs(directional_curvature) < 1e-12
            ? std::numeric_limits<double>::quiet_NaN()
            : descent_projection / directional_curvature;

    NativeTopResidualRow row;
    row.residual_id = OptionalStringFromPy(attrs, "residual_id", "");
    row.side = side;
    row.frame_id = OptionalUInt64FromPy(attrs, "frame_id", 0);
    row.image_id = OptionalUInt64FromPy(attrs, "image_id", 0);
    row.point2D_idx = OptionalUInt64FromPy(attrs, "point2D_idx", 0);
    row.point3D_id = point_id;
    row.residual_type =
        OptionalStringFromPy(attrs, "residual_type", "unknown");
    row.loss_bucket = OptionalStringFromPy(attrs, "loss_bucket", "unknown");
    row.source = attrs.contains("is_lc_observation") &&
                         py::cast<bool>(attrs[py::str("is_lc_observation")])
                     ? "lc"
                     : "local";
    row.descent_projection = descent_projection;
    row.opposing_projection = -descent_projection;
    row.raw_gradient_norm = raw_gradient.norm();
    row.robust_gradient_norm = robust_gradient.norm();
    row.raw_cost = raw_cost;
    row.robust_cost = robust_cost;
    row.rho1 = rho1;
    row.directional_curvature = directional_curvature;
    row.newton_step_along_projection =
        std::isfinite(newton_step) ? std::optional<double>(newton_step)
                                   : std::nullopt;
    row.raw_gradient = raw_gradient;
    row.robust_gradient = robust_gradient;

    if (edge.top_pinning_residuals.size() < top_k_residuals_) {
      edge.top_pinning_residuals.push_back(row);
      return;
    }
    const auto min_it = std::min_element(
        edge.top_pinning_residuals.begin(),
        edge.top_pinning_residuals.end(),
        [](const NativeTopResidualRow& lhs, const NativeTopResidualRow& rhs) {
          return lhs.directional_curvature.value_or(0.0) <
                 rhs.directional_curvature.value_or(0.0);
        });
    if (min_it != edge.top_pinning_residuals.end() &&
        directional_curvature >
            min_it->directional_curvature.value_or(0.0)) {
      *min_it = row;
    }
  }

  static std::optional<Eigen::Vector3d> ProjectionDirectionFromPy(
      const py::object& value) {
    if (value.is_none()) {
      return std::nullopt;
    }
    const py::array_t<double, py::array::c_style | py::array::forcecast> array(
        value);
    if (array.ndim() != 1 || array.shape(0) != 3) {
      throw std::runtime_error(
          "projection_direction must be None or a length-3 vector");
    }
    Eigen::Vector3d direction(array.data()[0], array.data()[1], array.data()[2]);
    const double norm = direction.norm();
    if (!(norm > 0.0) || !std::isfinite(norm)) {
      throw std::runtime_error(
          "projection_direction must have finite nonzero norm");
    }
    return direction / norm;
  }

  static std::optional<double> OptionalPositiveDoubleFromPy(
      const py::object& value, const std::string& name) {
    if (value.is_none()) {
      return std::nullopt;
    }
    const double parsed = py::cast<double>(value);
    if (!std::isfinite(parsed) || parsed <= 0.0) {
      throw std::runtime_error(name + " must be None or finite and positive");
    }
    return parsed;
  }

  static double PinningScore(
      const std::optional<double>& curvature,
      const double descent_projection,
      const std::optional<double>& newton_step,
      const std::optional<double>& target_step) {
    if (!curvature.has_value() || !std::isfinite(*curvature) ||
        *curvature <= 0.0 || !std::isfinite(descent_projection) ||
        descent_projection <= 0.0 || !newton_step.has_value() ||
        !std::isfinite(*newton_step) || *newton_step <= 0.0) {
      return 0.0;
    }
    if (!target_step.has_value()) {
      return *curvature / std::max(std::abs(*newton_step), 1.0);
    }
    if (*newton_step >= *target_step) {
      return 0.0;
    }
    const double blocked_fraction =
        (*target_step - *newton_step) / *target_step;
    return *curvature * blocked_fraction;
  }

  static py::list TopResidualsSummary(
      const std::vector<NativeTopResidualRow>& rows) {
    std::vector<NativeTopResidualRow> sorted = rows;
    std::sort(sorted.begin(),
              sorted.end(),
              [](const NativeTopResidualRow& lhs,
                 const NativeTopResidualRow& rhs) {
                if (lhs.opposing_projection == rhs.opposing_projection) {
                  return lhs.residual_id < rhs.residual_id;
                }
                return lhs.opposing_projection > rhs.opposing_projection;
              });
    py::list result;
    for (const NativeTopResidualRow& row : sorted) {
      py::dict item;
      item["residual_id"] = row.residual_id;
      item["side"] = row.side;
      item["frame_id"] = row.frame_id;
      item["image_id"] = row.image_id;
      item["point2D_idx"] = row.point2D_idx;
      item["point3D_id"] = row.point3D_id;
      item["residual_type"] = row.residual_type;
      item["loss_bucket"] = row.loss_bucket;
      item["source"] = row.source;
      item["descent_projection"] = row.descent_projection;
      item["opposing_projection"] = row.opposing_projection;
      item["raw_gradient_norm"] = row.raw_gradient_norm;
      item["robust_gradient_norm"] = row.robust_gradient_norm;
      item["raw_cost"] = row.raw_cost;
      item["robust_cost"] = row.robust_cost;
      item["rho1"] = row.rho1;
      item["directional_curvature_analytic"] =
          row.directional_curvature.has_value()
              ? py::cast(*row.directional_curvature)
              : py::none();
      item["newton_step_along_projection"] =
          row.newton_step_along_projection.has_value()
              ? py::cast(*row.newton_step_along_projection)
              : py::none();
      item["raw_gradient"] = Vector3ToList(row.raw_gradient);
      item["robust_gradient"] = Vector3ToList(row.robust_gradient);
      result.append(item);
    }
    return result;
  }

  static py::list TopSchurPinningResidualsSummary(
      const NativeEdgeJacobianAccumulator& edge,
      const double damping,
      const size_t top_k_residuals,
      const std::optional<Eigen::Vector3d>& projection_direction,
      const std::optional<double>& pinning_target_step) {
    py::list result;
    if (!projection_direction.has_value() || top_k_residuals == 0) {
      return result;
    }
    const Eigen::Vector3d& direction = *projection_direction;
    const Eigen::Matrix3d eye = Eigen::Matrix3d::Identity();
    struct ScoredRow {
      const NativeSchurResidualRow* row = nullptr;
      double right_reduced_descent_projection = 0.0;
      std::optional<double> right_reduced_directional_curvature;
      std::optional<double> right_reduced_newton_step;
      double pinning_score = 0.0;
      Eigen::Vector3d right_reduced_gradient = Eigen::Vector3d::Zero();
    };
    std::vector<ScoredRow> scored;
    scored.reserve(edge.schur_residuals.size());
    for (const NativeSchurResidualRow& row : edge.schur_residuals) {
      const auto point_it = edge.point_accumulators.find(row.point3D_id);
      if (point_it == edge.point_accumulators.end()) {
        continue;
      }
      const NativePointSchurAccumulator& point = point_it->second;
      if (point.left_count == 0 || point.right_count == 0) {
        continue;
      }
      Eigen::FullPivLU<Eigen::Matrix3d> analytic_lu(point.hpp_analytic +
                                                    damping * eye);
      if (!analytic_lu.isInvertible()) {
        continue;
      }

      Eigen::Vector3d right_reduced_gradient;
      std::optional<double> right_reduced_directional_curvature;
      if (row.side == "right") {
        right_reduced_gradient =
            row.robust_frame_gradient -
            row.analytic_cross *
                analytic_lu.solve(point.point_gradient_robust);
        const Eigen::Matrix3d right_reduced_hessian =
            row.analytic_hff -
            row.analytic_cross *
                analytic_lu.solve(point.right_analytic.transpose());
        right_reduced_directional_curvature =
            direction.dot(right_reduced_hessian * direction);
      } else if (row.side == "left") {
        // A pre-side residual has no direct post-frame parameter block. It
        // still changes the Schur-reduced post gradient through the eliminated
        // point gradient. We deliberately do not assign a directional
        // curvature to this row: curvature attribution through Hpp is not a
        // unique per-residual decomposition.
        right_reduced_gradient =
            -point.right_analytic *
            analytic_lu.solve(row.robust_point_gradient);
        right_reduced_directional_curvature = std::nullopt;
      } else {
        continue;
      }
      const double descent_projection = (-right_reduced_gradient).dot(direction);
      if (!std::isfinite(descent_projection)) {
        continue;
      }
      if (right_reduced_directional_curvature.has_value() &&
          !std::isfinite(*right_reduced_directional_curvature)) {
        continue;
      }
      std::optional<double> newton_step;
      if (right_reduced_directional_curvature.has_value() &&
          std::abs(*right_reduced_directional_curvature) >= 1e-12) {
        newton_step =
            descent_projection / *right_reduced_directional_curvature;
      }
      const double pinning_score =
          PinningScore(right_reduced_directional_curvature,
                       descent_projection,
                       newton_step,
                       pinning_target_step);
      if (pinning_score <= 0.0) {
        continue;
      }
      scored.push_back({&row,
                        descent_projection,
                        right_reduced_directional_curvature,
                        newton_step,
                        pinning_score,
                        right_reduced_gradient});
    }
    std::sort(scored.begin(),
              scored.end(),
              [](const ScoredRow& lhs, const ScoredRow& rhs) {
                if (lhs.pinning_score == rhs.pinning_score) {
                  return lhs.row->residual_id < rhs.row->residual_id;
                }
                return lhs.pinning_score > rhs.pinning_score;
              });
    for (size_t idx = 0; idx < std::min(top_k_residuals, scored.size());
         ++idx) {
      const ScoredRow& scored_row = scored[idx];
      const NativeSchurResidualRow& row = *scored_row.row;
      py::dict item;
      item["residual_id"] = row.residual_id;
      item["side"] = row.side;
      item["frame_id"] = row.frame_id;
      item["image_id"] = row.image_id;
      item["point2D_idx"] = row.point2D_idx;
      item["point3D_id"] = row.point3D_id;
      item["residual_type"] = row.residual_type;
      item["loss_bucket"] = row.loss_bucket;
      item["source"] = row.source;
      item["raw_cost"] = row.raw_cost;
      item["robust_cost"] = row.robust_cost;
      item["rho1"] = row.rho1;
      item["right_reduced_descent_projection"] =
          scored_row.right_reduced_descent_projection;
      item["right_reduced_directional_curvature_contribution_analytic"] =
          scored_row.right_reduced_directional_curvature.has_value()
              ? py::cast(*scored_row.right_reduced_directional_curvature)
              : py::none();
      item["right_reduced_newton_step_along_projection"] =
          scored_row.right_reduced_newton_step.has_value()
              ? py::cast(*scored_row.right_reduced_newton_step)
              : py::none();
      item["pinning_score"] = scored_row.pinning_score;
      item["right_reduced_gradient"] =
          Vector3ToList(scored_row.right_reduced_gradient);
      result.append(item);
    }
    return result;
  }

  static py::dict EdgeSummary(
      const NativeEdgeJacobianAccumulator& edge,
      const double damping,
      const size_t top_k_points,
      const size_t top_k_residuals,
      const std::optional<Eigen::Vector3d>& projection_direction,
      const std::optional<double>& pinning_target_step) {
    double raw_fro_sum = 0.0;
    double rho1_fro_sum = 0.0;
    double analytic_fro_sum = 0.0;
    double raw_spectral_max = 0.0;
    double rho1_spectral_max = 0.0;
    double analytic_spectral_max = 0.0;
    Eigen::Vector3d left_reduced_gradient_analytic =
        Eigen::Vector3d::Zero();
    Eigen::Vector3d right_reduced_gradient_analytic =
        Eigen::Vector3d::Zero();
    Eigen::Matrix3d left_reduced_hessian_analytic =
        Eigen::Matrix3d::Zero();
    Eigen::Matrix3d right_reduced_hessian_analytic =
        Eigen::Matrix3d::Zero();
    uint64_t contributing_points = 0;
    uint64_t singular_points = 0;
    struct PointRow {
      uint64_t point_id;
      uint64_t left_count;
      uint64_t right_count;
      double raw_fro;
      double rho1_fro;
      double analytic_fro;
      double raw_spectral;
      double rho1_spectral;
      double analytic_spectral;
      std::optional<double> left_descent_projection;
      std::optional<double> right_descent_projection;
      std::optional<double> left_directional_curvature;
      std::optional<double> right_directional_curvature;
      std::optional<double> left_newton_step;
      std::optional<double> right_newton_step;
      double right_pinning_score = 0.0;
    };
    std::vector<PointRow> point_rows;
    const Eigen::Matrix3d eye = Eigen::Matrix3d::Identity();
    for (const auto& [point_id, point] : edge.point_accumulators) {
      if (point.left_count == 0 || point.right_count == 0) {
        continue;
      }
      contributing_points += 1;
      Eigen::FullPivLU<Eigen::Matrix3d> raw_lu(point.hpp_raw + damping * eye);
      Eigen::FullPivLU<Eigen::Matrix3d> rho1_lu(point.hpp_rho1 + damping * eye);
      Eigen::FullPivLU<Eigen::Matrix3d> analytic_lu(point.hpp_analytic +
                                                    damping * eye);
      if (!raw_lu.isInvertible() || !rho1_lu.isInvertible() ||
          !analytic_lu.isInvertible()) {
        singular_points += 1;
        continue;
      }
      const Eigen::Matrix3d raw_block =
          point.left_raw * raw_lu.solve(point.right_raw.transpose());
      const Eigen::Matrix3d rho1_block =
          point.left_rho1 * rho1_lu.solve(point.right_rho1.transpose());
      const Eigen::Matrix3d analytic_block =
          point.left_analytic *
          analytic_lu.solve(point.right_analytic.transpose());
      left_reduced_gradient_analytic +=
          point.left_gradient_robust -
          point.left_analytic * analytic_lu.solve(point.point_gradient_robust);
      right_reduced_gradient_analytic +=
          point.right_gradient_robust -
          point.right_analytic * analytic_lu.solve(point.point_gradient_robust);
      const Eigen::Vector3d point_left_reduced_gradient =
          point.left_gradient_robust -
          point.left_analytic * analytic_lu.solve(point.point_gradient_robust);
      const Eigen::Vector3d point_right_reduced_gradient =
          point.right_gradient_robust -
          point.right_analytic * analytic_lu.solve(point.point_gradient_robust);
      const Eigen::Matrix3d point_left_reduced_hessian =
          point.left_hff_analytic -
          point.left_analytic * analytic_lu.solve(point.left_analytic.transpose());
      const Eigen::Matrix3d point_right_reduced_hessian =
          point.right_hff_analytic -
          point.right_analytic * analytic_lu.solve(point.right_analytic.transpose());
      left_reduced_hessian_analytic += point_left_reduced_hessian;
      right_reduced_hessian_analytic += point_right_reduced_hessian;
      const double raw_fro = raw_block.norm();
      const double rho1_fro = rho1_block.norm();
      const double analytic_fro = analytic_block.norm();
      const double raw_spectral =
          Eigen::JacobiSVD<Eigen::Matrix3d>(raw_block).singularValues()[0];
      const double rho1_spectral =
          Eigen::JacobiSVD<Eigen::Matrix3d>(rho1_block).singularValues()[0];
      const double analytic_spectral =
          Eigen::JacobiSVD<Eigen::Matrix3d>(analytic_block).singularValues()[0];
      raw_fro_sum += raw_fro;
      rho1_fro_sum += rho1_fro;
      analytic_fro_sum += analytic_fro;
      raw_spectral_max = std::max(raw_spectral_max, raw_spectral);
      rho1_spectral_max = std::max(rho1_spectral_max, rho1_spectral);
      analytic_spectral_max =
          std::max(analytic_spectral_max, analytic_spectral);
      std::optional<double> left_descent_projection;
      std::optional<double> right_descent_projection;
      std::optional<double> left_directional_curvature;
      std::optional<double> right_directional_curvature;
      std::optional<double> left_newton_step;
      std::optional<double> right_newton_step;
      double right_pinning_score = 0.0;
      if (projection_direction.has_value()) {
        const Eigen::Vector3d& direction = *projection_direction;
        left_descent_projection =
            (-point_left_reduced_gradient).dot(direction);
        right_descent_projection =
            (-point_right_reduced_gradient).dot(direction);
        left_directional_curvature =
            direction.dot(point_left_reduced_hessian * direction);
        right_directional_curvature =
            direction.dot(point_right_reduced_hessian * direction);
        if (std::abs(*left_directional_curvature) >= 1e-12) {
          left_newton_step =
              *left_descent_projection / *left_directional_curvature;
        }
        if (std::abs(*right_directional_curvature) >= 1e-12) {
          right_newton_step =
              *right_descent_projection / *right_directional_curvature;
        }
        right_pinning_score = PinningScore(right_directional_curvature,
                                           *right_descent_projection,
                                           right_newton_step,
                                           pinning_target_step);
      }
      point_rows.push_back({point_id,
                            point.left_count,
                            point.right_count,
                            raw_fro,
                            rho1_fro,
                            analytic_fro,
                            raw_spectral,
                            rho1_spectral,
                            analytic_spectral,
                            left_descent_projection,
                            right_descent_projection,
                            left_directional_curvature,
                            right_directional_curvature,
                            left_newton_step,
                            right_newton_step,
                            right_pinning_score});
    }
    std::sort(point_rows.begin(),
              point_rows.end(),
              [](const PointRow& lhs, const PointRow& rhs) {
                if (lhs.right_pinning_score != rhs.right_pinning_score) {
                  return lhs.right_pinning_score > rhs.right_pinning_score;
                }
                if (lhs.analytic_fro == rhs.analytic_fro) {
                  return lhs.point_id < rhs.point_id;
                }
                return lhs.analytic_fro > rhs.analytic_fro;
              });
    py::list top_points;
    for (size_t idx = 0; idx < std::min(top_k_points, point_rows.size());
         ++idx) {
      const PointRow& row = point_rows[idx];
      py::dict item;
      item["point3D_id"] = row.point_id;
      item["left_residual_count"] = row.left_count;
      item["right_residual_count"] = row.right_count;
      item["raw_schur_frobenius_norm"] = row.raw_fro;
      item["rho1_schur_frobenius_norm"] = row.rho1_fro;
      item["analytic_robust_schur_frobenius_norm"] = row.analytic_fro;
      item["raw_schur_spectral_norm"] = row.raw_spectral;
      item["rho1_schur_spectral_norm"] = row.rho1_spectral;
      item["analytic_robust_schur_spectral_norm"] = row.analytic_spectral;
      if (projection_direction.has_value()) {
        item["left_reduced_descent_projection"] =
            row.left_descent_projection.has_value()
                ? py::cast(*row.left_descent_projection)
                : py::none();
        item["right_reduced_descent_projection"] =
            row.right_descent_projection.has_value()
                ? py::cast(*row.right_descent_projection)
                : py::none();
        item["left_reduced_directional_curvature_analytic"] =
            row.left_directional_curvature.has_value()
                ? py::cast(*row.left_directional_curvature)
                : py::none();
        item["right_reduced_directional_curvature_analytic"] =
            row.right_directional_curvature.has_value()
                ? py::cast(*row.right_directional_curvature)
                : py::none();
        item["left_reduced_newton_step_along_projection"] =
            row.left_newton_step.has_value() ? py::cast(*row.left_newton_step)
                                             : py::none();
        item["right_reduced_newton_step_along_projection"] =
            row.right_newton_step.has_value()
                ? py::cast(*row.right_newton_step)
                : py::none();
        item["right_pinning_score"] = row.right_pinning_score;
      }
      top_points.append(item);
    }

    py::dict summary;
    summary["residual_ids_selected"] = edge.residual_ids_selected;
    summary["residuals_replayed"] = edge.residuals_replayed;
    summary["residuals_with_frame_jacobian"] =
        edge.residuals_with_frame_jacobian;
    summary["residuals_with_frame_point_jacobians"] =
        edge.residuals_with_frame_point_jacobians;
    summary["raw_cost_sum"] = edge.raw_cost_sum;
    summary["robust_cost_sum"] = edge.robust_cost_sum;
    summary["rho1_mean"] =
        edge.residuals_replayed == 0
            ? py::none()
            : py::cast(edge.rho1_sum /
                       static_cast<double>(edge.residuals_replayed));
    summary["rho2_mean"] =
        edge.residuals_replayed == 0
            ? py::none()
            : py::cast(edge.rho2_sum /
                       static_cast<double>(edge.residuals_replayed));
    summary["rho2_min"] =
        edge.residuals_replayed == 0 ? py::none() : py::cast(edge.rho2_min);
    summary["rho2_max"] =
        edge.residuals_replayed == 0 ? py::none() : py::cast(edge.rho2_max);
    summary["rho1_downweighted_fraction"] =
        edge.residuals_replayed == 0
            ? py::none()
            : py::cast(static_cast<double>(edge.rho1_downweighted_count) /
                       static_cast<double>(edge.residuals_replayed));
    summary["left_gradient_raw_norm"] = edge.left_gradient_raw.norm();
    summary["right_gradient_raw_norm"] = edge.right_gradient_raw.norm();
    summary["net_gradient_raw_norm"] =
        (edge.left_gradient_raw + edge.right_gradient_raw).norm();
    summary["left_gradient_robust_norm"] = edge.left_gradient_robust.norm();
    summary["right_gradient_robust_norm"] = edge.right_gradient_robust.norm();
    summary["net_gradient_robust_norm"] =
        (edge.left_gradient_robust + edge.right_gradient_robust).norm();
    summary["left_gradient_raw"] = Vector3ToList(edge.left_gradient_raw);
    summary["right_gradient_raw"] = Vector3ToList(edge.right_gradient_raw);
    summary["net_gradient_raw"] =
        Vector3ToList(edge.left_gradient_raw + edge.right_gradient_raw);
    summary["left_gradient_robust"] = Vector3ToList(edge.left_gradient_robust);
    summary["right_gradient_robust"] =
        Vector3ToList(edge.right_gradient_robust);
    summary["net_gradient_robust"] =
        Vector3ToList(edge.left_gradient_robust + edge.right_gradient_robust);
    summary["left_reduced_gradient_analytic"] =
        Vector3ToList(left_reduced_gradient_analytic);
    summary["right_reduced_gradient_analytic"] =
        Vector3ToList(right_reduced_gradient_analytic);
    summary["net_reduced_gradient_analytic"] = Vector3ToList(
        left_reduced_gradient_analytic + right_reduced_gradient_analytic);
    summary["left_reduced_gradient_analytic_norm"] =
        left_reduced_gradient_analytic.norm();
    summary["right_reduced_gradient_analytic_norm"] =
        right_reduced_gradient_analytic.norm();
    summary["net_reduced_gradient_analytic_norm"] =
        (left_reduced_gradient_analytic + right_reduced_gradient_analytic)
            .norm();
    summary["left_reduced_hessian_analytic_trace"] =
        left_reduced_hessian_analytic.trace();
    summary["right_reduced_hessian_analytic_trace"] =
        right_reduced_hessian_analytic.trace();
    if (projection_direction.has_value()) {
      const Eigen::Vector3d& direction = *projection_direction;
      const double left_descent_projection =
          (-left_reduced_gradient_analytic).dot(direction);
      const double right_descent_projection =
          (-right_reduced_gradient_analytic).dot(direction);
      const double net_descent_projection =
          (-(left_reduced_gradient_analytic + right_reduced_gradient_analytic))
              .dot(direction);
      const double left_directional_curvature =
          direction.dot(left_reduced_hessian_analytic * direction);
      const double right_directional_curvature =
          direction.dot(right_reduced_hessian_analytic * direction);
      summary["left_reduced_descent_projection"] = left_descent_projection;
      summary["right_reduced_descent_projection"] = right_descent_projection;
      summary["net_reduced_descent_projection"] = net_descent_projection;
      summary["left_reduced_directional_curvature_analytic"] =
          left_directional_curvature;
      summary["right_reduced_directional_curvature_analytic"] =
          right_directional_curvature;
      summary["left_reduced_newton_step_along_projection"] =
          std::abs(left_directional_curvature) < 1e-12
              ? py::none()
              : py::cast(left_descent_projection / left_directional_curvature);
      summary["right_reduced_newton_step_along_projection"] =
          std::abs(right_directional_curvature) < 1e-12
              ? py::none()
              : py::cast(right_descent_projection /
                         right_directional_curvature);
    }
    summary["schur_contributing_point_count"] = contributing_points;
    summary["schur_singular_point_count"] = singular_points;
    summary["raw_schur_frobenius_norm_sum"] = raw_fro_sum;
    summary["rho1_schur_frobenius_norm_sum"] = rho1_fro_sum;
    summary["analytic_robust_schur_frobenius_norm_sum"] = analytic_fro_sum;
    summary["raw_schur_spectral_norm_max"] = raw_spectral_max;
    summary["rho1_schur_spectral_norm_max"] = rho1_spectral_max;
    summary["analytic_robust_schur_spectral_norm_max"] = analytic_spectral_max;
    summary["top_points_by_analytic_robust_schur"] = top_points;
    summary["top_opposing_residuals"] =
        TopResidualsSummary(edge.top_opposing_residuals);
    summary["top_pinning_residuals"] =
        TopResidualsSummary(edge.top_pinning_residuals);
    summary["top_schur_pinning_residuals"] =
        TopSchurPinningResidualsSummary(
            edge,
            damping,
            top_k_residuals,
            projection_direction,
            pinning_target_step);
    return summary;
  }

  SnapshotTable snapshot_;
  bool accumulate_point_schur_ = true;
  std::optional<Eigen::Vector3d> projection_direction_;
  size_t top_k_residuals_ = 0;
  std::optional<double> pinning_target_step_;
  std::string schur_residual_source_filter_;
  std::unordered_map<std::string, NativeEdgeJacobianAccumulator> accumulators_;

  static py::list Vector3ToList(const Eigen::Vector3d& vector) {
    py::list result;
    result.append(vector.x());
    result.append(vector.y());
    result.append(vector.z());
    return result;
  }
};

}  // namespace

void BindGlobalPositioningTraceJacobianReducer(py::module& m) {
  py::class_<NativeGlobalPositioningTraceJacobianReducer>(
      m, "_GlobalPositioningTraceJacobianReducer")
      .def(py::init<const py::dict&,
                    bool,
                    const py::object&,
                    size_t,
                    const py::object&,
                    const std::string&>(),
           "snapshot"_a,
           "accumulate_point_schur"_a = true,
           "projection_direction"_a = py::none(),
           "top_k_residuals"_a = 0,
           "pinning_target_step"_a = py::none(),
           "schur_residual_source_filter"_a = "")
      .def("add_chunk",
           &NativeGlobalPositioningTraceJacobianReducer::AddChunk,
           "records"_a,
           "memberships"_a)
      .def("summary",
           &NativeGlobalPositioningTraceJacobianReducer::Summary,
           "schur_damping"_a = 1e-8,
           "top_k_points"_a = 20);
}
