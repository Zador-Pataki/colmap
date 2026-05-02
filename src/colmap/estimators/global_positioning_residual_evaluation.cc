#include "colmap/estimators/global_positioning_residual_evaluation.h"

#include "colmap/util/misc.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace colmap {

GlobalPositioningTraceResidualValues EvaluateGlobalPositioningResiduals(
    const GlobalPositioningResidualEvaluationOptions& options) {
  GlobalPositioningTraceResidualValues residual_values;
  residual_values.iteration = options.iteration;
  residual_values.residual_ids.reserve(options.replay_entries.size());
  residual_values.residual_dims.reserve(options.replay_entries.size());
  residual_values.residual_offsets.reserve(options.replay_entries.size());
  residual_values.evaluation_success.resize(options.replay_entries.size(),
                                            false);
  residual_values.raw_costs.resize(options.replay_entries.size(),
                                   std::numeric_limits<double>::quiet_NaN());
  residual_values.robust_costs.resize(options.replay_entries.size(),
                                      std::numeric_limits<double>::quiet_NaN());
  residual_values.loss_rho_values.resize(
      options.replay_entries.size() * 3,
      std::numeric_limits<double>::quiet_NaN());
  residual_values.has_raw_jacobians = options.write_raw_jacobians;
  if (options.write_raw_jacobians) {
    residual_values.parameter_block_sizes.reserve(
        options.replay_entries.size());
    residual_values.raw_jacobian_offsets.reserve(options.replay_entries.size());
    residual_values.parameter_blocks.reserve(options.replay_entries.size());
    residual_values.parameter_block_is_constant.reserve(
        options.replay_entries.size());
    residual_values.parameter_block_lower_bounds.reserve(
        options.replay_entries.size());
  }

  size_t total_scalar_residuals = 0;
  size_t total_jacobian_scalars = 0;
  for (const GlobalPositioningResidualReplayEntry& entry :
       options.replay_entries) {
    THROW_CHECK(!entry.residual_id.empty())
        << "Residual replay entry has an empty residual id.";
    THROW_CHECK(entry.cost_function != nullptr)
        << "Residual replay entry " << entry.residual_id
        << " has a null cost function.";
    THROW_CHECK_EQ(entry.residual_dimension,
                   static_cast<size_t>(entry.cost_function->num_residuals()))
        << "Residual replay entry " << entry.residual_id
        << " residual dimension does not match the Ceres cost function.";
    const std::vector<int>& cost_function_parameter_block_sizes =
        entry.cost_function->parameter_block_sizes();
    THROW_CHECK_EQ(entry.parameter_block_sizes.size(),
                   cost_function_parameter_block_sizes.size())
        << "Residual replay entry " << entry.residual_id
        << " parameter-block size count does not match the Ceres cost "
           "function.";
    THROW_CHECK_EQ(entry.parameter_blocks.size(),
                   entry.parameter_block_sizes.size())
        << "Residual replay entry " << entry.residual_id
        << " parameter-block pointer count does not match the stored block "
           "sizes.";
    THROW_CHECK_EQ(entry.parameter_block_descriptors.size(),
                   entry.parameter_block_sizes.size())
        << "Residual replay entry " << entry.residual_id
        << " parameter-block descriptor count does not match the stored block "
           "sizes.";
    for (size_t block_idx = 0; block_idx < entry.parameter_blocks.size();
         ++block_idx) {
      THROW_CHECK(entry.parameter_blocks[block_idx] != nullptr)
          << "Residual replay entry " << entry.residual_id
          << " has a null parameter block pointer at index " << block_idx
          << ".";
      THROW_CHECK_EQ(entry.parameter_block_sizes[block_idx],
                     cost_function_parameter_block_sizes[block_idx])
          << "Residual replay entry " << entry.residual_id
          << " parameter block size at index " << block_idx
          << " does not match the Ceres cost function.";
      THROW_CHECK_GT(entry.parameter_block_sizes[block_idx], 0)
          << "Residual replay entry " << entry.residual_id
          << " parameter block size at index " << block_idx
          << " must be positive.";
      THROW_CHECK(!entry.parameter_block_descriptors[block_idx].role.empty())
          << "Residual replay entry " << entry.residual_id
          << " parameter block descriptor at index " << block_idx
          << " has an empty role.";
      THROW_CHECK(!entry.parameter_block_descriptors[block_idx].kind.empty())
          << "Residual replay entry " << entry.residual_id
          << " parameter block descriptor at index " << block_idx
          << " has an empty kind.";
    }

    residual_values.residual_ids.push_back(entry.residual_id);
    residual_values.residual_dims.push_back(entry.residual_dimension);
    residual_values.residual_offsets.push_back(total_scalar_residuals);
    total_scalar_residuals += entry.residual_dimension;
    if (options.write_raw_jacobians) {
      std::vector<size_t> parameter_block_sizes;
      std::vector<size_t> raw_jacobian_offsets;
      std::vector<bool> parameter_block_is_constant;
      std::vector<std::vector<double>> parameter_block_lower_bounds;
      parameter_block_sizes.reserve(entry.parameter_block_sizes.size());
      raw_jacobian_offsets.reserve(entry.parameter_block_sizes.size());
      parameter_block_is_constant.reserve(entry.parameter_block_sizes.size());
      parameter_block_lower_bounds.reserve(entry.parameter_block_sizes.size());
      for (size_t block_idx = 0; block_idx < entry.parameter_block_sizes.size();
           ++block_idx) {
        const int parameter_block_size = entry.parameter_block_sizes[block_idx];
        parameter_block_sizes.push_back(
            static_cast<size_t>(parameter_block_size));
        raw_jacobian_offsets.push_back(total_jacobian_scalars);
        parameter_block_is_constant.push_back(
            options.problem.IsParameterBlockConstant(
                entry.parameter_blocks[block_idx]));
        std::vector<double> lower_bounds;
        lower_bounds.reserve(static_cast<size_t>(parameter_block_size));
        for (int parameter_idx = 0; parameter_idx < parameter_block_size;
             ++parameter_idx) {
          lower_bounds.push_back(options.problem.GetParameterLowerBound(
              entry.parameter_blocks[block_idx], parameter_idx));
        }
        parameter_block_lower_bounds.push_back(std::move(lower_bounds));
        total_jacobian_scalars += entry.residual_dimension *
                                  static_cast<size_t>(parameter_block_size);
      }
      residual_values.parameter_block_sizes.push_back(
          std::move(parameter_block_sizes));
      residual_values.raw_jacobian_offsets.push_back(
          std::move(raw_jacobian_offsets));
      residual_values.parameter_blocks.push_back(
          entry.parameter_block_descriptors);
      residual_values.parameter_block_is_constant.push_back(
          std::move(parameter_block_is_constant));
      residual_values.parameter_block_lower_bounds.push_back(
          std::move(parameter_block_lower_bounds));
    }
  }
  residual_values.raw_residuals.resize(
      total_scalar_residuals, std::numeric_limits<double>::quiet_NaN());
  if (options.write_raw_jacobians) {
    residual_values.raw_jacobians.resize(
        total_jacobian_scalars, std::numeric_limits<double>::quiet_NaN());
  }

  for (size_t entry_idx = 0; entry_idx < options.replay_entries.size();
       ++entry_idx) {
    const GlobalPositioningResidualReplayEntry& entry =
        options.replay_entries[entry_idx];
    std::vector<double> raw_jacobian_workspace;
    std::vector<double*> raw_jacobian_blocks;
    if (options.write_raw_jacobians) {
      size_t workspace_offset = 0;
      for (const int parameter_block_size : entry.parameter_block_sizes) {
        workspace_offset += entry.residual_dimension *
                            static_cast<size_t>(parameter_block_size);
      }
      raw_jacobian_workspace.assign(workspace_offset,
                                    std::numeric_limits<double>::quiet_NaN());
      raw_jacobian_blocks.reserve(entry.parameter_block_sizes.size());
      workspace_offset = 0;
      for (const int parameter_block_size : entry.parameter_block_sizes) {
        raw_jacobian_blocks.push_back(raw_jacobian_workspace.data() +
                                      workspace_offset);
        workspace_offset += entry.residual_dimension *
                            static_cast<size_t>(parameter_block_size);
      }
    }

    double* raw_residuals = residual_values.raw_residuals.data() +
                            residual_values.residual_offsets[entry_idx];
    const bool evaluation_success = entry.cost_function->Evaluate(
        entry.parameter_blocks.data(),
        raw_residuals,
        options.write_raw_jacobians ? raw_jacobian_blocks.data() : nullptr);
    residual_values.evaluation_success[entry_idx] = evaluation_success;
    if (!evaluation_success) {
      continue;
    }
    if (options.write_raw_jacobians) {
      for (size_t block_idx = 0; block_idx < entry.parameter_block_sizes.size();
           ++block_idx) {
        const size_t jacobian_size =
            entry.residual_dimension *
            static_cast<size_t>(entry.parameter_block_sizes[block_idx]);
        std::copy_n(
            raw_jacobian_blocks[block_idx],
            jacobian_size,
            residual_values.raw_jacobians.data() +
                residual_values.raw_jacobian_offsets[entry_idx][block_idx]);
      }
    }

    double squared_norm = 0.0;
    for (size_t residual_idx = 0; residual_idx < entry.residual_dimension;
         ++residual_idx) {
      squared_norm += raw_residuals[residual_idx] * raw_residuals[residual_idx];
    }

    const double raw_cost = 0.5 * squared_norm;
    residual_values.raw_costs[entry_idx] = raw_cost;
    double rho[3] = {squared_norm, 1.0, 0.0};
    if (entry.loss_function != nullptr) {
      entry.loss_function->Evaluate(squared_norm, rho);
    }
    residual_values.loss_rho_values[3 * entry_idx] = rho[0];
    residual_values.loss_rho_values[3 * entry_idx + 1] = rho[1];
    residual_values.loss_rho_values[3 * entry_idx + 2] = rho[2];
    residual_values.robust_costs[entry_idx] = 0.5 * rho[0];
  }

  return residual_values;
}

}  // namespace colmap
