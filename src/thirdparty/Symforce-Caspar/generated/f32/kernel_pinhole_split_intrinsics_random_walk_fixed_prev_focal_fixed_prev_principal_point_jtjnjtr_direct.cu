#include "kernel_pinhole_split_intrinsics_random_walk_fixed_prev_focal_fixed_prev_principal_point_jtjnjtr_direct.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPointJtjnjtrDirectKernel(
        float* next_focal_njtr,
        unsigned int next_focal_njtr_num_alloc,
        SharedIndex* next_focal_njtr_indices,
        float* next_focal_jac,
        unsigned int next_focal_jac_num_alloc,
        float* next_principal_point_njtr,
        unsigned int next_principal_point_njtr_num_alloc,
        SharedIndex* next_principal_point_njtr_indices,
        float* next_principal_point_jac,
        unsigned int next_principal_point_jac_num_alloc,
        float* const out_next_focal_njtr,
        unsigned int out_next_focal_njtr_num_alloc,
        float* const out_next_principal_point_njtr,
        unsigned int out_next_principal_point_njtr_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[8192];

  __shared__ SharedIndex next_focal_njtr_indices_loc[1024];
  next_focal_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? next_focal_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ SharedIndex next_principal_point_njtr_indices_loc[1024];
  next_principal_point_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? next_principal_point_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
}

void PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPointJtjnjtrDirect(
    float* next_focal_njtr,
    unsigned int next_focal_njtr_num_alloc,
    SharedIndex* next_focal_njtr_indices,
    float* next_focal_jac,
    unsigned int next_focal_jac_num_alloc,
    float* next_principal_point_njtr,
    unsigned int next_principal_point_njtr_num_alloc,
    SharedIndex* next_principal_point_njtr_indices,
    float* next_principal_point_jac,
    unsigned int next_principal_point_jac_num_alloc,
    float* const out_next_focal_njtr,
    unsigned int out_next_focal_njtr_num_alloc,
    float* const out_next_principal_point_njtr,
    unsigned int out_next_principal_point_njtr_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPointJtjnjtrDirectKernel<<<
      n_blocks,
      1024>>>(next_focal_njtr,
              next_focal_njtr_num_alloc,
              next_focal_njtr_indices,
              next_focal_jac,
              next_focal_jac_num_alloc,
              next_principal_point_njtr,
              next_principal_point_njtr_num_alloc,
              next_principal_point_njtr_indices,
              next_principal_point_jac,
              next_principal_point_jac_num_alloc,
              out_next_focal_njtr,
              out_next_focal_njtr_num_alloc,
              out_next_principal_point_njtr,
              out_next_principal_point_njtr_num_alloc,
              problem_size);
}

}  // namespace caspar